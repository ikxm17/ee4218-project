"""FastAPI demo app: TinyissimoYOLO inference comparison GUI.

Serves a browser-based 4-panel comparison (hero image + ground-truth panel
+ TFLite + HDL + HLS-stub) plus a per-runner timing breakdown. Intended to
run on the Kria board under sudo so PYNQ can program the FPGA.

Usage (on the board, from the repo root)::

    echo asdfzxcv | sudo -S XILINX_XRT=/usr \\
        /opt/ee4218/ee4218-venv/bin/python3 \\
        software/gui/demo/app.py \\
        --bitstream hardware/output/preserved/playground_FIXED_5e86ce6c/playground_FIXED_5e86ce6c.bit \\
        --model-path software/models/tflite/tinyissimo_ptq_full_integer_quant.tflite \\
        --image-dir software/models/demo_images \\
        --host 0.0.0.0 --port 8000

Then from a laptop on the same LAN: ``http://kria-01:8000``.

Endpoints
---------
* ``GET  /``                 → serves the static frontend
* ``GET  /api/images``       → JSON list of selectable image filenames
* ``GET  /api/image/{name}`` → raw image bytes for the hero view
* ``POST /api/run/{name}``   → runs all three runners on the selected image,
                               returns JSON with per-runner base64 PNG and
                               timing dicts

The overlay and TFLite interpreter are loaded once at startup via
``lifespan``; per-click latency is just inference + post-processing.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Make the ``software`` package importable when this script is invoked
# directly (``python software/gui/demo/app.py``) rather than as a module.
# ``parents[3]`` maps software/gui/demo/app.py → repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from software.inference.demo_runners import (  # noqa: E402
    HDLRunner,
    HLSRunner,
    HLSStubRunner,
    MODEL_INFO,
    RunnerResult,
    TFLiteRunner,
    make_ground_truth_png,
    repostprocess_boxes_only,
)
from software.overlay.drivers.tinyissimoyolo_accelerator import (  # noqa: E402
    CLASS_COLORS,
    CLASS_NAMES,
)

# Defaults match the runner defaults — single source of truth for both the
# Pydantic validators below and the /api/config response so the frontend
# slider initial values can never disagree with the server defaults.
DEFAULT_CONF_THRESH = 0.3
DEFAULT_NMS_THRESH = 0.45


class RunRequest(BaseModel):
    """Threshold parameters accepted by /api/run and /api/repostprocess."""

    conf_thresh: float = Field(DEFAULT_CONF_THRESH, ge=0.0, le=1.0)
    nms_thresh: float = Field(DEFAULT_NMS_THRESH, ge=0.0, le=1.0)

logger = logging.getLogger("demo_gui")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

STATIC_DIR = Path(__file__).parent / "static"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------- runtime state


class DemoState:
    """Holds the loaded runners and the image directory for request handling.

    Also owns the per-image raw-tensor cache that powers live slider updates
    via /api/repostprocess. The cache is keyed by image filename and holds
    one entry per runner that successfully completed a full ``run()``. Each
    entry stores the raw output tensor (8×8×67 float32, ~17 KB), the
    preprocessed RGB array (256×256×3 uint8, ~200 KB), and the original
    timings dict so live re-postprocess responses can keep pre-process /
    inference cells frozen at their original values.
    """

    def __init__(
        self,
        tflite: Optional[TFLiteRunner],
        hdl: Optional[HDLRunner],
        hls: HLSRunner | HLSStubRunner,
        image_dir: Path,
    ):
        self.tflite = tflite
        self.hdl = hdl
        self.hls = hls
        self.image_dir = image_dir
        # image_name -> runner_name -> {raw_tensor, preproc_arr, full_timings}
        self._cache: dict[str, dict[str, dict]] = {}
        self._cache_lock = threading.Lock()

    def list_images(self) -> list[dict]:
        """Return ``[{name, label, categories, source}, ...]`` for the dropdown.

        Reads ``_manifest.json`` (produced by the COCO scraper in scripts/)
        for category metadata and pretty labels. Files present on disk but
        missing from the manifest fall through to a "loose" group so
        reference_image.jpg etc. still appear in the picker.
        """
        if not self.image_dir.is_dir():
            return []
        entries: list[dict] = []
        seen: set[str] = set()
        manifest_path = self.image_dir / "_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("could not parse manifest: %s", exc)
                manifest = {}
            for img in manifest.get("images", []):
                name = img.get("file_name")
                if not name or not (self.image_dir / name).is_file():
                    continue
                cats = img.get("categories", [])
                source = img.get("source", "manifest")
                image_id = img.get("image_id")
                # Format the label according to the source type. COCO
                # entries get "{Categories} — coco-{id}"; local references
                # (no image_id) get a humanised filename so the dropdown
                # never shows "Unlabeled — coco-None".
                if source == "coco_val2017" and image_id is not None:
                    pretty_cats = ", ".join(c.title() for c in cats) or "Unlabeled"
                    label = f"{pretty_cats} — coco-{image_id}"
                else:
                    label = Path(name).stem.replace("_", " ").title()
                entries.append(
                    {
                        "name": name,
                        "label": label,
                        "categories": cats,
                        "source": source,
                    }
                )
                seen.add(name)
        # Sweep up loose files (e.g. reference_image.jpg) — anything in the
        # directory with a supported extension that the manifest didn't claim.
        for p in sorted(self.image_dir.iterdir()):
            if (
                p.is_file()
                and p.suffix.lower() in SUPPORTED_EXTS
                and p.name not in seen
                and not p.name.startswith("_")
            ):
                entries.append(
                    {
                        "name": p.name,
                        "label": p.stem.replace("_", " ").title(),
                        "categories": [],
                        "source": "loose",
                    }
                )
        return entries

    def cache_run(
        self, image_name: str, runner_name: str, result: RunnerResult
    ) -> None:
        """Stash the raw tensor + preprocessed array for live re-postprocess.

        No-op for the stub runner (which has nothing to cache). Thread-safe
        for concurrent FastAPI handlers.
        """
        if result.raw_tensor is None or result.preproc_arr is None:
            return
        with self._cache_lock:
            self._cache.setdefault(image_name, {})[runner_name] = {
                "raw_tensor": result.raw_tensor,
                "preproc_arr": result.preproc_arr,
                "full_timings": dict(result.timings),
            }

    def get_cached(self, image_name: str) -> Optional[dict]:
        """Return the cache entries for ``image_name`` or None if not cached."""
        with self._cache_lock:
            return self._cache.get(image_name)

    def invalidate_cache(self, image_name: Optional[str] = None) -> None:
        """Drop a single image's entries (or the whole cache if name is None)."""
        with self._cache_lock:
            if image_name is None:
                self._cache.clear()
            else:
                self._cache.pop(image_name, None)

    def resolve_image(self, name: str) -> Path:
        """Resolve an image name safely inside ``image_dir`` — no path traversal."""
        # Reject absolute paths, parent refs, and slashes
        if "/" in name or "\\" in name or ".." in name or Path(name).is_absolute():
            raise HTTPException(status_code=400, detail="invalid image name")
        path = (self.image_dir / name).resolve()
        try:
            path.relative_to(self.image_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="image outside image-dir")
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"image not found: {name}")
        if path.suffix.lower() not in SUPPORTED_EXTS:
            raise HTTPException(status_code=400, detail="unsupported file type")
        return path


# state is populated by ``lifespan`` at startup; request handlers read from it.
state: Optional[DemoState] = None

# Parsed CLI args — populated in ``main`` before ``uvicorn.run``.
_args: Optional[argparse.Namespace] = None


# ------------------------------------------------------------------- app


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load heavy runners once at server startup.

    TFLite and HDL are loaded separately so the demo can still start if one
    fails (e.g. running on a dev machine without pynq) — the failed runner
    simply returns an error for its panel instead of crashing the server.
    """
    global state
    assert _args is not None, "CLI args must be parsed before lifespan runs"

    tflite_runner: Optional[TFLiteRunner] = None
    hdl_runner: Optional[HDLRunner] = None

    try:
        logger.info("loading tflite model: %s", _args.model_path)
        tflite_runner = TFLiteRunner(_args.model_path)
        logger.info("tflite runner ready")
    except Exception as exc:  # noqa: BLE001
        logger.exception("failed to load tflite runner: %s", exc)

    try:
        logger.info("loading bitstream: %s", _args.bitstream)
        hdl_runner = HDLRunner(_args.bitstream)
        logger.info("hdl runner ready")
    except Exception as exc:  # noqa: BLE001
        logger.exception("failed to load hdl runner: %s", exc)

    # HLS runner shares the HDL runner's overlay/driver — there is exactly
    # one bitstream load and one driver instance for both engines, with
    # the engine selected per-run by configure(engine=N) before start().
    # Falls back to the stub if HDL itself failed to load (no overlay to
    # share) so the GUI's 4-panel layout never breaks on a dev machine.
    hls_runner: HLSRunner | HLSStubRunner
    if hdl_runner is not None:
        try:
            hls_runner = HLSRunner(hdl_runner.driver)
            logger.info("hls runner ready (sharing overlay with HDL)")
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed to load hls runner: %s", exc)
            hls_runner = HLSStubRunner()
    else:
        logger.warning("HDL runner failed to load — HLS will use stub")
        hls_runner = HLSStubRunner()

    state = DemoState(
        tflite=tflite_runner,
        hdl=hdl_runner,
        hls=hls_runner,
        image_dir=Path(_args.image_dir).resolve(),
    )
    logger.info("image_dir=%s (%d files)", state.image_dir, len(state.list_images()))
    yield
    logger.info("demo server stopping")


app = FastAPI(title="TinyissimoYOLO Demo GUI", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/images")
async def list_images():
    assert state is not None
    return JSONResponse({"images": state.list_images()})


@app.get("/api/image/{name}")
async def get_image(name: str):
    """Return the raw image bytes so the browser can render the hero view."""
    assert state is not None
    path = state.resolve_image(name)
    # Infer content type from suffix; browser handles the rest.
    suffix = path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
    }.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media_type)


def _result_to_json(result: RunnerResult) -> dict:
    """Encode a RunnerResult as a JSON-friendly dict with base64 PNG.

    The ``raw_tensor`` and ``preproc_arr`` fields are stripped here — they
    are numpy arrays that can't be JSON-encoded and are only used by the
    in-process cache (see ``DemoState.cache_run``).
    """
    return {
        "runner": result.runner,
        "boxes": result.boxes,
        "scores": result.scores,
        "class_ids": result.class_ids,
        "annotated_png_b64": base64.b64encode(result.annotated_png).decode("ascii"),
        "timings": result.timings,
    }


def _error_panel(runner: str, message: str) -> dict:
    return {
        "runner": runner,
        "error": message,
        "boxes": [],
        "scores": [],
        "class_ids": [],
        "annotated_png_b64": None,
        "timings": {
            "preprocess_ms": None,
            "inference_ms": None,
            "postprocess_ms": None,
            "render_ms": None,
            "total_ms": None,
            "cycles": None,
            "cycle_time_ms": None,
            "host_walltime_ms": None,
            "conf_thresh": None,
            "nms_thresh": None,
        },
    }


def _bgr_to_rgb(bgr: tuple) -> tuple:
    """Swap a BGR triple to RGB. The driver stores CLASS_COLORS as BGR
    (OpenCV convention); the GUI renders with PIL (RGB), so the chip
    backgrounds in the legend must use the swapped values to match what
    ``_draw_detections`` actually paints onto the bbox label bars.
    """
    return (bgr[2], bgr[1], bgr[0])


def _build_model_info() -> dict:
    """Return a deep copy of MODEL_INFO with quantization fields filled in.

    The static facts come from ``MODEL_INFO`` in demo_runners.py; the
    dynamic quantization scale/zero-point strings come from the live tflite
    interpreter via ``TFLiteRunner.describe_quant``. If the TFLite runner
    failed to load (e.g. dev machine without tflite_runtime), the
    quantization fields stay None and the frontend can show "(unavailable)".
    """
    info = copy.deepcopy(MODEL_INFO)
    if state is not None and state.tflite is not None:
        try:
            quant = state.tflite.describe_quant()
            info["quantization"]["input_quant"] = quant["input_quant"]
            info["quantization"]["output_quant"] = quant["output_quant"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not describe TFLite quantization: %s", exc)
    return info


@app.get("/api/config")
async def get_config():
    """Return the configuration the frontend needs at page load.

    Single round-trip on page load: class list + colors + threshold
    defaults + model architecture/quantization/hardware info. The frontend
    uses this to populate the legend chips, the model-info panel, and the
    initial slider positions — no values are hardcoded in JS.
    """
    assert state is not None
    return JSONResponse(
        {
            "class_names": [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())],
            "class_colors_rgb": [
                _bgr_to_rgb(CLASS_COLORS[i])
                for i in sorted(CLASS_COLORS.keys())
            ],
            "default_conf_thresh": DEFAULT_CONF_THRESH,
            "default_nms_thresh": DEFAULT_NMS_THRESH,
            "input_size": 256,
            "model_info": _build_model_info(),
        }
    )


@app.post("/api/run/{name}")
async def run_inference(name: str, body: RunRequest = RunRequest()):
    """Run all three runners on ``name`` and return their outputs.

    Threshold parameters come from the JSON request body (validated by
    Pydantic). After a successful runner result, the raw output tensor and
    preprocessed array are cached so the live ``/api/repostprocess``
    endpoint can re-filter on slider drags without re-running inference.
    """
    assert state is not None
    path = state.resolve_image(name)
    logger.info(
        "inference request: %s (conf=%.2f, nms=%.2f)",
        path,
        body.conf_thresh,
        body.nms_thresh,
    )

    # Unannotated 256×256 view for the "ground truth" panel. Generated once
    # per request so the frontend sees exactly the same pixel stretch the
    # runner panels use.
    try:
        gt_png = make_ground_truth_png(path)
        gt_b64 = base64.b64encode(gt_png).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        logger.exception("ground-truth render failed")
        gt_b64 = None

    results: dict[str, dict] = {}

    # TFLite
    if state.tflite is None:
        results["tflite"] = _error_panel("tflite", "TFLite runner not loaded")
    else:
        try:
            tflite_res = state.tflite.run(
                path, conf_thresh=body.conf_thresh, nms_thresh=body.nms_thresh
            )
            state.cache_run(name, "tflite", tflite_res)
            results["tflite"] = _result_to_json(tflite_res)
        except Exception as exc:  # noqa: BLE001
            logger.exception("tflite runner failed")
            results["tflite"] = _error_panel("tflite", str(exc))

    # HDL
    if state.hdl is None:
        results["hdl"] = _error_panel("hdl", "HDL runner not loaded (overlay failed?)")
    else:
        try:
            hdl_res = state.hdl.run(
                path, conf_thresh=body.conf_thresh, nms_thresh=body.nms_thresh
            )
            state.cache_run(name, "hdl", hdl_res)
            results["hdl"] = _result_to_json(hdl_res)
        except Exception as exc:  # noqa: BLE001
            logger.exception("hdl runner failed")
            results["hdl"] = _error_panel("hdl", str(exc))

    # HLS — shares the HDL overlay/driver; configure(engine=1) picks the
    # HLS engine for this run.  Cached like HDL so live slider updates
    # re-postprocess both panels from their own raw tensors.
    try:
        hls_res = state.hls.run(
            path, conf_thresh=body.conf_thresh, nms_thresh=body.nms_thresh
        )
        state.cache_run(name, "hls", hls_res)
        results["hls"] = _result_to_json(hls_res)
    except Exception as exc:  # noqa: BLE001
        logger.exception("hls runner failed")
        results["hls"] = _error_panel("hls", str(exc))

    return JSONResponse(
        {
            "image": name,
            "ground_truth_png_b64": gt_b64,
            "results": results,
        }
    )


@app.post("/api/repostprocess/{name}")
async def repostprocess(name: str, body: RunRequest = RunRequest()):
    """Re-run only postprocessing on cached tensors for live slider updates.

    This endpoint never touches the TFLite interpreter or the FPGA — it
    operates entirely on the cached numpy arrays from the most recent full
    ``/api/run`` for this image. Threshold changes don't affect inference
    output, only how it's filtered, so re-running the cheap post_process
    + nms stages (<1 ms per runner) gives true live interactivity without
    serialising slider drags on the single-instance accelerator.

    Skips PIL drawing and PNG encoding — the browser redraws each panel
    on a cached Canvas overlay using the returned boxes/scores/class_ids.
    That removes ~50 ms of per-tick latency that the live slider drags
    cannot afford.

    Returns 404 if no full run has been cached for this image — the
    frontend should fall back to "click Run Inference first".
    """
    assert state is not None
    state.resolve_image(name)  # path-traversal validation

    cached = state.get_cached(name)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail="no cached run for this image — click Run Inference first",
        )

    results: dict[str, dict] = {}
    for runner_key, entry in cached.items():
        try:
            boxes, scores, class_ids, post_ms = repostprocess_boxes_only(
                entry["raw_tensor"],
                body.conf_thresh,
                body.nms_thresh,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("repostprocess failed for %s", runner_key)
            results[runner_key] = _error_panel(runner_key, str(exc))
            continue

        # Reconstruct the timings dict by overlaying the new post_ms on the
        # cached full-run timings. Pre-process and inference cells stay
        # frozen at their original values — visually obvious that those
        # didn't re-run. End-to-End uses the canonical inference time
        # (which is cycle_time_ms for HDL, invoke wall-clock for TFLite —
        # the runners already store the canonical value in inference_ms).
        new_timings = dict(entry["full_timings"])
        new_timings["postprocess_ms"] = post_ms
        pre = new_timings.get("preprocess_ms") or 0.0
        inf = new_timings.get("inference_ms") or 0.0
        new_timings["total_ms"] = pre + inf + post_ms
        new_timings["conf_thresh"] = body.conf_thresh
        new_timings["nms_thresh"] = body.nms_thresh

        results[runner_key] = {
            "runner": runner_key,
            "boxes": list(boxes),
            "scores": [float(s) for s in scores],
            "class_ids": [int(c) for c in class_ids],
            # No annotated_png_b64 — client draws via Canvas from the
            # cached ground-truth image and the boxes above.
            "annotated_png_b64": None,
            "timings": new_timings,
        }

    return JSONResponse({"image": name, "results": results})


# Serve static assets from /static/... so index.html can reference them
# with stable paths. Mounted late so it doesn't shadow API routes.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------- entrypoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bitstream", type=Path, required=True,
        help="Path to the .bit file with the HDL accelerator",
    )
    parser.add_argument(
        "--model-path", type=Path, required=True,
        help="Path to the TFLite model (int8 or float)",
    )
    parser.add_argument(
        "--image-dir", type=Path, required=True,
        help="Directory of selectable demo images",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Bind address (default 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Bind port (default 8000)",
    )
    return parser.parse_args()


def main():
    global _args
    _args = parse_args()

    import uvicorn  # noqa: WPS433

    uvicorn.run(app, host=_args.host, port=_args.port, log_level="info")


if __name__ == "__main__":
    main()
