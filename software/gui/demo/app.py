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
import logging
import sys
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

from software.inference.demo_runners import (  # noqa: E402
    HDLRunner,
    HLSStubRunner,
    RunnerResult,
    TFLiteRunner,
    make_ground_truth_png,
)

logger = logging.getLogger("demo_gui")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

STATIC_DIR = Path(__file__).parent / "static"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------- runtime state


class DemoState:
    """Holds the loaded runners and the image directory for request handling."""

    def __init__(
        self,
        tflite: Optional[TFLiteRunner],
        hdl: Optional[HDLRunner],
        hls: HLSStubRunner,
        image_dir: Path,
    ):
        self.tflite = tflite
        self.hdl = hdl
        self.hls = hls
        self.image_dir = image_dir

    def list_images(self) -> list[str]:
        """Sorted list of filenames in ``image_dir`` with supported extensions."""
        if not self.image_dir.is_dir():
            return []
        return sorted(
            p.name
            for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )

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
    """Encode a RunnerResult as a JSON-friendly dict with base64 PNG."""
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
            "total_ms": None,
            "cycles": None,
            "cycle_time_ms": None,
        },
    }


@app.post("/api/run/{name}")
async def run_inference(name: str):
    """Run all three runners on ``name`` and return their outputs."""
    assert state is not None
    path = state.resolve_image(name)
    logger.info("inference request: %s", path)

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
            results["tflite"] = _result_to_json(state.tflite.run(path))
        except Exception as exc:  # noqa: BLE001
            logger.exception("tflite runner failed")
            results["tflite"] = _error_panel("tflite", str(exc))

    # HDL
    if state.hdl is None:
        results["hdl"] = _error_panel("hdl", "HDL runner not loaded (overlay failed?)")
    else:
        try:
            results["hdl"] = _result_to_json(state.hdl.run(path))
        except Exception as exc:  # noqa: BLE001
            logger.exception("hdl runner failed")
            results["hdl"] = _error_panel("hdl", str(exc))

    # HLS (stub never fails)
    results["hls"] = _result_to_json(state.hls.run(path))

    return JSONResponse(
        {
            "image": name,
            "ground_truth_png_b64": gt_b64,
            "results": results,
        }
    )


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
