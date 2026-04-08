"""Unified inference runner interface for the demo GUI.

Exposes three runner classes with a common `.run(image_path) -> RunnerResult`
contract so the FastAPI demo layer can iterate them uniformly:

- ``TFLiteRunner`` — tflite_runtime baseline on the PS
- ``HDLRunner``   — PYNQ Overlay + TinyissimoYoloAcceleratorDriver
- ``HLSStubRunner`` — placeholder for the HLS accelerator runner (TBD)

Design notes
------------
* Model / overlay are loaded once in ``__init__`` so per-click latency is just
  the inference + post-processing, not interpreter allocation and bitstream load.
* Canonical post-processing comes from
  ``software.overlay.drivers.tinyissimoyolo_accelerator`` — both runners call
  the same ``post_process``/``nms`` to avoid drift between the TFLite baseline
  and the HDL path.
* ``HDLRunner`` drives the accelerator primitives directly (not ``driver.run``)
  so the wall-clock timing can be split cleanly into "inference" vs
  "postprocess". It also reports ``cycles`` (from register 0x00C) and the
  cycle-derived compute time at 100 MHz.
* ``pynq`` and ``tflite_runtime`` are imported lazily so this module can be
  imported on a dev machine for linting without either installed.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from software.overlay.drivers.tinyissimoyolo_accelerator import (
    CLASS_COLORS,
    CLASS_NAMES,
    TinyissimoYoloAcceleratorDriver,
    nms,
    post_process,
)

INPUT_SIZE = 256
PL_CLOCK_HZ = 100_000_000  # 100 MHz PL clock → 10 ns / cycle

# Static facts about the model that the demo GUI surfaces in its model-info
# panel. Mirrors notes/project/model-architecture.md — if the model is ever
# rebuilt with different shapes / parameter counts, both files must be
# updated together. The quantization scale/zero-point fields are filled at
# request time by app.py from the live tflite interpreter so they always
# match the actual loaded model.
MODEL_INFO = {
    "architecture": {
        "model": "TinyissimoYOLO v1-small",
        "family": "Purely sequential CNN, no skip connections (Conv+BN+SiLU + MaxPool only)",
        "input_shape": "1 × 256 × 256 × 3 (NHWC, uint8)",
        "output_shape": "1 × 8 × 8 × 67 (8×8 grid, 64 DFL + 3 class logits per cell)",
        "stride": "32 px (5 × 2-stride max-pools: 256→128→64→32→16→8)",
        "param_count": "~401K (~401KB at int8)",
        "layers": 17,
        "detect_head": "Parallel branches: cv2 (box regression, DFL 16-bin) + cv3 (classification, sigmoid)",
        "classes": ["chair", "bowl", "cup"],
        "training_data": "COCO val2017 subset, filtered to {chair, bowl, cup}, min area 2000 px²",
        "doc_ref": "notes/project/model-architecture.md",
    },
    "quantization": {
        "scheme": "Full int8 PTQ (TFLite)",
        # input_quant / output_quant filled at runtime by app.py from
        # interpreter.get_input_details()/get_output_details()
        "input_quant": None,
        "output_quant": None,
        "calibration": "Post-training quantization on a COCO val2017 subset",
    },
    "hardware": {
        "accelerator": "tinyissimoyolo_accel — custom RTL with AXI-Lite control and AXI-Lite pixel input",
        "pl_clock_hz": PL_CLOCK_HZ,
        "pixel_transport": "AXI-Lite FIFO, 65 536 writes per inference (one per pixel)",
        "cycle_counter_addr": "0x00C",
        "output_layout": "URAM channel-group-major → unpacked to (8, 8, 67) tensor at scale ≈ 0.10655, zero-point 10",
    },
}


# Try to load a real TrueType font for bbox labels. DejaVu Sans Bold is
# almost always present on Debian/Ubuntu/PetaLinux; fall back to PIL's
# default bitmap font if it isn't.
try:
    _LABEL_FONT = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11
    )
except (OSError, IOError):
    _LABEL_FONT = ImageFont.load_default()


@dataclass
class RunnerResult:
    """Uniform output produced by every runner.

    ``boxes`` / ``scores`` / ``class_ids`` are native Python lists (not numpy)
    because the canonical ``post_process`` returns lists and the FastAPI layer
    will JSON-encode them. ``annotated_png`` is a 256×256 PNG with bboxes
    drawn, suitable for base64-encoding into the response. ``timings`` uses
    ``None`` for metrics that do not apply to a given runner (e.g. TFLite has
    no cycle count).

    ``raw_tensor`` / ``preproc_arr`` are non-JSON-serialisable numpy arrays
    held for the live re-postprocess cache (see DemoState in app.py). The
    FastAPI response encoder strips them out before JSON encoding. They are
    ``None`` for the stub runner which has nothing to cache.
    """

    runner: str
    boxes: list
    scores: list
    class_ids: list
    annotated_png: bytes
    timings: dict
    raw_tensor: Optional[np.ndarray] = None
    preproc_arr: Optional[np.ndarray] = None


# --------------------------------------------------------------------- helpers


def _ms(dt_seconds: float) -> float:
    return dt_seconds * 1000.0


def _load_and_preprocess(image_path) -> tuple[np.ndarray, float]:
    """Return ``(uint8 (256,256,3) RGB array, preprocess_ms)``.

    Matches the preprocessing in ``run_inference.py:119`` and
    ``run_inference_hdl.py:42-43`` so all three runners see the exact same
    tensor. Resize is the plain PIL bilinear stretch — no letterbox — because
    the model was trained on stretched inputs.
    """
    t0 = time.perf_counter()
    img = Image.open(image_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.array(img, dtype=np.uint8)
    return arr, _ms(time.perf_counter() - t0)


def _draw_detections(
    rgb_256: np.ndarray,
    boxes: list,
    scores: list,
    class_ids: list,
) -> bytes:
    """Draw bboxes + class labels onto a 256×256 RGB array and return PNG bytes.

    ``CLASS_COLORS`` in the driver is stored BGR (OpenCV convention); PIL uses
    RGB, so we swap per class. The label bar uses ``draw.textbbox()`` for
    exact pixel-width measurement (no magic char-width guess), the polished
    DejaVuSans-Bold TrueType font when available, and 2 px horizontal /
    1 px vertical padding around the measured text.
    """
    img = Image.fromarray(rgb_256, mode="RGB").copy()
    draw = ImageDraw.Draw(img)

    pad_x = 2
    pad_y = 1

    for (x, y, w, h), score, cls in zip(boxes, scores, class_ids):
        cls_i = int(cls)
        bgr = CLASS_COLORS.get(cls_i, (255, 255, 255))
        rgb = (bgr[2], bgr[1], bgr[0])

        # Clamp to image in case post_process returned out-of-bounds coords
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(INPUT_SIZE - 1, int(x + w))
        y2 = min(INPUT_SIZE - 1, int(y + h))
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=2)

        label = f"{CLASS_NAMES.get(cls_i, f'cls{cls_i}')} {score:.2f}"

        # Measure label exactly. textbbox returns (l, t, r, b) for the
        # tightest bounding box around the rendered glyphs at origin (0, 0).
        try:
            l, t, r, b = draw.textbbox((0, 0), label, font=_LABEL_FONT)
            text_w = r - l
            text_h = b - t
        except AttributeError:
            # Fallback for very old PIL versions without textbbox
            text_w, text_h = draw.textsize(label, font=_LABEL_FONT)

        bar_w = text_w + 2 * pad_x
        bar_h = text_h + 2 * pad_y

        # Place above the bbox when there's room, otherwise just inside the
        # top edge. Clamp to image bounds so the bar never gets clipped.
        if y1 >= bar_h:
            bar_y = y1 - bar_h
        else:
            bar_y = y1
        bar_x = x1
        if bar_x + bar_w > INPUT_SIZE:
            bar_x = INPUT_SIZE - bar_w
        if bar_x < 0:
            bar_x = 0

        draw.rectangle(
            [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=rgb
        )
        draw.text(
            (bar_x + pad_x, bar_y + pad_y),
            label,
            fill=(255, 255, 255),
            font=_LABEL_FONT,
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def repostprocess(
    raw_tensor: np.ndarray,
    preproc_arr: np.ndarray,
    conf_thresh: float,
    nms_thresh: float,
) -> tuple:
    """Re-run only the postprocessing stage on cached tensors.

    Used by both the full ``run()`` paths (DRY) and the live
    ``/api/repostprocess`` endpoint that fires when the user drags a
    threshold slider. Returns ``(boxes, scores, class_ids, annotated_png,
    postprocess_ms)`` — the post_ms is computed inside the helper so callers
    don't have to time it themselves.
    """
    t0 = time.perf_counter()
    boxes, scores, class_ids = post_process(raw_tensor, conf_thresh, INPUT_SIZE)
    boxes, scores, class_ids = nms(boxes, scores, class_ids, nms_thresh)
    png = _draw_detections(preproc_arr, boxes, scores, class_ids)
    return boxes, scores, class_ids, png, _ms(time.perf_counter() - t0)


def make_ground_truth_png(image_path) -> bytes:
    """Return a 256×256 RGB PNG of the preprocessed input, no bboxes drawn.

    Used by the demo GUI's "ground truth" panel so viewers can see exactly
    what the models see (stretched to 256×256) without any detections on top.
    Implemented as a zero-detection call to ``_draw_detections`` so the
    output pixel format matches the runner panels byte-for-byte.
    """
    arr, _ = _load_and_preprocess(image_path)
    return _draw_detections(arr, [], [], [])


def _make_stub_png(text: str, color=(64, 64, 64)) -> bytes:
    """A grey 256×256 PNG with centred text — used for the HLS stub panel."""
    img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), color=color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # Rough centring — default PIL bitmap font is ~6×11 px per char.
    w = 6 * len(text)
    h = 11
    draw.text(
        ((INPUT_SIZE - w) // 2, (INPUT_SIZE - h) // 2),
        text,
        fill=(220, 220, 220),
        font=font,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _zero_timings() -> dict:
    return {
        "preprocess_ms": None,
        "inference_ms": None,
        "postprocess_ms": None,
        "total_ms": None,
        "cycles": None,
        "cycle_time_ms": None,
        "host_walltime_ms": None,  # HDL only — driver wall-clock incl. AXI preload
        "conf_thresh": None,
        "nms_thresh": None,
    }


# -------------------------------------------------------------------- runners


class TFLiteRunner:
    """Load the TFLite interpreter once; run inference per image.

    Full-int8 models expect uint8 input directly. Float models (e.g.
    ``tinyissimo_ptq_float32.tflite``) expect NHWC float32 in [0, 1]. This
    mirrors the conditional at ``run_inference.py:121-123``.

    Thresholds are passed in per ``run()`` call rather than stored on the
    instance, so the runner is stateless w.r.t. user input — safe to share
    across concurrent FastAPI requests.
    """

    name = "tflite"

    def __init__(self, model_path):
        import tflite_runtime.interpreter as tflite  # noqa: WPS433

        self._interpreter = tflite.Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()

        in_det = self._interpreter.get_input_details()[0]
        out_det = self._interpreter.get_output_details()[0]

        self._in_idx = in_det["index"]
        self._out_idx = out_det["index"]
        self._is_full_int8 = in_det["dtype"] == np.uint8
        self._out_scale, self._out_zp = out_det["quantization"]

    def describe_quant(self) -> dict:
        """Return human-readable quantization strings from the live interpreter.

        Used by app.py to populate the model-info panel without having the
        FastAPI layer poke at private interpreter internals.
        """
        in_det = self._interpreter.get_input_details()[0]
        out_det = self._interpreter.get_output_details()[0]
        in_scale, in_zp = in_det["quantization"]
        out_scale, out_zp = out_det["quantization"]
        return {
            "input_quant": (
                f"{in_det['dtype'].__name__}, "
                f"scale {in_scale:.6g}, zero-point {in_zp}"
            ),
            "output_quant": (
                f"{out_det['dtype'].__name__}, "
                f"scale ≈ {out_scale:.5f}, zero-point {out_zp}"
            ),
        }

    def run(
        self,
        image_path,
        conf_thresh: float = 0.3,
        nms_thresh: float = 0.45,
    ) -> RunnerResult:
        arr, pre_ms = _load_and_preprocess(image_path)

        # --- inference ----------------------------------------------------
        t0 = time.perf_counter()
        if self._is_full_int8:
            input_data = np.expand_dims(arr, axis=0)  # uint8
        else:
            input_data = (
                np.expand_dims(arr, axis=0).astype(np.float32) / 255.0
            )
        self._interpreter.set_tensor(self._in_idx, input_data)
        self._interpreter.invoke()
        raw_output = self._interpreter.get_tensor(self._out_idx)
        infer_ms = _ms(time.perf_counter() - t0)

        # --- post-process -------------------------------------------------
        if self._is_full_int8:
            data = (
                raw_output[0].astype(np.float32) - self._out_zp
            ) * self._out_scale
        else:
            data = raw_output[0].astype(np.float32)

        boxes, scores, class_ids, png, post_ms = repostprocess(
            data, arr, conf_thresh, nms_thresh
        )

        return RunnerResult(
            runner=self.name,
            boxes=list(boxes),
            scores=[float(s) for s in scores],
            class_ids=[int(c) for c in class_ids],
            annotated_png=png,
            timings={
                "preprocess_ms": pre_ms,
                "inference_ms": infer_ms,
                "postprocess_ms": post_ms,
                # End-to-end uses the canonical inference time. For TFLite
                # there is no separable bus cost — wall-clock is compute.
                "total_ms": pre_ms + infer_ms + post_ms,
                "cycles": None,
                "cycle_time_ms": None,
                "host_walltime_ms": None,
                "conf_thresh": conf_thresh,
                "nms_thresh": nms_thresh,
            },
            raw_tensor=data,
            preproc_arr=arr,
        )


class HDLRunner:
    """Load a PYNQ overlay + accelerator driver once; run inference per image.

    Calls the driver's primitives directly (instead of ``driver.run``) so the
    wall-clock timing can be split into ``inference`` and ``postprocess``
    phases. The cycle counter (register 0x00C) is read after ``STATUS.done``
    asserts and converted to ms at the 100 MHz PL clock.

    The "Inference Time" displayed in the GUI is the cycle-derived value
    (``cycles × 10 ns``), not the host wall-clock — the host wall-clock
    includes the AXI-Lite pixel preload (~150 ms for 65 K writes) which is
    a bus protocol cost, not inference cost. The host wall-clock is still
    measured and surfaced in ``timings.host_walltime_ms`` for the notes
    block / model-info caveat.
    """

    name = "hdl"

    def __init__(self, bitstream_path):
        from pynq import Overlay  # noqa: WPS433

        self._overlay = Overlay(str(bitstream_path), ignore_version=True)

        if "tinyissimoyolo_accel_0" not in self._overlay.ip_dict:
            available = ", ".join(sorted(self._overlay.ip_dict.keys()))
            raise RuntimeError(
                "tinyissimoyolo_accel_0 not in overlay.ip_dict; "
                f"available: {available}"
            )

        self._driver = TinyissimoYoloAcceleratorDriver(
            self._overlay.tinyissimoyolo_accel_0
        )

    def run(
        self,
        image_path,
        conf_thresh: float = 0.3,
        nms_thresh: float = 0.45,
    ) -> RunnerResult:
        arr, pre_ms = _load_and_preprocess(image_path)

        # --- inference (includes AXI-Lite pixel preload) ------------------
        t0 = time.perf_counter()
        self._driver.configure(mode=0)
        self._driver.start()
        self._driver.write_pixels(arr)
        if not self._driver.wait_done(timeout_s=2.0):
            raise TimeoutError(
                "HDL accelerator did not assert STATUS.done within 2s"
            )
        cycles = int(self._driver.cycle_count)
        raw_table = self._driver.read_results_raw()
        raw_tensor = self._driver.unpack_detections(raw_table)
        host_walltime_ms = _ms(time.perf_counter() - t0)

        cycle_time_ms = (cycles / PL_CLOCK_HZ) * 1000.0

        # --- post-process via shared helper -------------------------------
        boxes, scores, class_ids, png, post_ms = repostprocess(
            raw_tensor, arr, conf_thresh, nms_thresh
        )

        return RunnerResult(
            runner=self.name,
            boxes=list(boxes),
            scores=[float(s) for s in scores],
            class_ids=[int(c) for c in class_ids],
            annotated_png=png,
            timings={
                "preprocess_ms": pre_ms,
                # "Inference Time" cell in the GUI is cycle-derived, NOT
                # the host wall-clock. This makes the apples-to-apples
                # column comparison fair vs TFLite's invoke() wall-clock.
                "inference_ms": cycle_time_ms,
                "postprocess_ms": post_ms,
                # End-to-end is computed from the canonical inference time
                # (cycle-derived), so the table is internally consistent at
                # every cell. Host wall-clock + AXI overhead live in
                # host_walltime_ms below for the notes block / caveat.
                "total_ms": pre_ms + cycle_time_ms + post_ms,
                "cycles": cycles,
                "cycle_time_ms": cycle_time_ms,
                "host_walltime_ms": host_walltime_ms,
                "conf_thresh": conf_thresh,
                "nms_thresh": nms_thresh,
            },
            raw_tensor=raw_tensor,
            preproc_arr=arr,
        )


class HLSStubRunner:
    """Placeholder — returns a grey panel until a real HLS runner lands.

    Keeps the 4-panel GUI layout populated so the slot is ready when an HLS
    bitstream + runner is integrated. Accepts the same threshold params as
    the other runners (signature-only — they're ignored) so callers don't
    have to special-case it.
    """

    name = "hls"

    def __init__(self):
        self._stub_png = _make_stub_png("HLS not implemented")

    def run(
        self,
        image_path,  # noqa: ARG002
        conf_thresh: float = 0.3,  # noqa: ARG002
        nms_thresh: float = 0.45,  # noqa: ARG002
    ) -> RunnerResult:
        return RunnerResult(
            runner=self.name,
            boxes=[],
            scores=[],
            class_ids=[],
            annotated_png=self._stub_png,
            timings=_zero_timings(),
            raw_tensor=None,
            preproc_arr=None,
        )
