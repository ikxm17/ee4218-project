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


@dataclass
class RunnerResult:
    """Uniform output produced by every runner.

    ``boxes`` / ``scores`` / ``class_ids`` are native Python lists (not numpy)
    because the canonical ``post_process`` returns lists and the FastAPI layer
    will JSON-encode them. ``annotated_png`` is a 256×256 PNG with bboxes
    drawn, suitable for base64-encoding into the response. ``timings`` uses
    ``None`` for metrics that do not apply to a given runner (e.g. TFLite has
    no cycle count).
    """

    runner: str
    boxes: list
    scores: list
    class_ids: list
    annotated_png: bytes
    timings: dict


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
    RGB, so we swap per class.
    """
    img = Image.fromarray(rgb_256, mode="RGB").copy()
    draw = ImageDraw.Draw(img)

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
        # Flat label strip — 12 px tall, fixed width, drawn above bbox when
        # there's room, otherwise just inside the top edge.
        label_y = y1 - 12 if y1 >= 12 else y1
        label_w = max(60, min(INPUT_SIZE - x1, 8 * len(label)))
        draw.rectangle(
            [x1, label_y, x1 + label_w, label_y + 12], fill=rgb
        )
        draw.text((x1 + 2, label_y + 1), label, fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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
    }


# -------------------------------------------------------------------- runners


class TFLiteRunner:
    """Load the TFLite interpreter once; run inference per image.

    Full-int8 models expect uint8 input directly. Float models (e.g.
    ``tinyissimo_ptq_float32.tflite``) expect NHWC float32 in [0, 1]. This
    mirrors the conditional at ``run_inference.py:121-123``.
    """

    name = "tflite"

    def __init__(
        self,
        model_path,
        conf_thresh: float = 0.3,
        nms_thresh: float = 0.45,
    ):
        import tflite_runtime.interpreter as tflite  # noqa: WPS433

        self._interpreter = tflite.Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()

        in_det = self._interpreter.get_input_details()[0]
        out_det = self._interpreter.get_output_details()[0]

        self._in_idx = in_det["index"]
        self._out_idx = out_det["index"]
        self._is_full_int8 = in_det["dtype"] == np.uint8
        self._out_scale, self._out_zp = out_det["quantization"]
        self._conf_thresh = conf_thresh
        self._nms_thresh = nms_thresh

    def run(self, image_path) -> RunnerResult:
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
        t1 = time.perf_counter()
        if self._is_full_int8:
            data = (
                raw_output[0].astype(np.float32) - self._out_zp
            ) * self._out_scale
        else:
            data = raw_output[0].astype(np.float32)

        boxes, scores, class_ids = post_process(
            data, self._conf_thresh, INPUT_SIZE
        )
        boxes, scores, class_ids = nms(
            boxes, scores, class_ids, self._nms_thresh
        )
        png = _draw_detections(arr, boxes, scores, class_ids)
        post_ms = _ms(time.perf_counter() - t1)

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
                "total_ms": pre_ms + infer_ms + post_ms,
                "cycles": None,
                "cycle_time_ms": None,
            },
        )


class HDLRunner:
    """Load a PYNQ overlay + accelerator driver once; run inference per image.

    Calls the driver's primitives directly (instead of ``driver.run``) so the
    wall-clock timing can be split into ``inference`` and ``postprocess``
    phases. The cycle counter (register 0x00C) is read after ``STATUS.done``
    asserts and converted to ms at the 100 MHz PL clock.
    """

    name = "hdl"

    def __init__(
        self,
        bitstream_path,
        conf_thresh: float = 0.3,
        nms_thresh: float = 0.45,
    ):
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
        self._conf_thresh = conf_thresh
        self._nms_thresh = nms_thresh

    def run(self, image_path) -> RunnerResult:
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
        infer_ms = _ms(time.perf_counter() - t0)

        # --- post-process -------------------------------------------------
        t1 = time.perf_counter()
        boxes, scores, class_ids = post_process(
            raw_tensor, self._conf_thresh, INPUT_SIZE
        )
        boxes, scores, class_ids = nms(
            boxes, scores, class_ids, self._nms_thresh
        )
        png = _draw_detections(arr, boxes, scores, class_ids)
        post_ms = _ms(time.perf_counter() - t1)

        cycle_time_ms = (cycles / PL_CLOCK_HZ) * 1000.0

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
                "total_ms": pre_ms + infer_ms + post_ms,
                "cycles": cycles,
                "cycle_time_ms": cycle_time_ms,
            },
        )


class HLSStubRunner:
    """Placeholder — returns a grey panel until a real HLS runner lands.

    Keeps the 4-panel GUI layout populated so the slot is ready when an HLS
    bitstream + runner is integrated.
    """

    name = "hls"

    def __init__(self):
        self._stub_png = _make_stub_png("HLS not implemented")

    def run(self, image_path) -> RunnerResult:  # noqa: ARG002
        return RunnerResult(
            runner=self.name,
            boxes=[],
            scores=[],
            class_ids=[],
            annotated_png=self._stub_png,
            timings=_zero_timings(),
        )
