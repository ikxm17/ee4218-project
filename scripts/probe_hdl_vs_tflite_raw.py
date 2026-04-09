#!/usr/bin/env python3
"""Compare raw int8 output tensors from the HDL accelerator and TFLite.

The dequantization scales for layers 13 and 16 are essentially identical
between ``tinyissimoyolo_accelerator.py`` (hardcoded ``OUTPUT_SCALE=0.10655``)
and the tflite model's output_details (``0.1065545753``) — a 0.005 percent
difference. That means the 4-vs-1 detection divergence observed on the
bowl-of-oranges demo image cannot be explained by dequant math alone. The
divergence has to be in the raw int8 tensors themselves.

This script runs one image through both paths, reads the raw int8 output
tensor from each *before* dequantization, reshapes them into the same
(8, 8, 67) layout, and prints per-element difference statistics. This
definitively answers whether the HDL silicon reproduces the tflite runtime's
int8 outputs exactly or whether the two diverge.

Run on the Kria board, under sudo (PYNQ needs /dev/mem)::

    echo asdfzxcv | sudo -S XILINX_XRT=/usr \\
        /opt/ee4218/ee4218-venv/bin/python3 scripts/probe_hdl_vs_tflite_raw.py \\
        --bitstream hardware/output/preserved/playground_FIXED_5e86ce6c/playground_FIXED_5e86ce6c.xsa \\
        --model-path software/models/tflite/tinyissimo_ptq_full_integer_quant.tflite \\
        --image software/models/demo_images/000000050896.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Make the ``software`` package importable when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from software.overlay.drivers.tinyissimoyolo_accelerator import (  # noqa: E402
    OUTPUT_SCALE,
    OUTPUT_ZP,
    TinyissimoYoloAcceleratorDriver,
)


def unpack_raw_int8(raw_table: np.ndarray) -> np.ndarray:
    """Replicate ``TinyissimoYoloAcceleratorDriver.unpack_detections`` but
    WITHOUT the dequantization step, so we can diff raw int8 vs the tflite
    interpreter's raw int8 output.

    Args:
        raw_table: (320, 16) int8 from ``read_results_raw()``.

    Returns:
        (8, 8, 67) int8 — cv2 bbox logits (64 channels) + cv3 class logits
        (3 channels), matching the layout of tflite output tensor index 0.
    """
    cv2 = raw_table[:256].reshape(4, 8, 8, 16).transpose(1, 2, 0, 3).reshape(8, 8, 64)
    cv3 = raw_table[256:320].reshape(8, 8, 16)[:, :, :3]
    return np.concatenate([cv2, cv3], axis=-1).astype(np.int8)


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((256, 256))
    return np.array(img, dtype=np.uint8)


def run_hdl(bitstream_path: Path, image: np.ndarray) -> np.ndarray:
    from pynq import Overlay  # noqa: WPS433

    overlay = Overlay(str(bitstream_path), ignore_version=True)
    driver = TinyissimoYoloAcceleratorDriver(overlay.tinyissimoyolo_accel_0)

    driver.configure(mode=0)
    driver.start()
    driver.write_pixels(image)
    if not driver.wait_done(timeout_s=2.0):
        raise TimeoutError("HDL accelerator did not finish within 2 s")

    raw_table = driver.read_results_raw()
    return unpack_raw_int8(raw_table)


def run_tflite(model_path: Path, image: np.ndarray) -> tuple[np.ndarray, float, int]:
    import tflite_runtime.interpreter as tflite  # noqa: WPS433

    interp = tflite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()

    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    nhwc = np.expand_dims(image, axis=0)
    if in_det["dtype"] == np.uint8:
        interp.set_tensor(in_det["index"], nhwc)
    else:
        interp.set_tensor(
            in_det["index"], nhwc.astype(np.float32) / 255.0,
        )
    interp.invoke()

    raw_int8 = interp.get_tensor(out_det["index"]).astype(np.int8)[0]
    scale, zp = out_det["quantization"]
    return raw_int8, float(scale), int(zp)


def report_diff(hdl: np.ndarray, tfl: np.ndarray) -> None:
    assert hdl.shape == tfl.shape, f"shape mismatch: {hdl.shape} vs {tfl.shape}"
    hdl_i = hdl.astype(np.int32)
    tfl_i = tfl.astype(np.int32)
    diff = hdl_i - tfl_i

    exact = np.array_equal(hdl, tfl)
    n_total = hdl.size
    n_diff = int(np.count_nonzero(diff))
    max_abs = int(np.abs(diff).max())
    mean_abs = float(np.abs(diff).mean())

    print()
    print("=" * 60)
    print("Raw int8 tensor diff — HDL vs TFLite")
    print("=" * 60)
    print(f"  shape:              {hdl.shape}")
    print(f"  elements:           {n_total}")
    print(f"  exactly equal:      {exact}")
    print(f"  positions differ:   {n_diff}  ({100 * n_diff / n_total:.2f}%)")
    print(f"  max |diff|:         {max_abs}")
    print(f"  mean |diff|:        {mean_abs:.4f}")

    if exact:
        return

    print()
    print("  diff histogram (HDL - TFLite):")
    bins = np.bincount(np.clip(diff.flatten() + 128, 0, 255), minlength=256)
    for v, count in enumerate(bins):
        if count > 0:
            print(f"    delta = {v - 128:+4d}  count = {count}")

    # Break down cv2 (first 64 channels) vs cv3 (last 3 channels)
    print()
    print("  per-channel-band breakdown:")
    cv2_diff = diff[..., :64]
    cv3_diff = diff[..., 64:]
    print(
        f"    cv2 (bbox DFL, 64 ch): "
        f"n_diff={int(np.count_nonzero(cv2_diff))}/{cv2_diff.size}  "
        f"max|={int(np.abs(cv2_diff).max())}  "
        f"mean|={float(np.abs(cv2_diff).mean()):.4f}"
    )
    print(
        f"    cv3 (class, 3 ch):    "
        f"n_diff={int(np.count_nonzero(cv3_diff))}/{cv3_diff.size}  "
        f"max|={int(np.abs(cv3_diff).max())}  "
        f"mean|={float(np.abs(cv3_diff).mean()):.4f}"
    )

    # Per-grid-cell breakdown for the 3 class channels — these are what
    # the 0.3 confidence threshold acts on, so any diff here directly
    # changes which cells produce detections.
    print()
    print("  class logits (cv3) — HDL vs TFLite at each of 64 grid cells:")
    print(f"    {'(y,x)':<8} {'cls':<4} {'HDL':>5} {'TFL':>5} {'diff':>5}")
    for y in range(8):
        for x in range(8):
            for c in range(3):
                h = int(hdl[y, x, 64 + c])
                t = int(tfl[y, x, 64 + c])
                if h != t:
                    cname = ["chair", "bowl", "cup"][c]
                    print(
                        f"    ({y},{x})   {cname:<5} {h:>5d} {t:>5d} {h - t:>+5d}"
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bitstream", type=Path, required=True,
        help="Path to the .xsa (or .bit) with the HDL accelerator",
    )
    parser.add_argument(
        "--model-path", type=Path, required=True,
        help="Path to the TFLite model",
    )
    parser.add_argument(
        "--image", type=Path, required=True,
        help="Image file to run through both paths",
    )
    args = parser.parse_args()

    print(f"[probe] image:     {args.image}")
    print(f"[probe] model:     {args.model_path}")
    print(f"[probe] bitstream: {args.bitstream}")
    image = load_image(args.image)

    print("[probe] running TFLite...")
    tfl_raw, tfl_scale, tfl_zp = run_tflite(args.model_path, image)

    print("[probe] running HDL...")
    hdl_raw = run_hdl(args.bitstream, image)

    print()
    print("Dequantization params used downstream of these raw tensors:")
    print(
        f"  HDL hardcoded:   scale = {OUTPUT_SCALE:.10f}  zp = {OUTPUT_ZP}"
    )
    print(f"  TFLite runtime:  scale = {tfl_scale:.10f}  zp = {tfl_zp}")
    print(
        f"  scale rel diff:  {100 * (tfl_scale - OUTPUT_SCALE) / OUTPUT_SCALE:+.4f}%"
    )

    report_diff(hdl_raw, tfl_raw)


if __name__ == "__main__":
    main()
