"""HDL accelerator counterpart of run_inference.py.

Loads the TinyissimoYOLO bitstream, runs inference on a single image
through the hardware accelerator, and saves results in the same on-disk
format as run_inference.py so the existing visualisation script (and
the comparison harness) can be reused unchanged.

Usage (on the Kria board):

    sudo XILINX_XRT=/usr /opt/ee4218/ee4218-venv/bin/python3 \\
        software/inference/run_inference_hdl.py \\
        --bitstream hardware/output/tinyissimoyolo.bit \\
        --input-image software/inference/data/input_image.jpg \\
        --result-dir results_hdl/ \\
        --compare-golden \\
            hardware/testbench/inference_hdl/golden_layer16_uram.mem
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

# These imports must work on a dev machine for offboard tests / linting,
# even though pynq itself is only available on the Kria board.
from software.overlay.drivers.tinyissimoyolo_accelerator import (
    OUTPUT_SCALE,
    OUTPUT_ZP,
    TinyissimoYoloAcceleratorDriver,
)
from software.overlay.tests.checks import load_golden_uram_mem


def load_image(path: str) -> np.ndarray:
    """Identical preprocessing to run_inference.py:119 so the HDL path
    consumes the exact same uint8 tensor that the TFLite baseline does."""
    img = Image.open(path).convert("RGB").resize((256, 256))
    return np.array(img, dtype=np.uint8)


def save_results(result_dir: str, result: dict) -> None:
    """Write the on-disk layout matching run_inference.py:142-156.

    Expects `result` to contain `boxes`, `scores`, `class_ids`,
    `cycle_count`, and `raw_tensor` (the (8, 8, 67) float32 tensor
    from `TinyissimoYoloAcceleratorDriver.run()`).
    """
    os.makedirs(result_dir, exist_ok=True)

    np.save(os.path.join(result_dir, "boxes.npy"),
            np.array(result["boxes"]))
    np.save(os.path.join(result_dir, "scores.npy"),
            np.array(result["scores"]))
    np.save(os.path.join(result_dir, "class_ids.npy"),
            np.array(result["class_ids"]))
    # HDL run() applies NMS in-place, so kept indices are 0..N-1.
    np.save(os.path.join(result_dir, "indices.npy"),
            np.arange(len(result["boxes"])))

    # Reshape to TFLite output shape so compare_hdl_vs_tflite.py can
    # diff both raw tensors without knowing which side produced which.
    np.save(os.path.join(result_dir, "raw_out_0.npy"),
            result["raw_tensor"].reshape(1, 8, 8, 67))

    meta = {
        "source":       "hdl",
        "is_full_int8": True,
        "scales":       [OUTPUT_SCALE],
        "zps":          [OUTPUT_ZP],
        "cycle_count":  int(result["cycle_count"]),
    }
    with open(os.path.join(result_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def compare_against_golden(raw_table: np.ndarray, golden_path: str) -> bool:
    """Compare the cv3 region of an already-captured result table against
    `golden_layer16_uram.mem` byte-for-byte.

    Takes the raw table directly (not a driver) so the caller can reuse
    the table from `driver.run()`'s result dict and avoid a duplicate
    MMIO readout.

    Returns True on PASS. Prints a per-position diff summary on FAIL.
    """
    cv3_actual = raw_table[256:320, :3]    # 3 valid channels per cell
    cv3_golden = load_golden_uram_mem(golden_path, num_words=64)[:, :3]

    if np.array_equal(cv3_actual, cv3_golden):
        print(f"[golden] PASS: 64 cv3 words match {golden_path}")
        return True

    diff_mask = cv3_actual != cv3_golden
    num_bad = int(diff_mask.sum())
    print(f"[golden] FAIL: {num_bad} byte mismatches vs {golden_path}")
    bad_positions = np.argwhere(diff_mask)[:8]
    for pos in bad_positions:
        word, lane = int(pos[0]), int(pos[1])
        a = int(cv3_actual[word, lane])
        g = int(cv3_golden[word, lane])
        print(f"  word={word:3d} lane={lane}  actual={a:+4d}  golden={g:+4d}")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bitstream", type=str, required=True,
        help="Path to the .bit file containing the accelerator")
    parser.add_argument(
        "--input-image", type=str,
        default="software/inference/data/input_image.jpg",
        help="Path to the test image (default: canonical input_image.jpg)")
    parser.add_argument(
        "--result-dir", type=str, default="results_hdl",
        help="Directory to save .npy outputs and meta.json")
    parser.add_argument(
        "--compare-golden", type=str, default=None,
        help="Path to golden_layer16_uram.mem for bit-exact comparison")
    parser.add_argument(
        "--conf-thresh", type=float, default=0.3,
        help="Confidence threshold for detection (default 0.3)")
    parser.add_argument(
        "--nms-thresh", type=float, default=0.45,
        help="IoU threshold for NMS (default 0.45)")
    args = parser.parse_args()

    # Deferred PYNQ import so the module can be imported on a dev machine.
    try:
        from pynq import Overlay
    except ImportError as exc:
        print(f"FATAL: pynq is not installed in this environment ({exc})",
              file=sys.stderr)
        sys.exit(2)

    print(f"[overlay] loading {args.bitstream}")
    overlay = Overlay(args.bitstream, ignore_version=True)

    if "tinyissimoyolo_accel_0" not in overlay.ip_dict:
        print("FATAL: tinyissimoyolo_accel_0 not in overlay.ip_dict; "
              "available: " + ", ".join(sorted(overlay.ip_dict.keys())),
              file=sys.stderr)
        sys.exit(3)

    driver = TinyissimoYoloAcceleratorDriver(overlay.tinyissimoyolo_accel_0)

    print(f"[image] loading {args.input_image}")
    image = load_image(args.input_image)

    print("[run] driving accelerator")
    result = driver.run(image,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh)

    print(f"[run] cycle_count={result['cycle_count']}, "
          f"detections={len(result['boxes'])}")

    save_results(args.result_dir, result)
    print(f"[run] results saved to {args.result_dir}")

    pass_status = True
    if args.compare_golden:
        pass_status = compare_against_golden(
            result["raw_table"], args.compare_golden)

    if not pass_status:
        sys.exit(1)


if __name__ == "__main__":
    main()
