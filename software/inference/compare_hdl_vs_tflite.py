"""Compare HDL accelerator output against TFLite baseline.

Reads two result directories produced by `run_inference.py` (TFLite) and
`run_inference_hdl.py` (HDL accelerator) and prints a comparison table:

  - Raw tensor delta after dequantising both sides into float32
  - Per-detection geometric match (IoU > threshold) and class agreement

Usage:

    python software/inference/compare_hdl_vs_tflite.py \\
        --tflite-dir results_tflite/ \\
        --hdl-dir results_hdl/

Both directories must contain `raw_out_0.npy`, `boxes.npy`, `scores.npy`,
`class_ids.npy`, and `meta.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# compute_iou lives in run_inference.py — single source of truth for the
# [x, y, w, h] IoU formula across CPU and HDL paths.
from software.inference.run_inference import compute_iou


def load_result_dir(path: str) -> dict:
    """Load a result directory written by run_inference.py or
    run_inference_hdl.py. Returns a dict with raw, boxes, scores,
    class_ids, and meta."""
    return {
        "raw":       np.load(os.path.join(path, "raw_out_0.npy")),
        "boxes":     np.load(os.path.join(path, "boxes.npy")),
        "scores":    np.load(os.path.join(path, "scores.npy")),
        "class_ids": np.load(os.path.join(path, "class_ids.npy")),
        "meta":      json.loads(open(os.path.join(path, "meta.json")).read()),
    }


def dequantise_tflite_raw(raw: np.ndarray, scale: float, zp: int) -> np.ndarray:
    """Apply (raw - zp) * scale, matching run_inference.py:70.

    Works for both uint8 and int8 quantised tensors — the int promotion
    to int64 happens implicitly via numpy's dtype rules.
    """
    return (raw.astype(np.int64) - zp).astype(np.float32) * scale


def compare_raw_tensors(a: np.ndarray, b: np.ndarray) -> dict:
    """Element-wise stats over two same-shape float tensors."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {
        "max_abs_diff":  float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "shape":         a.shape,
    }


def compare_detections(
    *,
    boxes_a: np.ndarray,
    classes_a: np.ndarray,
    boxes_b: np.ndarray,
    classes_b: np.ndarray,
    iou_thresh: float = 0.5,
) -> dict:
    """Greedy IoU-based matching between two detection lists.

    For each box in `a`, find its best-IoU partner in `b`. If the IoU
    exceeds `iou_thresh`, mark them matched and record whether the
    classes also agree. Each `b` box can be claimed at most once.

    Returns a dict with `matched`, `unmatched_a`, `unmatched_b`, and
    a list of per-match `{iou, class_match}` records.
    """
    matched = 0
    used_b: set[int] = set()
    matches: list[dict] = []

    for i, box_a in enumerate(boxes_a):
        best_iou = 0.0
        best_j = -1
        for j, box_b in enumerate(boxes_b):
            if j in used_b:
                continue
            iou = compute_iou(box_a, box_b)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thresh:
            matched += 1
            used_b.add(best_j)
            matches.append({
                "iou":         best_iou,
                "class_match": bool(classes_a[i] == classes_b[best_j]),
            })

    return {
        "matched":     matched,
        "unmatched_a": int(len(boxes_a) - matched),
        "unmatched_b": int(len(boxes_b) - len(used_b)),
        "matches":     matches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite-dir", required=True,
                        help="Directory of TFLite results")
    parser.add_argument("--hdl-dir", required=True,
                        help="Directory of HDL accelerator results")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU threshold for matching boxes")
    args = parser.parse_args()

    tflite = load_result_dir(args.tflite_dir)
    hdl = load_result_dir(args.hdl_dir)

    print(f"=== Raw tensor comparison ===")
    tflite_scale = tflite["meta"]["scales"][0]
    tflite_zp = tflite["meta"]["zps"][0]
    tflite_dq = dequantise_tflite_raw(tflite["raw"], tflite_scale, tflite_zp)
    # HDL raw is already dequantised (run_inference_hdl.py saves the
    # post-unpack_detections float tensor).
    hdl_dq = hdl["raw"].astype(np.float32)
    if tflite_dq.shape != hdl_dq.shape:
        print(f"  shape mismatch: tflite={tflite_dq.shape}, hdl={hdl_dq.shape}")
        sys.exit(1)
    stats = compare_raw_tensors(tflite_dq, hdl_dq)
    print(f"  shape:         {stats['shape']}")
    print(f"  max |diff|:    {stats['max_abs_diff']:.6f}")
    print(f"  mean |diff|:   {stats['mean_abs_diff']:.6f}")

    print(f"\n=== Detection comparison (IoU thresh = {args.iou_thresh}) ===")
    print(f"  TFLite detections: {len(tflite['boxes'])}")
    print(f"  HDL    detections: {len(hdl['boxes'])}")
    report = compare_detections(
        boxes_a=hdl["boxes"],
        classes_a=hdl["class_ids"],
        boxes_b=tflite["boxes"],
        classes_b=tflite["class_ids"],
        iou_thresh=args.iou_thresh,
    )
    print(f"  matched:           {report['matched']}")
    print(f"  unmatched HDL:     {report['unmatched_a']}")
    print(f"  unmatched TFLite:  {report['unmatched_b']}")
    for i, m in enumerate(report["matches"]):
        cls = "OK" if m["class_match"] else "WRONG-CLASS"
        print(f"    match {i}: iou={m['iou']:.3f}  class={cls}")

    all_class_match = all(m["class_match"] for m in report["matches"])
    full_match = (
        report["unmatched_a"] == 0
        and report["unmatched_b"] == 0
        and all_class_match
    )
    if full_match:
        print("\nPASS: HDL detections match TFLite within tolerance")
        sys.exit(0)
    else:
        print("\nFAIL: detection sets diverge")
        sys.exit(1)


if __name__ == "__main__":
    main()
