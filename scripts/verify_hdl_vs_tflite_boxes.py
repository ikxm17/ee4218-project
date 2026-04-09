"""End-to-end detection equivalence gate: HDL goldens vs TFLite final boxes.

This is the load-bearing offline verification for whether the HDL accelerator
and the TFLite reference agree at the level that actually matters for the
deliverable — the final bounding boxes, class predictions, and per-cell
confidence threshold decisions.

Why this script exists
----------------------
`scripts/verify_goldens_vs_tflite.py` diffs the per-layer int8 intermediate
tensors and reports `DIVERGED` by design: the Python golden (and the RTL
it matches) uses a single-step round-half-up requantize
(`(acc*m0 + (1<<(n-1))) >> n`) that does not match TFLite's two-stage
gemmlowp `SRDHM + RDP` pipeline. The two disagree by ≤ 1 LSB at rounding
boundaries, compounded through 17 layers to max|d| ≤ 8 LSB at late layers.
See `hardware/scripts/generate_conv3d_golden.py:224-229` and
`hardware/rtl/conv3d.v:417-420` for the original documentation of this
choice.

That LSB-level divergence is absorbed before it reaches the detector
output, because:

  1. The int8 output is dequantized by ~0.10655 (each LSB = 0.1 in float)
  2. Sigmoid on the 3 class logits saturates away small input perturbations
     except near the 0.5 crossover
  3. DFL softmax over the 64 box logits averages 16 bin-weighted sums,
     damping per-channel noise
  4. The `int()` cast on decoded pixel coordinates floors any sub-pixel
     drift
  5. NMS compares magnitudes at a 0.3 confidence threshold, not exact
     values — a ±0.025 perturbation cannot flip a strong detection

So the right question to ask is not "are the int8 tensors identical?" —
they cannot be — but "do both pipelines detect the same things?". This
script answers that.

Logic
-----
  1. Reconstruct the exact input image from `pixels_layer0.mem`.
  2. Load HDL goldens for layer 13 (cv2 box head, 8×8×64) and layer 16
     (cv3 class head, 8×8×3), stitch into (8, 8, 67) int8.
  3. Dequantize with the driver's hardcoded `OUTPUT_SCALE`/`OUTPUT_ZP`.
  4. Run TFLite on the same image, grab the (1, 8, 8, 67) output tensor,
     dequantize with the same constants for parity.
  5. Run both dequantized tensors through the shared `post_process` + `nms`
     from `software/overlay/drivers/tinyissimoyolo_accelerator.py`.
  6. Compare: detection count, class IDs, confidences, box corner shifts,
     and per-grid-cell confidence-threshold decisions.
  7. Exit 0 iff all of: counts match, classes match, max corner shift ≤
     `--max-corner-px`, and 0/64 cells flip the confidence threshold.

Run via:
    python scripts/verify_hdl_vs_tflite_boxes.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_goldens_vs_tflite import (  # noqa: E402
    LAYER_MAP,
    image_from_pixels_mem,
    load_interpreter,
    unpack_uram,
)
from software.overlay.drivers.tinyissimoyolo_accelerator import (  # noqa: E402
    CLASS_NAMES,
    OUTPUT_SCALE,
    OUTPUT_ZP,
    nms,
    post_process,
)
from software.overlay.tests.checks import load_golden_uram_mem  # noqa: E402


def load_hdl_layer(golden_dir: pathlib.Path, idx: int) -> np.ndarray:
    """Unpack one golden_layer{N}_uram.mem file into its (H, W, C) int8 shape."""
    entry = next(e for e in LAYER_MAP if e[0] == idx)
    _, _, _, words, _, H, W, C = entry
    raw = load_golden_uram_mem(
        str(golden_dir / f"golden_layer{idx}_uram.mem"),
        num_words=words,
    )
    return unpack_uram(raw, H, W, C)


def stitch_hdl_detection(golden_dir: pathlib.Path) -> np.ndarray:
    """Combine HDL layer 13 (cv2 box, 64ch) + layer 16 (cv3 class, 3ch) into (8,8,67)."""
    cv2 = load_hdl_layer(golden_dir, 13)
    cv3 = load_hdl_layer(golden_dir, 16)
    assert cv2.shape == (8, 8, 64), f"unexpected cv2 shape: {cv2.shape}"
    assert cv3.shape == (8, 8, 3), f"unexpected cv3 shape: {cv3.shape}"
    return np.concatenate([cv2, cv3], axis=-1).astype(np.int8)


def get_tflite_detection(model_path: str, pixels_path: str) -> np.ndarray:
    """Run TFLite on the .mem-reconstructed image and return the (8,8,67) int8 output."""
    interp = load_interpreter(model_path)
    image = image_from_pixels_mem(pixels_path)
    in_idx = interp.get_input_details()[0]["index"]
    interp.set_tensor(in_idx, image)
    interp.invoke()
    out_det = interp.get_output_details()[0]
    det = interp.get_tensor(out_det["index"])
    return det[0].astype(np.int8)


def run_postprocess(det_int8: np.ndarray, conf_thresh: float, nms_thresh: float):
    """Dequantize and run the shared postprocess + NMS."""
    dequant = (det_int8.astype(np.float32) - OUTPUT_ZP) * OUTPUT_SCALE
    boxes, scores, cids = post_process(dequant, conf_thresh=conf_thresh)
    boxes, scores, cids = nms(boxes, scores, cids, iou_thresh=nms_thresh)
    return boxes, scores, cids, dequant


def print_detections(label: str, boxes, scores, cids) -> None:
    print(f"\n=== {label} detections ({len(boxes)}) ===")
    if not boxes:
        print("  (none)")
        return
    print(f"  {'cls':<6} {'conf':>6} {'box (xywh)':<30}")
    rows = sorted(zip(cids, scores, boxes), key=lambda r: -r[1])
    for c, s, b in rows:
        print(f"  {CLASS_NAMES.get(c, f'cls{c}'):<6} {s:>6.3f} {str(b):<30}")


def compare_boxes(
    hdl: tuple, tfl: tuple, max_corner_px: int
) -> tuple[bool, list[str]]:
    """Return (passed, reasons) for the detection-list comparison."""
    h_boxes, h_scores, h_cids = hdl
    t_boxes, t_scores, t_cids = tfl
    reasons: list[str] = []

    if len(h_boxes) != len(t_boxes):
        reasons.append(f"detection count differs: HDL={len(h_boxes)} TFL={len(t_boxes)}")
        return False, reasons

    # Match boxes by (class, center) — sort both by score descending, then zip.
    h_rows = sorted(zip(h_cids, h_scores, h_boxes), key=lambda r: -r[1])
    t_rows = sorted(zip(t_cids, t_scores, t_boxes), key=lambda r: -r[1])

    for i, ((hc, hs, hb), (tc, ts, tb)) in enumerate(zip(h_rows, t_rows)):
        if hc != tc:
            reasons.append(
                f"det#{i}: class differs HDL={CLASS_NAMES.get(hc, hc)} "
                f"TFL={CLASS_NAMES.get(tc, tc)}"
            )
        # box format is [x, y, w, h]; diff each component in pixels.
        shifts = [abs(int(a) - int(b)) for a, b in zip(hb, tb)]
        max_shift = max(shifts)
        if max_shift > max_corner_px:
            reasons.append(
                f"det#{i}: corner shift {max_shift}px > {max_corner_px}px "
                f"(HDL={hb} TFL={tb})"
            )

    return (len(reasons) == 0), reasons


def compare_threshold_cells(
    hdl_dequant: np.ndarray, tfl_dequant: np.ndarray, conf_thresh: float
) -> tuple[int, float, float]:
    """Count grid cells where the class-confidence threshold decision flips."""
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    hdl_conf = sigmoid(hdl_dequant[..., 64:]).max(axis=-1)
    tfl_conf = sigmoid(tfl_dequant[..., 64:]).max(axis=-1)
    hdl_pass = hdl_conf > conf_thresh
    tfl_pass = tfl_conf > conf_thresh
    flipped = int((hdl_pass != tfl_pass).sum())
    return flipped, float(np.abs(hdl_conf - tfl_conf).max()), float(np.abs(hdl_conf - tfl_conf).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default="software/models/tflite/tinyissimo_ptq_full_integer_quant.tflite",
        help="Path to the TFLite model",
    )
    parser.add_argument(
        "--pixels-mem",
        default="hardware/testbench/inference_hdl/pixels_layer0.mem",
        help="Path to pixels_layer0.mem (channel-major int8)",
    )
    parser.add_argument(
        "--golden-dir",
        default="hardware/testbench/inference_hdl",
        help="Directory containing golden_layer{N}_uram.mem files",
    )
    parser.add_argument(
        "--conf-thresh", type=float, default=0.3,
        help="Classification confidence threshold (post-sigmoid)",
    )
    parser.add_argument(
        "--nms-thresh", type=float, default=0.45,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--max-corner-px", type=int, default=1,
        help="Max allowed per-corner pixel shift between HDL and TFLite boxes",
    )
    args = parser.parse_args()

    golden_dir = pathlib.Path(args.golden_dir)

    print(f"[boxes] model:      {args.model}")
    print(f"[boxes] pixels:     {args.pixels_mem}")
    print(f"[boxes] golden dir: {golden_dir}")
    print(f"[boxes] conf={args.conf_thresh}  nms={args.nms_thresh}  max_corner={args.max_corner_px}px")

    det_hdl_int8 = stitch_hdl_detection(golden_dir)
    print(
        f"\nHDL golden (stitched): shape={det_hdl_int8.shape} "
        f"cv2 range [{det_hdl_int8[..., :64].min()},{det_hdl_int8[..., :64].max()}] "
        f"cv3 range [{det_hdl_int8[..., 64:].min()},{det_hdl_int8[..., 64:].max()}]"
    )

    det_tfl_int8 = get_tflite_detection(args.model, args.pixels_mem)
    print(
        f"TFLite output:         shape={det_tfl_int8.shape} "
        f"cv2 range [{det_tfl_int8[..., :64].min()},{det_tfl_int8[..., :64].max()}] "
        f"cv3 range [{det_tfl_int8[..., 64:].min()},{det_tfl_int8[..., 64:].max()}]"
    )

    # Informational: raw int8 diff (expected to be bounded, not zero).
    diff = det_hdl_int8.astype(np.int32) - det_tfl_int8.astype(np.int32)
    print(
        f"\nRaw int8 diff: mismatches={int((diff != 0).sum())}/{diff.size}  "
        f"max|d|={int(np.abs(diff).max())}  mean|d|={float(np.abs(diff).mean()):.3f}"
    )
    print(
        f"  cv2 (box, ch 0-63):  mismatches={int((diff[...,:64]!=0).sum())}/"
        f"{diff[...,:64].size}  max|d|={int(np.abs(diff[...,:64]).max())}"
    )
    print(
        f"  cv3 (cls, ch 64-66): mismatches={int((diff[...,64:]!=0).sum())}/"
        f"{diff[...,64:].size}  max|d|={int(np.abs(diff[...,64:]).max())}"
    )

    h_boxes, h_scores, h_cids, h_dequant = run_postprocess(
        det_hdl_int8, args.conf_thresh, args.nms_thresh
    )
    t_boxes, t_scores, t_cids, t_dequant = run_postprocess(
        det_tfl_int8, args.conf_thresh, args.nms_thresh
    )

    print_detections("HDL golden", h_boxes, h_scores, h_cids)
    print_detections("TFLite reference", t_boxes, t_scores, t_cids)

    flipped, max_conf_d, mean_conf_d = compare_threshold_cells(
        h_dequant, t_dequant, args.conf_thresh
    )
    print("\n=== per-cell class-confidence comparison (post-sigmoid max) ===")
    print(f"  max |conf diff|:  {max_conf_d:.6f}")
    print(f"  mean |conf diff|: {mean_conf_d:.6f}")
    print(f"  cells flipping the {args.conf_thresh} threshold: {flipped}/64")

    passed_boxes, reasons = compare_boxes(
        (h_boxes, h_scores, h_cids),
        (t_boxes, t_scores, t_cids),
        args.max_corner_px,
    )
    passed = passed_boxes and (flipped == 0)

    print("\n" + "=" * 60)
    if passed:
        print("PASS: HDL goldens and TFLite produce equivalent detections.")
        print(
            f"      boxes agree; {flipped}/64 threshold flips; "
            f"max corner shift ≤ {args.max_corner_px}px"
        )
        return 0

    print("FAIL: HDL goldens and TFLite differ at the detection level.")
    for r in reasons:
        print(f"  - {r}")
    if flipped > 0:
        print(f"  - {flipped}/64 grid cells flip the {args.conf_thresh} threshold")
    return 1


if __name__ == "__main__":
    sys.exit(main())
