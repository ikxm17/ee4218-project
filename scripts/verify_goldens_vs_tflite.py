"""Compare the 17 per-layer golden .mem files against TFLite intermediate tensors.

This is the offline verification gate that answers "do the golden files on disk
agree with the TFLite reference for every HDL layer, bit-exact?". It runs in a
few seconds on the dev host, no bitstream rebuild, no board access — it only
needs the TFLite model, the `pixels_layer0.mem` input, and the
`golden_layer*_uram.mem` files the HDL testbench consumes.

Logic:
  1. Reconstruct the INT8 input image from `pixels_layer0.mem` (the exact bytes
     the HDL sees) and feed it to TFLite.
  2. Walk the TFLite op graph, find the 17 CONV_2D ops in topological order,
     classify each one's tail chain (Logistic→Mul→?MaxPool2D or bare conv) and
     validate it against the HDL `LAYER_CFG` shape / has_silu / has_pool.
  3. Load each `golden_layer{N}_uram.mem`, unpack the URAM layout into
     [H, W, C] int8, and diff it against the TFLite tensor.
  4. Print a per-layer mismatch table and return a non-zero exit code if any
     layer is not bit-exact.

Pair with `scripts/diag_accel_silicon_vs_tflite.py` which runs the same
comparison on the board against live silicon URAM.

Run via:
    python scripts/verify_goldens_vs_tflite.py
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from software.overlay.tests.checks import load_golden_uram_mem  # noqa: E402


# Authoritative per-layer URAM map (mirrors diag_accel_per_layer_sweep.LAYER_MAP
# plus the spatial/channel dims needed to unpack into [H, W, C]).
LAYER_MAP = [
    # (idx, fmap_buf, base, words, valid_lanes, H, W, C)
    ( 0, 0,   0, 16384, 16, 128, 128,  16),
    ( 1, 1,   0, 16384, 16, 128, 128,  16),
    ( 2, 0,   0,  4096, 16,  64,  64,  16),
    ( 3, 1,   0,  8192, 16,  64,  64,  32),
    ( 4, 0,   0,  2048, 16,  32,  32,  32),
    ( 5, 1,   0,  4096, 16,  32,  32,  64),
    ( 6, 0,   0,  1024, 16,  16,  16,  64),
    ( 7, 1,   0,  1024, 16,  16,  16,  64),
    ( 8, 0,   0,   512, 16,   8,   8, 128),
    ( 9, 1,   0,   512, 16,   8,   8, 128),
    (10, 0,   0,   128, 16,   8,   8,  24),
    (11, 1, 256,   256, 16,   8,   8,  64),
    (12, 0, 256,   256, 16,   8,   8,  64),
    (13, 1, 256,   256, 16,   8,   8,  64),
    (14, 1, 512,   128, 16,   8,   8,  24),
    (15, 0, 512,   128, 16,   8,   8,  24),
    (16, 1, 512,    64,  3,   8,   8,   3),
]

# (h_in, w_in, cin, cout, has_pool, has_silu) — subset of hardware/scripts/
# generate_conv3d_golden.py LAYER_CFG used to validate TFLite op mapping.
LAYER_CFG = [
    (256, 256,   3,  16, True,  True),   #  0
    (128, 128,  16,  16, False, True),   #  1
    (128, 128,  16,  16, True,  True),   #  2
    ( 64,  64,  16,  32, False, True),   #  3
    ( 64,  64,  32,  32, True,  True),   #  4
    ( 32,  32,  32,  64, False, True),   #  5
    ( 32,  32,  64,  64, True,  True),   #  6
    ( 16,  16,  64,  64, False, True),   #  7
    ( 16,  16,  64, 128, True,  True),   #  8
    (  8,   8, 128, 128, False, True),   #  9
    (  8,   8, 128,  24, False, True),   # 10
    (  8,   8,  24,  64, False, True),   # 11
    (  8,   8,  64,  64, False, True),   # 12
    (  8,   8,  64,  64, False, False),  # 13  CONV1_LIN
    (  8,   8,  24,  24, False, True),   # 14
    (  8,   8,  24,  24, False, True),   # 15
    (  8,   8,  24,   3, False, False),  # 16  CONV1_LIN
]


def load_interpreter(model_path: str):
    """Return a TFLite Interpreter with `experimental_preserve_all_tensors=True`.

    Prefers `tflite_runtime` (matches the on-board environment in
    `software/inference/inspect_intermediates.py`); falls back to
    `tensorflow.lite` on the dev host if `tflite_runtime` isn't installed.
    Prints which backend was used.
    """
    try:
        import tflite_runtime.interpreter as tflite
        print(f"[tflite] using tflite_runtime: {model_path}")
    except ImportError:
        import tensorflow as tf
        tflite = tf.lite
        print(f"[tflite] using tensorflow.lite: {model_path}")
    interp = tflite.Interpreter(
        model_path=model_path,
        experimental_preserve_all_tensors=True,
    )
    interp.allocate_tensors()
    return interp


def image_from_pixels_mem(mem_path: str) -> np.ndarray:
    """Reconstruct the uint8 image TFLite should see from pixels_layer0.mem.

    The .mem file is channel-major int8, 3×256×256 = 196608 hex lines, one
    2-hex-char two's-complement byte per line. This returns a (1, 256, 256, 3)
    uint8 tensor in NHWC — exactly what the TFLite interpreter expects for
    `set_tensor(input_idx, ...)`.

    The +128 step converts int8 back to uint8 (model input zero-point = -128).
    The transpose turns CHW into HWC. This is the SAME image the silicon sees
    via `TinyissimoYoloAcceleratorDriver.write_pixels` — preserving that
    equivalence is the whole point of reading from the .mem file instead of a
    JPEG.
    """
    with open(mem_path) as f:
        mem_bytes = np.array(
            [int(line.strip(), 16) for line in f if line.strip()],
            dtype=np.uint8,
        )
    assert mem_bytes.size == 3 * 256 * 256, (
        f"{mem_path}: expected {3 * 256 * 256} lines, got {mem_bytes.size}"
    )
    mem_int8 = mem_bytes.view(np.int8).reshape(3, 256, 256)
    image_hwc = (mem_int8.astype(np.int16) + 128).astype(np.uint8).transpose(1, 2, 0)
    tensor = np.expand_dims(image_hwc, 0)
    assert tensor.dtype == np.uint8 and tensor.shape == (1, 256, 256, 3), (
        f"reconstructed image has shape={tensor.shape} dtype={tensor.dtype}"
    )
    return tensor


def build_hdl_to_tflite_map(interp) -> list[dict]:
    """Walk the TFLite op graph and map HDL layers 0..16 → observable tensors.

    Strategy:
      - Pull `_get_ops_details()` and walk in graph order.
      - Index CONV_2D ops. The first 17 match HDL layers 0..16 one-to-one,
        because the TFLite graph's conv sequence mirrors the TinyissimoYOLO
        backbone (layers 0..10), then the cv2 head (11, 12, 13) as a linear
        chain off layer 10's output, then the cv3 head (14, 15, 16) branching
        off the same layer-10 tensor. Any CONV_2D beyond the first 17 (e.g.
        the DFL 1x1 conv) is post-processing and ignored.
      - For each matched CONV_2D op, look for a tail chain:
          Logistic(conv) → Mul(conv, logistic) → (MaxPool2D?)
        If the chain is present and HDL says has_silu=True, the observable
        tensor is the Mul output (or the MaxPool output if has_pool=True).
        If HDL says has_silu=False (CONV1_LIN, layers 13 and 16), the
        observable tensor is the conv output directly.
      - Validate the conv's filter shape ([cout, k, k, cin]) and output
        spatial shape against LAYER_CFG. Raise if they disagree — this is a
        loud signal that the model file doesn't match the HDL topology.

    Returns:
        List of dicts (one per HDL layer, index 0..16) with keys:
            hdl_idx, conv_op_idx, conv_out_tensor, tail_kind, observable_tensor
    """
    ops = interp._get_ops_details()
    tdet = {t["index"]: t for t in interp.get_tensor_details()}

    conv_ops = [(i, op) for i, op in enumerate(ops) if op["op_name"] == "CONV_2D"]
    if len(conv_ops) < 17:
        raise RuntimeError(
            f"expected >=17 CONV_2D ops in TFLite graph, found {len(conv_ops)}"
        )

    def find_next(op_name: str, consumer_of: int, start: int) -> int | None:
        """First op at index >=start with op_name that reads consumer_of."""
        for j in range(start, len(ops)):
            op = ops[j]
            if op["op_name"] == op_name and consumer_of in list(op["inputs"]):
                return j
        return None

    mapping: list[dict] = []
    for hdl_idx in range(17):
        cfg = LAYER_CFG[hdl_idx]
        h_in, w_in, cin, cout, has_pool, has_silu = cfg
        conv_op_idx, conv_op = conv_ops[hdl_idx]
        conv_out = int(conv_op["outputs"][0])

        # Shape validation against LAYER_CFG.
        filt_idx = int(conv_op["inputs"][1])
        filt_shape = tdet[filt_idx]["shape"].tolist()   # [cout, kH, kW, cin]
        in_idx = int(conv_op["inputs"][0])
        in_shape = tdet[in_idx]["shape"].tolist()       # [1, H, W, C]
        if (in_shape[1], in_shape[2], in_shape[3]) != (h_in, w_in, cin):
            raise RuntimeError(
                f"HDL layer {hdl_idx}: LAYER_CFG expects in=({h_in},{w_in},{cin}) "
                f"but TFLite conv#{hdl_idx} has in={in_shape[1:]}"
            )
        if (filt_shape[0], filt_shape[3]) != (cout, cin):
            raise RuntimeError(
                f"HDL layer {hdl_idx}: LAYER_CFG expects cout={cout} cin={cin} "
                f"but TFLite filter is {filt_shape}"
            )

        # Tail chain walk.
        if has_silu:
            log_j = find_next("LOGISTIC", conv_out, conv_op_idx + 1)
            if log_j is None:
                raise RuntimeError(
                    f"HDL layer {hdl_idx}: has_silu=True but no LOGISTIC found"
                )
            log_out = int(ops[log_j]["outputs"][0])
            mul_j = find_next("MUL", conv_out, log_j + 1)
            if mul_j is None or log_out not in list(ops[mul_j]["inputs"]):
                raise RuntimeError(
                    f"HDL layer {hdl_idx}: SiLU MUL(conv, logistic) not found"
                )
            mul_out = int(ops[mul_j]["outputs"][0])

            if has_pool:
                pool_j = find_next("MAX_POOL_2D", mul_out, mul_j + 1)
                if pool_j is None:
                    raise RuntimeError(
                        f"HDL layer {hdl_idx}: has_pool=True but no MAX_POOL_2D"
                    )
                observable = int(ops[pool_j]["outputs"][0])
                tail_kind = "conv+silu+pool"
            else:
                observable = mul_out
                tail_kind = "conv+silu"
        else:
            if has_pool:
                raise RuntimeError(
                    f"HDL layer {hdl_idx}: has_pool=True on CONV1_LIN is impossible"
                )
            observable = conv_out
            tail_kind = "conv (linear)"

        mapping.append({
            "hdl_idx":          hdl_idx,
            "conv_op_idx":      conv_op_idx,
            "conv_out_tensor":  conv_out,
            "tail_kind":        tail_kind,
            "observable_tensor": observable,
        })

    return mapping


def unpack_uram(raw: np.ndarray, H: int, W: int, C: int) -> np.ndarray:
    """Convert (num_words, 16) int8 URAM layout to (H, W, C) int8.

    The URAM stores activations channel-group-major: outer loop iterates
    channel groups of 16, then row, then column. Each word's 16 lanes are
    the 16 channels of a single spatial position in that group. For C not a
    multiple of 16, the final group is padded — we slice to C at the end to
    drop the padding lanes (e.g. layer 16 has C=3, only lanes 0..2 valid).
    """
    c_groups = math.ceil(C / 16)
    expected = c_groups * H * W
    if raw.shape != (expected, 16):
        raise ValueError(
            f"URAM shape mismatch: got {raw.shape}, expected ({expected}, 16) "
            f"for H={H} W={W} C={C} (c_groups={c_groups})"
        )
    grouped = raw.reshape(c_groups, H, W, 16)
    hwc = grouped.transpose(1, 2, 0, 3).reshape(H, W, c_groups * 16)
    return hwc[:, :, :C]


def compare_layer(
    golden_hwc: np.ndarray, tflite_tensor: np.ndarray
) -> dict:
    """Diff unpacked golden [H, W, C] against TFLite NHWC tensor (batch stripped).

    Both sides must be int8. Returns a dict of statistics used to build the
    per-layer table row.
    """
    if tflite_tensor.ndim == 4 and tflite_tensor.shape[0] == 1:
        tflite_hwc = tflite_tensor[0]
    else:
        tflite_hwc = tflite_tensor
    if golden_hwc.shape != tflite_hwc.shape:
        raise ValueError(
            f"shape mismatch: golden={golden_hwc.shape} tflite={tflite_hwc.shape}"
        )
    if golden_hwc.dtype != np.int8 or tflite_hwc.dtype != np.int8:
        raise ValueError(
            f"dtype mismatch: golden={golden_hwc.dtype} tflite={tflite_hwc.dtype}"
        )
    diff = golden_hwc.astype(np.int32) - tflite_hwc.astype(np.int32)
    abs_diff = np.abs(diff)
    mismatches = np.argwhere(diff != 0)
    first_positions = [tuple(int(x) for x in row) for row in mismatches[:4]]
    return {
        "shape":         golden_hwc.shape,
        "num_mismatch":  int((diff != 0).sum()),
        "total":         int(diff.size),
        "max_abs":       int(abs_diff.max()) if abs_diff.size else 0,
        "mean_abs":      float(abs_diff.mean()) if abs_diff.size else 0.0,
        "first_pos":     first_positions,
    }


def verdict_for(stats: dict, max_lsb: int = 0) -> str:
    """Classify per-layer stats into BIT-EXACT / CLOSE / WITHIN-TOL / DIVERGED.

    Args:
        stats: output of :func:`compare_layer`.
        max_lsb: non-strict tolerance. If > 0, any layer with
            ``stats["max_abs"] <= max_lsb`` is labelled ``WITHIN-TOL`` rather
            than ``DIVERGED`` and counted as a pass in the summary. This
            exists because the Python golden uses a single-step round-half-up
            requantize that diverges from TFLite's gemmlowp pipeline by a
            bounded LSB amount — see ``hardware/scripts/README.md`` Step 3a
            for the full explanation.
    """
    if stats["num_mismatch"] == 0:
        return "BIT-EXACT"
    if stats["max_abs"] <= 1:
        return "CLOSE (max<=1)"
    if max_lsb > 0 and stats["max_abs"] <= max_lsb:
        return f"WITHIN-TOL (max<={max_lsb})"
    return "DIVERGED"


def main():
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
        "--max-lsb",
        type=int,
        default=0,
        help=(
            "Bounded LSB tolerance. If > 0, per-layer diffs with max|d| <= N "
            "are labelled WITHIN-TOL instead of DIVERGED and counted as a "
            "pass. Use --max-lsb 8 to treat the known round-half-up vs "
            "gemmlowp SRDHM+RDP drift as acceptable. See "
            "hardware/scripts/README.md Step 3a."
        ),
    )
    args = parser.parse_args()

    interp = load_interpreter(args.model)

    image = image_from_pixels_mem(args.pixels_mem)
    in_idx = interp.get_input_details()[0]["index"]
    interp.set_tensor(in_idx, image)
    interp.invoke()

    mapping = build_hdl_to_tflite_map(interp)
    print("\n=== HDL layer -> TFLite tensor mapping ===")
    print(f"{'Layer':>5} | {'op_idx':>6} | {'conv_out':>8} | {'observable':>10} | tail")
    print("-" * 60)
    for m in mapping:
        print(
            f"{m['hdl_idx']:>5} | {m['conv_op_idx']:>6} | "
            f"{m['conv_out_tensor']:>8} | {m['observable_tensor']:>10} | {m['tail_kind']}"
        )

    golden_dir = pathlib.Path(args.golden_dir)

    print("\n=== per-layer comparison: golden .mem vs TFLite ===")
    header = (
        f"{'Layer':>5} | {'Shape':<18} | {'Mismatch':>14} | "
        f"{'Max|d|':>6} | {'Mean|d|':>8} | Verdict"
    )
    print(header)
    print("-" * len(header))

    n_bit_exact = 0
    n_within_tol = 0
    n_failed = 0
    for (idx, _buf, _base, words, _vlanes, H, W, C) in LAYER_MAP:
        gold_raw = load_golden_uram_mem(
            str(golden_dir / f"golden_layer{idx}_uram.mem"),
            num_words=words,
        )
        gold_hwc = unpack_uram(gold_raw, H, W, C)

        tflite_tensor = interp.get_tensor(mapping[idx]["observable_tensor"])
        stats = compare_layer(gold_hwc, tflite_tensor)
        v = verdict_for(stats, max_lsb=args.max_lsb)
        if v == "BIT-EXACT":
            n_bit_exact += 1
        elif v.startswith("WITHIN-TOL"):
            n_within_tol += 1
        else:
            n_failed += 1

        shape_str = f"{stats['shape'][0]}x{stats['shape'][1]}x{stats['shape'][2]}"
        mismatch_str = f"{stats['num_mismatch']:>6}/{stats['total']:<6}"
        print(
            f"{idx:>5} | {shape_str:<18} | {mismatch_str:>14} | "
            f"{stats['max_abs']:>6} | {stats['mean_abs']:>8.3f} | {v}"
        )
        if v != "BIT-EXACT" and stats["first_pos"]:
            print(f"        first diffs (h,w,c): {stats['first_pos']}")

    print("-" * len(header))
    total = len(LAYER_MAP)
    if args.max_lsb > 0:
        verdict = "PASS with tolerance" if n_failed == 0 else "FAIL"
        print(
            f"=== summary: {n_bit_exact}/{total} bit-exact, "
            f"{n_within_tol}/{total} within ±{args.max_lsb} LSB, "
            f"{n_failed} failed ({verdict}) ==="
        )
    else:
        print(
            f"=== summary: {n_bit_exact}/{total} bit-exact, "
            f"{n_failed} failed ==="
        )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
