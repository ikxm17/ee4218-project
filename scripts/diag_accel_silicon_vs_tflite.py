"""Compare live silicon URAM against TFLite intermediate tensors, layer by layer.

The ground-truth on-board counterpart to `scripts/verify_goldens_vs_tflite.py`.
Instead of loading golden .mem files from disk, this loads the playground
bitstream, drives the accelerator through the existing
`TinyissimoYoloAcceleratorDriver`, and reads each layer's URAM window after
stopping inference with `set_max_layers(N+1)`. The reference tensor comes from
TFLite run on the exact same input bytes (reconstructed from pixels_layer0.mem)
so silicon ↔ TFLite can be diffed bit-exact.

Run via:
    ssh kria-01 'cd ~/workspace/ee4218-project && \\
        echo asdfzxcv | sudo -S XILINX_XRT=/usr PYTHONPATH=. \\
        /opt/ee4218/ee4218-venv/bin/python3 scripts/diag_accel_silicon_vs_tflite.py'

Saves silicon + TFLite snapshots to `silicon_vs_tflite.npz` for post-mortem.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pynq import Overlay  # noqa: E402

from software.overlay.drivers.tinyissimoyolo_accelerator import (  # noqa: E402
    TinyissimoYoloAcceleratorDriver,
)


# Authoritative per-layer URAM map — see scripts/diag_accel_per_layer_sweep.py
# and scripts/verify_goldens_vs_tflite.py. Must stay in sync with those files.
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

# (h_in, w_in, cin, cout, has_pool, has_silu) — subset of LAYER_CFG from
# hardware/scripts/generate_conv3d_golden.py used to validate TFLite op mapping.
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

    Prefers `tflite_runtime` (installed in the board venv); falls back to
    `tensorflow.lite` if unavailable. Prints which backend was used.
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


def image_from_pixels_mem(mem_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct both the uint8 HWC image and the (1,256,256,3) TFLite tensor.

    Returns:
        (image_hwc, tflite_input):
            image_hwc is (256, 256, 3) uint8 for TinyissimoYoloAcceleratorDriver
                .write_pixels.
            tflite_input is (1, 256, 256, 3) uint8 for interpreter.set_tensor.

    Both originate from the same `pixels_layer0.mem` bytes (channel-major int8,
    3×256×256). This guarantees the silicon and TFLite see IDENTICAL input — no
    JPEG/resampling/colorspace path in between.
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
    tflite_input = np.expand_dims(image_hwc, 0)
    assert image_hwc.dtype == np.uint8 and image_hwc.shape == (256, 256, 3)
    assert tflite_input.dtype == np.uint8 and tflite_input.shape == (1, 256, 256, 3)
    return image_hwc, tflite_input


def build_hdl_to_tflite_map(interp) -> list[dict]:
    """Walk the TFLite op graph and map HDL layers 0..16 → observable tensors.

    Identical logic to `scripts/verify_goldens_vs_tflite.build_hdl_to_tflite_map`
    — duplicated intentionally so each script is self-contained. See that file's
    docstring for the strategy. The two copies MUST stay in sync; if you change
    one, update the other.
    """
    ops = interp._get_ops_details()
    tdet = {t["index"]: t for t in interp.get_tensor_details()}

    conv_ops = [(i, op) for i, op in enumerate(ops) if op["op_name"] == "CONV_2D"]
    if len(conv_ops) < 17:
        raise RuntimeError(
            f"expected >=17 CONV_2D ops in TFLite graph, found {len(conv_ops)}"
        )

    def find_next(op_name: str, consumer_of: int, start: int) -> int | None:
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

        filt_idx = int(conv_op["inputs"][1])
        filt_shape = tdet[filt_idx]["shape"].tolist()
        in_idx = int(conv_op["inputs"][0])
        in_shape = tdet[in_idx]["shape"].tolist()
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

    Mirror of `scripts/verify_goldens_vs_tflite.unpack_uram` — duplicated by
    design so this script has no cross-import on its sibling.
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
    silicon_hwc: np.ndarray, tflite_tensor: np.ndarray
) -> dict:
    """Diff unpacked silicon [H, W, C] against TFLite NHWC tensor (batch stripped).

    Mirror of `scripts/verify_goldens_vs_tflite.compare_layer`.
    """
    if tflite_tensor.ndim == 4 and tflite_tensor.shape[0] == 1:
        tflite_hwc = tflite_tensor[0]
    else:
        tflite_hwc = tflite_tensor
    if silicon_hwc.shape != tflite_hwc.shape:
        raise ValueError(
            f"shape mismatch: silicon={silicon_hwc.shape} tflite={tflite_hwc.shape}"
        )
    if silicon_hwc.dtype != np.int8 or tflite_hwc.dtype != np.int8:
        raise ValueError(
            f"dtype mismatch: silicon={silicon_hwc.dtype} tflite={tflite_hwc.dtype}"
        )
    diff = silicon_hwc.astype(np.int32) - tflite_hwc.astype(np.int32)
    abs_diff = np.abs(diff)
    mismatches = np.argwhere(diff != 0)
    first_positions = [tuple(int(x) for x in row) for row in mismatches[:4]]
    return {
        "shape":         silicon_hwc.shape,
        "num_mismatch":  int((diff != 0).sum()),
        "total":         int(diff.size),
        "max_abs":       int(abs_diff.max()) if abs_diff.size else 0,
        "mean_abs":      float(abs_diff.mean()) if abs_diff.size else 0.0,
        "first_pos":     first_positions,
    }


def verdict_for(stats: dict) -> str:
    """Classify per-layer stats into BIT-EXACT / CLOSE / DIVERGED."""
    if stats["num_mismatch"] == 0:
        return "BIT-EXACT"
    if stats["max_abs"] <= 1:
        return "CLOSE (max<=1)"
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
        "--bitstream",
        default="hardware/output/playground.bit",
        help="Path to the playground.bit to program the FPGA",
    )
    parser.add_argument(
        "--snapshot",
        default="silicon_vs_tflite.npz",
        help="Output .npz with per-layer silicon + TFLite tensors",
    )
    args = parser.parse_args()

    bit_path = pathlib.Path(args.bitstream)
    print(
        f"=== bitstream md5: {hashlib.md5(bit_path.read_bytes()).hexdigest()} "
        f"({bit_path}) ==="
    )

    interp = load_interpreter(args.model)
    image_hwc, tflite_input = image_from_pixels_mem(args.pixels_mem)

    in_idx = interp.get_input_details()[0]["index"]
    interp.set_tensor(in_idx, tflite_input)
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

    ol = Overlay(str(bit_path), ignore_version=True)
    drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)

    snapshots: dict[str, np.ndarray] = {}

    print("\n=== per-layer comparison: silicon URAM vs TFLite ===")
    header = (
        f"{'Layer':>5} | {'Shape':<18} | {'Mismatch':>14} | "
        f"{'Max|d|':>6} | {'Mean|d|':>8} | {'cyc':>9} | Verdict"
    )
    print(header)
    print("-" * len(header))

    n_bit_exact = 0
    n_failed = 0
    try:
        for (idx, buf, base, words, _vlanes, H, W, C) in LAYER_MAP:
            drv.configure(mode=0)
            drv.set_max_layers(idx + 1)
            drv.start()
            drv.write_pixels(image_hwc)
            if not drv.wait_done(timeout_s=3.0):
                print(f"  layer {idx}: TIMEOUT waiting for done — ABORTING")
                n_failed += 1
                break

            sil_raw = drv.read_window(base, buf, words)
            snapshots[f"layer{idx:02d}"] = sil_raw
            sil_hwc = unpack_uram(sil_raw, H, W, C)

            tflite_tensor = interp.get_tensor(mapping[idx]["observable_tensor"])
            snapshots[f"tflite_layer{idx:02d}"] = tflite_tensor

            stats = compare_layer(sil_hwc, tflite_tensor)
            v = verdict_for(stats)
            if v == "BIT-EXACT":
                n_bit_exact += 1
            else:
                n_failed += 1

            shape_str = f"{stats['shape'][0]}x{stats['shape'][1]}x{stats['shape'][2]}"
            mismatch_str = f"{stats['num_mismatch']:>6}/{stats['total']:<6}"
            print(
                f"{idx:>5} | {shape_str:<18} | {mismatch_str:>14} | "
                f"{stats['max_abs']:>6} | {stats['mean_abs']:>8.3f} | "
                f"{drv.cycle_count:>9} | {v}"
            )
            if v != "BIT-EXACT" and stats["first_pos"]:
                print(f"        first diffs (h,w,c): {stats['first_pos']}")
    finally:
        if snapshots:
            np.savez(args.snapshot, **snapshots)
            print(f"\n=== saved {len(snapshots)} snapshots to {args.snapshot} ===")
        ol.free()

    print("-" * len(header))
    print(
        f"=== summary: {n_bit_exact}/{len(LAYER_MAP)} bit-exact, "
        f"{n_failed} failed ==="
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
