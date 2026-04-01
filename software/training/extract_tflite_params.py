#!/usr/bin/env python3
"""
extract_tflite_params.py
========================
Extracts weights and quantization parameters from a full-integer-quantised
TFLite model and writes them into files ready to be consumed by a Verilog
HDL module.

Output directory layout
-----------------------
output/
  layer_<N>/
    weights.hex          - INT8 weights, one value per line (signed decimal)
    bias.hex             - INT32 biases, one value per line (signed decimal)
    zp_in.txt            - single int8 zero-point for the layer input tensor
    zp_out.txt           - single int8 zero-point for the layer output tensor
    m0.txt               - INT32 multiplier(s), one per output channel
    n_shift.txt          - shift amount(s), one per output channel
    meta.txt             - human-readable layer summary

Quantisation maths recap
------------------------
For each output channel k the effective floating-point rescaling factor is:

    effective_scale_k = (scale_in * weight_scale_k) / scale_out

This is decomposed into:

    effective_scale_k ≈ m0_k * 2^(-n_shift_k)

where m0_k is an unsigned 31-bit integer (stored as int32, MSB = 0) such
that 0.5 <= m0_k / 2^31 < 1, and n_shift_k >= 0.

The fixed-point multiply in Verilog is:

    acc_shifted = (acc * m0_k + rounding) >> n_shift_k

with the input zero-point subtracted before accumulation and the output
zero-point added after shifting.

Usage
-----
    python3 extract_tflite_params.py <model.tflite> [--out <output_dir>]
"""

import argparse
import math
import os
import struct
import sys

import flatbuffers
import numpy as np

# ---------------------------------------------------------------------------
# TFLite flatbuffer helpers
# ---------------------------------------------------------------------------
try:
    from tflite.Model import Model
    from tflite.BuiltinOperator import BuiltinOperator
except ImportError:
    sys.exit("ERROR: 'tflite' package not found.  Install with:\n"
             "  pip install tflite --break-system-packages")

# Operator codes we care about
CONV_2D          = 3
DEPTHWISE_CONV   = 4
FULLY_CONNECTED  = 29
CONV_OPS         = {CONV_2D, DEPTHWISE_CONV, FULLY_CONNECTED}

OP_NAMES = {
    CONV_2D:        "CONV_2D",
    DEPTHWISE_CONV: "DEPTHWISE_CONV_2D",
    FULLY_CONNECTED:"FULLY_CONNECTED",
}

# TFLite tensor type codes -> numpy dtype
TFLITE_DTYPE = {
    0: np.float32,
    1: np.float16,
    2: np.int32,
    3: np.uint8,
    4: np.int64,
    7: np.int16,
    9: np.int8,
}


# ---------------------------------------------------------------------------
# Quantisation decomposition
# ---------------------------------------------------------------------------
def decompose_scale(scale: float, n_bits: int = 31) -> tuple[int, int]:
    """
    Decompose a positive floating-point scale factor into (m0, n_shift) such
    that:

        scale ≈ m0 * 2^(-n_shift)

    m0 is a non-negative integer with its MSB set (i.e. 2^(n_bits-1) <= m0
    < 2^n_bits), stored as a signed 32-bit int.  n_shift >= 0.

    This matches the "double-high fixed-point" convention used in TFLite's
    reference kernels (gemmlowp MultiplyByQuantizedMultiplier).

    Parameters
    ----------
    scale  : positive float
    n_bits : number of fractional bits in m0 (default 31 for INT8 inference)

    Returns
    -------
    (m0, n_shift) both int
    """
    if scale <= 0.0:
        return 0, 0

    # Find the smallest n >= 0 such that scale * 2^n is in [0.5, 1.0)
    # equivalently: n = ceil(-log2(scale))  clamped to >= 0
    log2_scale = math.log2(scale)
    n = max(0, math.ceil(-log2_scale))

    # m0 in [2^(n_bits-1), 2^n_bits)
    m0_float = scale * (2.0 ** n) * (2.0 ** n_bits)
    m0 = int(round(m0_float))

    # If rounding pushed m0 to 2^n_bits, renormalise
    if m0 >= (1 << n_bits):
        m0 >>= 1
        n  -= 1

    n_shift = n + n_bits  # total right-shift to apply after multiply
    return int(m0), int(n_shift)


# ---------------------------------------------------------------------------
# Tensor data extraction
# ---------------------------------------------------------------------------
def get_tensor_data(model_buf: bytearray, model, tensor_idx: int) -> np.ndarray | None:
    """Return the numpy array stored in a tensor's buffer, or None."""
    sg = model.Subgraphs(0)
    t  = sg.Tensors(tensor_idx)
    buf_idx = t.Buffer()
    if buf_idx == 0:
        return None
    buf = model.Buffers(buf_idx)
    if buf.DataLength() == 0:
        return None

    raw = bytes([buf.Data(i) for i in range(buf.DataLength())])
    dtype = TFLITE_DTYPE.get(t.Type())
    if dtype is None:
        return None
    arr = np.frombuffer(raw, dtype=dtype)
    shape = [t.Shape(j) for j in range(t.ShapeLength())]
    if shape:
        arr = arr.reshape(shape)
    return arr


def get_quant_params(tensor):
    """Return (scales_list, zeropoints_list) for a tensor."""
    q = tensor.Quantization()
    if q is None:
        return [], []
    scales = [q.Scale(i)     for i in range(q.ScaleLength())]
    zps    = [q.ZeroPoint(i) for i in range(q.ZeroPointLength())]
    return scales, zps


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------
def write_hex_int8(path: str, arr: np.ndarray):
    """Write int8 weight array as signed decimal integers, one per line."""
    flat = arr.flatten().astype(np.int8)
    with open(path, "w") as f:
        for v in flat:
            f.write(f"{int(v)}\n")


def write_hex_int32(path: str, arr: np.ndarray):
    """Write int32 bias array as signed decimal integers, one per line."""
    flat = arr.flatten().astype(np.int32)
    with open(path, "w") as f:
        for v in flat:
            f.write(f"{int(v)}\n")


def write_int_list(path: str, values: list, comment: str = ""):
    """Write a list of integers, one per line, with an optional header comment."""
    with open(path, "w") as f:
        if comment:
            f.write(f"// {comment}\n")
        for v in values:
            f.write(f"{int(v)}\n")


def write_meta(path: str, info: dict):
    """Write human-readable metadata."""
    with open(path, "w") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------
def extract(model_path: str, out_dir: str):
    print(f"Loading {model_path} ...")
    with open(model_path, "rb") as fh:
        model_buf = bytearray(fh.read())

    model = Model.GetRootAs(model_buf, 0)
    sg    = model.Subgraphs(0)

    # Build opcode lookup
    op_codes = [model.OperatorCodes(i).BuiltinCode()
                for i in range(model.OperatorCodesLength())]

    conv_layer_idx = 0
    for op_i in range(sg.OperatorsLength()):
        op   = sg.Operators(op_i)
        code = op_codes[op.OpcodeIndex()]
        if code not in CONV_OPS:
            continue

        # Tensor indices
        inputs  = [op.Inputs(j)  for j in range(op.InputsLength())]
        outputs = [op.Outputs(j) for j in range(op.OutputsLength())]

        if len(inputs) < 2:
            continue

        in_idx  = inputs[0]
        w_idx   = inputs[1]
        b_idx   = inputs[2] if len(inputs) > 2 else -1
        out_idx = outputs[0]

        in_tensor  = sg.Tensors(in_idx)
        w_tensor   = sg.Tensors(w_idx)
        out_tensor = sg.Tensors(out_idx)
        b_tensor   = sg.Tensors(b_idx) if b_idx >= 0 else None

        # --- quantisation parameters ---
        in_scales,  in_zps  = get_quant_params(in_tensor)
        w_scales,   w_zps   = get_quant_params(w_tensor)
        out_scales, out_zps = get_quant_params(out_tensor)

        if not in_scales or not w_scales or not out_scales:
            print(f"  [Op {op_i}] Skipping - missing quantisation params")
            continue

        scale_in  = in_scales[0]
        scale_out = out_scales[0]
        zp_in     = int(in_zps[0])  if in_zps  else 0
        zp_out    = int(out_zps[0]) if out_zps else 0

        n_out_ch = len(w_scales)  # one scale per output channel

        # Compute per-channel effective scale and decompose
        m0_list      = []
        n_shift_list = []
        eff_scales   = []
        for w_s in w_scales:
            eff = (scale_in * w_s) / scale_out
            m0, ns = decompose_scale(eff)
            m0_list.append(m0)
            n_shift_list.append(ns)
            eff_scales.append(eff)

        # --- weight / bias tensors ---
        weights = get_tensor_data(model_buf, model, w_idx)
        biases  = get_tensor_data(model_buf, model, b_idx) if b_idx >= 0 else None

        if weights is None:
            print(f"  [Op {op_i}] Skipping - no weight data in buffer")
            continue

        # --- output directory ---
        layer_dir = os.path.join(out_dir, f"layer_{conv_layer_idx:02d}")
        os.makedirs(layer_dir, exist_ok=True)

        w_shape  = list(weights.shape)
        op_name  = OP_NAMES.get(code, f"OP_{code}")

        # Tensor names (strip b'' wrapper)
        def tname(t): return t.Name().decode("utf-8") if t.Name() else "?"

        # ---- write files ----
        write_hex_int8(os.path.join(layer_dir, "weights.hex"), weights)

        if biases is not None:
            write_hex_int32(os.path.join(layer_dir, "bias.hex"), biases)
        else:
            # Write zeros so downstream HDL always has a bias file
            write_int_list(os.path.join(layer_dir, "bias.hex"),
                           [0] * n_out_ch, "no bias - zeros inserted")

        write_int_list(os.path.join(layer_dir, "zp_in.txt"),  [zp_in],
                       f"input zero-point for {tname(in_tensor)}")
        write_int_list(os.path.join(layer_dir, "zp_out.txt"), [zp_out],
                       f"output zero-point for {tname(out_tensor)}")
        write_int_list(os.path.join(layer_dir, "m0.txt"),      m0_list,
                       "per-channel multiplier (Q31 unsigned, stored as int32)")
        write_int_list(os.path.join(layer_dir, "n_shift.txt"), n_shift_list,
                       "per-channel right-shift amount")

        # metadata
        meta = {
            "layer_index"     : conv_layer_idx,
            "op_index_in_graph": op_i,
            "op_type"         : op_name,
            "weight_shape"    : str(w_shape),
            "weight_dtype"    : "int8",
            "bias_dtype"      : "int32",
            "input_tensor"    : tname(in_tensor),
            "input_scale"     : f"{scale_in:.10g}",
            "input_zp"        : zp_in,
            "output_tensor"   : tname(out_tensor),
            "output_scale"    : f"{scale_out:.10g}",
            "output_zp"       : zp_out,
            "n_output_channels": n_out_ch,
            "weight_scales_sample": str([f"{s:.6g}" for s in w_scales[:4]]) + ("..." if n_out_ch > 4 else ""),
            "eff_scales_sample": str([f"{s:.6g}" for s in eff_scales[:4]]) + ("..." if n_out_ch > 4 else ""),
            "m0_sample"       : str(m0_list[:4]) + ("..." if n_out_ch > 4 else ""),
            "n_shift_sample"  : str(n_shift_list[:4]) + ("..." if n_out_ch > 4 else ""),
            "verilog_note"    : (
                "acc_out = (acc_in - zp_in already subtracted before MAC);"
                " result = ((acc * m0 + 2^(n_shift-1)) >> n_shift) + zp_out;"
                " clamp to [-128, 127]"
            ),
        }
        write_meta(os.path.join(layer_dir, "meta.txt"), meta)

        print(f"  [layer {conv_layer_idx:02d}] op={op_i:3d}  {op_name:20s}"
              f"  weights={w_shape}  n_ch={n_out_ch}")
        conv_layer_idx += 1

    print(f"\nDone. {conv_layer_idx} layers extracted → {out_dir}/")
    print_usage_guide(out_dir)


# ---------------------------------------------------------------------------
# Usage guide
# ---------------------------------------------------------------------------
def print_usage_guide(out_dir: str):
    guide = f"""
=== Verilog Integration Guide ===

Each layer_{'{N:02d}'} directory contains:

  weights.hex   - INT8 weight values (flattened, shape-first order)
                  Shape: [out_ch, kH, kW, in_ch] for CONV_2D
                  Load with $readmemh or $readmemb, or parse as signed decimal.

  bias.hex      - INT32 bias values, one per output channel.

  zp_in.txt     - Single INT8 value.  Subtract from every input activation
                  BEFORE the MAC accumulation.

  zp_out.txt    - Single INT8 value.  Add to the result AFTER rescaling.

  m0.txt        - Per-channel INT32 multiplier (Q31, value in [2^30, 2^31)).
                  One entry per output channel.

  n_shift.txt   - Per-channel right-shift amount (unsigned integer).
                  One entry per output channel.

Fixed-point rescaling (per output channel k):
  1. acc = sum_over_ij( w[k,i,j] * (x[i,j] - zp_in) ) + bias[k]
  2. scaled = (acc * m0[k] + (1 << (n_shift[k]-1))) >> n_shift[k]
  3. y[k]   = clamp(scaled + zp_out, -128, 127)

Note: m0 is an unsigned 31-bit value stored as int32 (MSB always 0).
      The multiply acc * m0 requires a 64-bit (or 2x32-bit) intermediate.
"""
    guide_path = os.path.join(out_dir, "VERILOG_GUIDE.txt")
    with open(guide_path, "w") as f:
        f.write(guide)
    print(guide)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract weights & quant params from a full-integer TFLite model.")
    parser.add_argument("model",  help="Path to .tflite file")
    parser.add_argument("--out",  default="tflite_extracted",
                        help="Output directory (default: tflite_extracted/)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"ERROR: File not found: {args.model}")

    os.makedirs(args.out, exist_ok=True)
    extract(args.model, args.out)


if __name__ == "__main__":
    main()
