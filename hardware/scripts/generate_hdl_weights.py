#!/usr/bin/env python3
"""
generate_hdl_weights.py
=======================
Reads a full-integer-quantised TFLite model and produces every synthesis-ready
artifact the HDL inference accelerator needs:

  - weight_rom.{coe,mem}     128-bit packed weight ROM (all layers)
  - bias_rom.{coe,mem}       INT32 bias ROM (per output channel)
  - m0_rom.{coe,mem}         Q31 per-channel multipliers
  - nshift_rom.{coe,mem}     Per-channel right-shift amounts
  - zp_in_rom.{coe,mem}      Per-layer input zero-points
  - zp_out_rom.{coe,mem}     Per-layer output zero-points
  - silu_lut.{coe,mem}       Per-layer SiLU LUTs (17 x 256)
  - layer_config.svh         SystemVerilog layer table header
  - rom_summary.json         Machine-readable parameter summary
  - weight_rom_golden.npz    NumPy archive for testbench verification

Usage
-----
    python generate_hdl_weights.py [model.tflite] [--out <dir>] [--verify]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# TFLite flatbuffer imports
# ---------------------------------------------------------------------------
try:
    from tflite.Model import Model
except ImportError:
    sys.exit("ERROR: 'tflite' package not found. Install with:\n"
             "  pip install tflite")

# TFLite tensor type codes -> numpy dtype
TFLITE_DTYPE = {
    0: np.float32, 1: np.float16, 2: np.int32,
    3: np.uint8,   4: np.int64,   7: np.int16, 9: np.int8,
}

CONV_2D = 3

# ---------------------------------------------------------------------------
# HDL layer configuration table
# ---------------------------------------------------------------------------
# Layer types: 2-bit encoding with meaningful bit fields
#   Bit 1: kernel size    (0 = 3x3,  1 = 1x1)
#   Bit 0: modifier       (3x3: 0=no pool, 1=pool)  (1x1: 0=SiLU, 1=linear)
CONV3      = 0  # 2'b00: 3x3 + SiLU
CONV3_POOL = 1  # 2'b01: 3x3 + SiLU + max pool
CONV1      = 2  # 2'b10: 1x1 + SiLU
CONV1_LIN  = 3  # 2'b11: 1x1 + linear

LAYER_TYPE_NAMES = {
    CONV3:      "CONV3",
    CONV3_POOL: "CONV3_POOL",
    CONV1:      "CONV1",
    CONV1_LIN:  "CONV1_LIN",
}


@dataclass
class HdlLayer:
    hdl_idx: int
    layer_type: int
    h_in: int
    w_in: int
    cin: int
    cout: int
    # Computed fields (filled after construction)
    cin_groups: int = 0
    kernel_size: int = 0   # 3 or 1
    wt_base_addr: int = 0
    qp_base_addr: int = 0
    wt_word_count: int = 0


# The 17 HDL layers mapped from TinyissimoYOLO TFLite.
# H/W are INPUT spatial dimensions (pre-pool for fused layers).
HDL_LAYERS = [
    HdlLayer( 0, CONV3_POOL, 256, 256,   3,  16),
    HdlLayer( 1, CONV3,      128, 128,  16,  16),
    HdlLayer( 2, CONV3_POOL, 128, 128,  16,  16),
    HdlLayer( 3, CONV3,       64,  64,  16,  32),
    HdlLayer( 4, CONV3_POOL,  64,  64,  32,  32),
    HdlLayer( 5, CONV3,       32,  32,  32,  64),
    HdlLayer( 6, CONV3_POOL,  32,  32,  64,  64),
    HdlLayer( 7, CONV3,       16,  16,  64,  64),
    HdlLayer( 8, CONV3_POOL,  16,  16,  64, 128),
    HdlLayer( 9, CONV3,        8,   8, 128, 128),
    HdlLayer(10, CONV1,        8,   8, 128,  24),
    HdlLayer(11, CONV3,        8,   8,  24,  64),
    HdlLayer(12, CONV3,        8,   8,  64,  64),
    HdlLayer(13, CONV1_LIN,    8,   8,  64,  64),
    HdlLayer(14, CONV3,        8,   8,  24,  24),
    HdlLayer(15, CONV3,        8,   8,  24,  24),
    HdlLayer(16, CONV1_LIN,    8,   8,  24,   3),
]

NUM_LAYERS = len(HDL_LAYERS)


# ---------------------------------------------------------------------------
# Quantisation helpers (from extract_tflite_params.py)
# ---------------------------------------------------------------------------
def decompose_scale(scale: float, n_bits: int = 31) -> tuple[int, int]:
    """Decompose scale into (m0, n_shift) where scale ~ m0 * 2^(-n_shift)."""
    if scale <= 0.0:
        return 0, 0
    log2_scale = math.log2(scale)
    n = max(0, math.ceil(-log2_scale))
    m0_float = scale * (2.0 ** n) * (2.0 ** n_bits)
    m0 = int(round(m0_float))
    if m0 >= (1 << n_bits):
        m0 >>= 1
        n -= 1
    n_shift = n + n_bits
    return int(m0), int(n_shift)


# ---------------------------------------------------------------------------
# TFLite tensor helpers
# ---------------------------------------------------------------------------
def get_tensor_data(model_buf: bytearray, model, tensor_idx: int) -> np.ndarray | None:
    sg = model.Subgraphs(0)
    t = sg.Tensors(tensor_idx)
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
    q = tensor.Quantization()
    if q is None:
        return [], []
    scales = [q.Scale(i) for i in range(q.ScaleLength())]
    zps = [q.ZeroPoint(i) for i in range(q.ZeroPointLength())]
    return scales, zps


# ---------------------------------------------------------------------------
# Extracted layer data
# ---------------------------------------------------------------------------
@dataclass
class ExtractedLayer:
    weights: np.ndarray       # int8 [Cout, kH, kW, Cin]
    biases: np.ndarray        # int32 [Cout]
    scale_in: float
    zp_in: int
    scale_out: float
    zp_out: int
    weight_scales: list       # per output channel
    m0_list: list[int] = field(default_factory=list)
    n_shift_list: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Extract conv layers from TFLite
# ---------------------------------------------------------------------------
def extract_conv_layers(model_path: str) -> list[ExtractedLayer]:
    """Extract all CONV_2D layers from the TFLite model."""
    with open(model_path, "rb") as fh:
        model_buf = bytearray(fh.read())

    model = Model.GetRootAs(model_buf, 0)
    sg = model.Subgraphs(0)

    op_codes = [model.OperatorCodes(i).BuiltinCode()
                for i in range(model.OperatorCodesLength())]

    layers = []
    for op_i in range(sg.OperatorsLength()):
        op = sg.Operators(op_i)
        code = op_codes[op.OpcodeIndex()]
        if code != CONV_2D:
            continue

        inputs = [op.Inputs(j) for j in range(op.InputsLength())]
        outputs = [op.Outputs(j) for j in range(op.OutputsLength())]
        if len(inputs) < 2:
            continue

        in_idx, w_idx = inputs[0], inputs[1]
        b_idx = inputs[2] if len(inputs) > 2 else -1
        out_idx = outputs[0]

        in_tensor = sg.Tensors(in_idx)
        w_tensor = sg.Tensors(w_idx)
        out_tensor = sg.Tensors(out_idx)

        in_scales, in_zps = get_quant_params(in_tensor)
        w_scales, _ = get_quant_params(w_tensor)
        out_scales, out_zps = get_quant_params(out_tensor)

        if not in_scales or not w_scales or not out_scales:
            continue

        scale_in = in_scales[0]
        scale_out = out_scales[0]
        zp_in = int(in_zps[0]) if in_zps else 0
        zp_out = int(out_zps[0]) if out_zps else 0

        weights = get_tensor_data(model_buf, model, w_idx)
        biases = get_tensor_data(model_buf, model, b_idx) if b_idx >= 0 else None

        if weights is None:
            continue

        if biases is None:
            n_out_ch = weights.shape[0]
            biases = np.zeros(n_out_ch, dtype=np.int32)

        # Compute per-channel m0 / n_shift
        m0_list = []
        n_shift_list = []
        for ws in w_scales:
            eff = (scale_in * ws) / scale_out
            m0, ns = decompose_scale(eff)
            m0_list.append(m0)
            n_shift_list.append(ns)

        layers.append(ExtractedLayer(
            weights=weights,
            biases=biases,
            scale_in=scale_in,
            zp_in=zp_in,
            scale_out=scale_out,
            zp_out=zp_out,
            weight_scales=list(w_scales),
            m0_list=m0_list,
            n_shift_list=n_shift_list,
        ))

    return layers


# ---------------------------------------------------------------------------
# Validate extracted layers against HDL table
# ---------------------------------------------------------------------------
def validate_layers(extracted: list[ExtractedLayer], c_par: int):
    if len(extracted) < NUM_LAYERS:
        sys.exit(f"ERROR: Expected {NUM_LAYERS} conv layers, found {len(extracted)}. "
                 f"Is this the correct model?")

    for i, hdl in enumerate(HDL_LAYERS):
        ext = extracted[i]
        cout, kh, kw, cin = ext.weights.shape

        # Determine expected kernel size from layer type
        expected_k = 1 if hdl.layer_type in (CONV1, CONV1_LIN) else 3

        errors = []
        if cin != hdl.cin:
            errors.append(f"Cin mismatch: TFLite={cin}, HDL={hdl.cin}")
        if cout != hdl.cout:
            errors.append(f"Cout mismatch: TFLite={cout}, HDL={hdl.cout}")
        if kh != expected_k or kw != expected_k:
            errors.append(f"Kernel mismatch: TFLite={kh}x{kw}, expected={expected_k}x{expected_k}")

        if errors:
            sys.exit(f"ERROR: Layer {i} validation failed:\n  " + "\n  ".join(errors))

        hdl.kernel_size = expected_k
        hdl.cin_groups = math.ceil(hdl.cin / c_par)

    print(f"  Validated {NUM_LAYERS} layers against HDL table.")


# ---------------------------------------------------------------------------
# Weight ROM packing
# ---------------------------------------------------------------------------
def pack_weight_rom(extracted: list[ExtractedLayer], c_par: int) -> list[list[int]]:
    """
    Pack all layer weights into 128-bit ROM words.

    Returns a list of 16-byte lists (each byte is uint8, representing the
    two's complement of the int8 weight).

    ROM layout per layer:
      For each k_out, for each cin_group, K consecutive words (K=9 or 1).
      Each word: 16 int8 weights at one kernel position across c_par channels.
    """
    rom_words = []
    qp_offset = 0

    for i, hdl in enumerate(HDL_LAYERS):
        ext = extracted[i]
        w = ext.weights  # [Cout, kH, kW, Cin]
        cout, kh, kw, cin = w.shape
        K = kh * kw
        cin_groups = hdl.cin_groups

        hdl.wt_base_addr = len(rom_words)
        hdl.qp_base_addr = qp_offset

        for k_out in range(cout):
            for cin_grp in range(cin_groups):
                for kp in range(K):
                    ky = kp // kw
                    kx = kp % kw
                    word = []
                    for c in range(c_par):
                        cin_idx = cin_grp * c_par + c
                        if cin_idx < cin:
                            val = int(w[k_out, ky, kx, cin_idx])
                        else:
                            val = 0  # zero-pad
                        word.append(val & 0xFF)
                    rom_words.append(word)

        hdl.wt_word_count = len(rom_words) - hdl.wt_base_addr
        qp_offset += cout

    return rom_words


# ---------------------------------------------------------------------------
# COE and MEM file writers
# ---------------------------------------------------------------------------
def bytes_to_hex128(word_bytes: list[int]) -> str:
    """Pack 16 uint8 bytes into a 32-char hex string (little-endian: byte[0] at LSB)."""
    val = 0
    for i, b in enumerate(word_bytes):
        val |= (b & 0xFF) << (i * 8)
    return f"{val:032x}"


def int32_to_hex(v: int) -> str:
    """Convert a signed int32 to 8-char hex (two's complement)."""
    return f"{v & 0xFFFFFFFF:08x}"


def int8_to_hex(v: int) -> str:
    """Convert a signed int8 to 2-char hex (two's complement)."""
    return f"{v & 0xFF:02x}"


def pack_qp_hex72(bias: int, m0: int, n_shift: int) -> str:
    """Pack bias[31:0] | m0[31:0] | n_shift[5:0] into an 18-char (72-bit) hex string.

    Bit layout:  [71:70] = 0  |  [69:64] = n_shift  |  [63:32] = m0  |  [31:0] = bias
    """
    word72 = ((n_shift & 0x3F) << 64) | ((m0 & 0xFFFFFFFF) << 32) | (bias & 0xFFFFFFFF)
    return f"{word72:018x}"


def write_coe(path: str, hex_entries: list[str], radix: int = 16):
    """Write a Vivado COE file."""
    with open(path, "w") as f:
        f.write(f"memory_initialization_radix={radix};\n")
        f.write("memory_initialization_vector=\n")
        for i, entry in enumerate(hex_entries):
            sep = ";" if i == len(hex_entries) - 1 else ","
            f.write(f"{entry}{sep}\n")


def write_mem(path: str, hex_entries: list[str], comment: str = ""):
    """Write a Verilog $readmemh compatible .mem file."""
    with open(path, "w") as f:
        if comment:
            f.write(f"// {comment}\n")
        for entry in hex_entries:
            f.write(f"{entry}\n")


def write_dual(out_dir: str, name: str, hex_entries: list[str],
               radix: int = 16, comment: str = ""):
    """Write both .coe and .mem files."""
    write_coe(os.path.join(out_dir, f"{name}.coe"), hex_entries, radix)
    write_mem(os.path.join(out_dir, f"{name}.mem"), hex_entries, comment)


# ---------------------------------------------------------------------------
# SiLU LUT generation
# ---------------------------------------------------------------------------
def generate_silu_lut(extracted: list[ExtractedLayer]) -> list[int]:
    """
    Generate per-layer SiLU LUTs for activation.

    For each layer with SiLU, maps each possible int8 input q in [-128..127]:
      1. Dequantize: x = (q - zp_out) * scale_out        (conv output domain)
      2. SiLU:       silu = x / (1 + exp(-x))             (= x * sigmoid(x))
      3. Quantize:   q_silu = clamp(round(silu / scale_in_next) + zp_in_next)

    The output quantization uses the next layer's input domain, so the LUT
    absorbs both the nonlinear activation and the cross-layer requantization.

    Returns a flat list of 17*256 uint8 values.
    """
    lut_flat = []

    for i, hdl in enumerate(HDL_LAYERS):
        ext = extracted[i]
        has_activation = hdl.layer_type not in (CONV1_LIN,)

        # Next layer's input quantization for the SiLU output domain
        ext_next = extracted[i + 1] if (has_activation and i + 1 < len(extracted)) else None

        lut = []
        for q in range(-128, 128):
            if has_activation and ext_next is not None:
                x_real = (q - ext.zp_out) * ext.scale_out
                silu_real = x_real / (1.0 + math.exp(-x_real))
                q_silu = int(round(silu_real / ext_next.scale_in)) + ext_next.zp_in
                q_silu = max(-128, min(127, q_silu))
            else:
                q_silu = 0  # unused for linear layers (hardware bypasses)
            lut.append(q_silu & 0xFF)
        lut_flat.extend(lut)

    return lut_flat


# ---------------------------------------------------------------------------
# SystemVerilog header generation
# ---------------------------------------------------------------------------
def generate_svh(out_dir: str, rom_depth: int, bias_depth: int,
                 act_lut_depth: int, model_path: str,
                 extracted: list[ExtractedLayer] | None = None):
    """Generate layer_config.svh with the layer table and ROM constants."""
    lines = [
        "// Auto-generated by generate_hdl_weights.py",
        f"// Model: {os.path.basename(model_path)}",
        f"// Date:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "// DO NOT EDIT — regenerate from the TFLite model.",
        "",
        "`ifndef LAYER_CONFIG_SVH",
        "`define LAYER_CONFIG_SVH",
        "",
        f"localparam int NUM_LAYERS        = {NUM_LAYERS};",
        f"localparam int C_PAR             = 16;",
        f"localparam int WEIGHT_ROM_DEPTH  = {rom_depth};",
        f"localparam int BIAS_ROM_DEPTH    = {bias_depth};",
        f"localparam int QP_ROM_DEPTH      = {bias_depth};",
        f"localparam int QP_PACKED_ROM_DEPTH = {bias_depth};",
        f"localparam int ACT_LUT_DEPTH     = {act_lut_depth};",
        "",
        "// Layer type encoding (2-bit)",
        "//   Bit 1: kernel size  (0 = 3x3,  1 = 1x1)",
        "//   Bit 0: modifier     (3x3: 0=no pool, 1=pool)  (1x1: 0=SiLU, 1=linear)",
        "localparam logic [1:0] CONV3      = 2'b00;",
        "localparam logic [1:0] CONV3_POOL = 2'b01;",
        "localparam logic [1:0] CONV1      = 2'b10;",
        "localparam logic [1:0] CONV1_LIN  = 2'b11;",
        "",
        "// Layer configuration struct",
        "typedef struct packed {",
        "    logic [1:0]        layer_type;",
        "    logic [8:0]        h_in;",
        "    logic [8:0]        w_in;",
        "    logic [7:0]        cin;",
        "    logic [7:0]        cout;",
        "    logic [3:0]        cin_grp;",
        "    logic [14:0]       wt_base;",
        "    logic [9:0]        qp_base;",
        "    logic signed [7:0] zp_in;",
        "    logic signed [7:0] zp_out;",
        "} layer_cfg_t;",
        "",
        "localparam layer_cfg_t LAYER_CFG [0:NUM_LAYERS-1] = '{",
        "    /* layer_idx | type | h_in | w_in | cin | cout | grp | wt_base | qp_base | zp_in | zp_out */",
    ]

    # One struct initialiser per layer
    for i, hdl in enumerate(HDL_LAYERS):
        zp_in  = extracted[i].zp_in  if extracted else 0
        zp_out = extracted[i].zp_out if extracted else 0
        sep = "," if i < NUM_LAYERS - 1 else ""
        lines.append(
            f"    /* {i:2d} */ "
            f"'{{ {LAYER_TYPE_NAMES[hdl.layer_type] + ',':<15s}"
            f" {hdl.h_in:4d}, {hdl.w_in:4d},"
            f" {hdl.cin:4d}, {hdl.cout:4d},"
            f" {hdl.cin_groups:2d},"
            f" 15'h{hdl.wt_base_addr:04x},"
            f" 10'h{hdl.qp_base_addr:03x},"
            f" {zp_in:5d}, {zp_out:5d}"
            f" }}{sep}"
        )
    lines.append("};")

    lines.extend(["", "`endif // LAYER_CONFIG_SVH", ""])

    svh_path = os.path.join(out_dir, "layer_config.svh")
    with open(svh_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------
def generate_summary(out_dir: str, extracted: list[ExtractedLayer],
                     rom_words: list, model_path: str):
    summary = {
        "model": os.path.basename(model_path),
        "generated": datetime.now().isoformat(),
        "c_par": 16,
        "num_layers": NUM_LAYERS,
        "weight_rom_depth": len(rom_words),
        "weight_rom_bytes": len(rom_words) * 16,
        "bram36_estimate": math.ceil(len(rom_words) * 128 / (36 * 1024)),
        "total_output_channels": sum(h.cout for h in HDL_LAYERS),
        "qp_packed_rom_depth": sum(h.cout for h in HDL_LAYERS),
        "qp_packed_rom_width_bits": 72,
        "layers": [],
    }

    for i, hdl in enumerate(HDL_LAYERS):
        ext = extracted[i]
        summary["layers"].append({
            "hdl_idx": hdl.hdl_idx,
            "layer_type": LAYER_TYPE_NAMES[hdl.layer_type],
            "h_in": hdl.h_in,
            "w_in": hdl.w_in,
            "cin": hdl.cin,
            "cout": hdl.cout,
            "kernel": hdl.kernel_size,
            "cin_groups": hdl.cin_groups,
            "wt_base_addr": hdl.wt_base_addr,
            "wt_word_count": hdl.wt_word_count,
            "qp_base_addr": hdl.qp_base_addr,
            "scale_in": ext.scale_in,
            "zp_in": ext.zp_in,
            "scale_out": ext.scale_out,
            "zp_out": ext.zp_out,
            "weight_shape": list(ext.weights.shape),
        })

    json_path = os.path.join(out_dir, "rom_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Golden NPZ for testbench
# ---------------------------------------------------------------------------
def generate_golden_npz(out_dir: str, rom_words: list,
                        extracted: list[ExtractedLayer]):
    # ROM as 2D array: [N, 16] of int8
    rom_int8 = np.array(
        [[((b + 128) % 256) - 128 for b in word] for word in rom_words],
        dtype=np.int8
    )

    biases_flat = np.concatenate([e.biases for e in extracted[:NUM_LAYERS]])
    m0_flat = np.array([v for e in extracted[:NUM_LAYERS] for v in e.m0_list],
                       dtype=np.int32)
    nshift_flat = np.array([v for e in extracted[:NUM_LAYERS] for v in e.n_shift_list],
                           dtype=np.int32)
    zp_in_flat = np.array([e.zp_in for e in extracted[:NUM_LAYERS]], dtype=np.int8)
    zp_out_flat = np.array([e.zp_out for e in extracted[:NUM_LAYERS]], dtype=np.int8)
    layer_bases = np.array([h.wt_base_addr for h in HDL_LAYERS], dtype=np.int32)
    qp_bases = np.array([h.qp_base_addr for h in HDL_LAYERS], dtype=np.int32)

    # Packed QP: [N, 3] where columns are (bias, m0, n_shift) per output channel
    qp_packed = np.column_stack([biases_flat, m0_flat, nshift_flat])

    npz_path = os.path.join(out_dir, "weight_rom_golden.npz")
    np.savez(npz_path,
             rom_words=rom_int8,
             layer_bases=layer_bases,
             qp_bases=qp_bases,
             biases=biases_flat,
             m0=m0_flat,
             n_shift=nshift_flat,
             zp_in=zp_in_flat,
             zp_out=zp_out_flat,
             qp_packed=qp_packed)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_weight_rom(rom_words: list, extracted: list[ExtractedLayer],
                      c_par: int) -> bool:
    """Re-derive ROM contents from TFLite weights and compare."""
    print("\n=== Verification ===")
    all_ok = True

    for i, hdl in enumerate(HDL_LAYERS):
        ext = extracted[i]
        w = ext.weights
        cout, kh, kw, cin = w.shape
        K = kh * kw
        cin_groups = hdl.cin_groups

        mismatches = 0
        addr = hdl.wt_base_addr

        for k_out in range(cout):
            for cin_grp in range(cin_groups):
                for kp in range(K):
                    ky = kp // kw
                    kx = kp % kw
                    rom_word = rom_words[addr]
                    for c in range(c_par):
                        cin_idx = cin_grp * c_par + c
                        expected = 0
                        if cin_idx < cin:
                            expected = int(w[k_out, ky, kx, cin_idx])
                        # Convert ROM uint8 back to int8
                        got = rom_word[c]
                        got_signed = got if got < 128 else got - 256
                        if got_signed != expected:
                            mismatches += 1
                    addr += 1

        status = "PASS" if mismatches == 0 else "FAIL"
        if mismatches > 0:
            all_ok = False
        print(f"  Layer {i:2d} ({LAYER_TYPE_NAMES[hdl.layer_type]:16s}): "
              f"{hdl.wt_word_count:6d} words, {status}"
              f"{f' ({mismatches} mismatches)' if mismatches else ''}")

    print(f"\n  Overall: {'ALL PASS' if all_ok else 'FAILURES DETECTED'}")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate HDL synthesis-ready weight/param files from TFLite model.")
    parser.add_argument("model", nargs="?",
                        default="software/models/tflite/tinyissimo_ptq_full_integer_quant.tflite",
                        help="Path to .tflite file")
    parser.add_argument("--out", default="hardware/weights/hdl",
                        help="Output directory (default: hardware/weights/hdl/)")
    parser.add_argument("--c-par", type=int, default=16,
                        help="Channel parallelism (default: 16)")
    parser.add_argument("--verify", action="store_true",
                        help="Run self-verification after generation")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"ERROR: File not found: {args.model}")

    os.makedirs(args.out, exist_ok=True)
    c_par = args.c_par

    # --- Step 1-2: Extract from TFLite ---
    print(f"Loading {args.model} ...")
    extracted = extract_conv_layers(args.model)
    print(f"  Found {len(extracted)} CONV_2D layers.")

    # --- Step 3: Validate ---
    validate_layers(extracted, c_par)

    # --- Step 4: Pack weight ROM ---
    print("Packing weight ROM ...")
    rom_words = pack_weight_rom(extracted, c_par)
    print(f"  Weight ROM: {len(rom_words)} words x 128-bit = "
          f"{len(rom_words) * 16 / 1024:.1f} KB")

    # --- Step 5: Write weight_rom.{coe,mem} ---
    print("Writing output files ...")
    wt_hex = [bytes_to_hex128(w) for w in rom_words]
    write_dual(args.out, "weight_rom", wt_hex,
               comment=f"weight_rom.mem - {len(rom_words)} x 128-bit words")

    # --- Step 6: Write bias, m0, nshift, zp ROMs ---
    biases_flat = []
    m0_flat = []
    nshift_flat = []
    zp_in_flat = []
    zp_out_flat = []

    for i in range(NUM_LAYERS):
        ext = extracted[i]
        biases_flat.extend(ext.biases.tolist())
        m0_flat.extend(ext.m0_list)
        nshift_flat.extend(ext.n_shift_list)
        zp_in_flat.append(ext.zp_in)
        zp_out_flat.append(ext.zp_out)

    write_dual(args.out, "bias_rom",
               [int32_to_hex(v) for v in biases_flat],
               comment=f"bias_rom.mem - {len(biases_flat)} x 32-bit INT32 biases")

    write_dual(args.out, "m0_rom",
               [int32_to_hex(v) for v in m0_flat],
               comment=f"m0_rom.mem - {len(m0_flat)} x 32-bit Q31 multipliers")

    write_dual(args.out, "nshift_rom",
               [int32_to_hex(v) for v in nshift_flat],
               comment=f"nshift_rom.mem - {len(nshift_flat)} x 32-bit shift amounts")

    write_dual(args.out, "zp_in_rom",
               [int8_to_hex(v) for v in zp_in_flat],
               comment=f"zp_in_rom.mem - {len(zp_in_flat)} x 8-bit input zero-points")

    write_dual(args.out, "zp_out_rom",
               [int8_to_hex(v) for v in zp_out_flat],
               comment=f"zp_out_rom.mem - {len(zp_out_flat)} x 8-bit output zero-points")

    # --- Step 6b: Packed QP ROM (bias + m0 + n_shift in 72-bit words) ---
    qp_packed = []
    for i in range(NUM_LAYERS):
        ext = extracted[i]
        for ch in range(ext.biases.shape[0]):
            qp_packed.append(pack_qp_hex72(
                int(ext.biases[ch]), ext.m0_list[ch], ext.n_shift_list[ch]))
    write_dual(args.out, "qp_packed_rom", qp_packed,
               comment=f"qp_packed_rom.mem - {len(qp_packed)} x 72-bit packed "
                       f"{{bias[31:0], m0[31:0], n_shift[5:0]}}")
    print(f"  QP packed ROM: {len(qp_packed)} entries x 72-bit")

    # --- Step 7: SiLU LUTs ---
    print("Generating SiLU LUTs ...")
    silu_flat = generate_silu_lut(extracted)
    write_dual(args.out, "silu_lut",
               [int8_to_hex(v) for v in silu_flat],
               comment=f"silu_lut.mem - {NUM_LAYERS} x 256 entries, layer N at addr N*256")

    # --- Step 8: SystemVerilog header ---
    total_qp = sum(h.cout for h in HDL_LAYERS)
    generate_svh(args.out, len(rom_words), total_qp, len(silu_flat), args.model,
                 extracted=extracted)

    # --- Step 9: Summary JSON + golden NPZ ---
    summary = generate_summary(args.out, extracted, rom_words, args.model)
    generate_golden_npz(args.out, rom_words, extracted)

    # Print summary table
    print(f"\n{'Idx':>3} {'Type':<16} {'HxW':>9} {'Cin':>4} {'Cout':>4} "
          f"{'Groups':>6} {'WtBase':>7} {'Words':>6} {'QPBase':>6}")
    print("-" * 72)
    for i, hdl in enumerate(HDL_LAYERS):
        print(f"{i:3d} {LAYER_TYPE_NAMES[hdl.layer_type]:<16} "
              f"{hdl.h_in:4d}x{hdl.w_in:<4d} {hdl.cin:4d} {hdl.cout:4d} "
              f"{hdl.cin_groups:6d} {hdl.wt_base_addr:7d} {hdl.wt_word_count:6d} "
              f"{hdl.qp_base_addr:6d}")
    print("-" * 72)
    print(f"{'Total':>3} {'':16} {'':>9} {'':>4} "
          f"{total_qp:4d} {'':>6} {'':>7} {len(rom_words):6d}")

    print(f"\nOutput files in: {args.out}/")
    print(f"  Weight ROM:  {len(rom_words):,d} words ({len(rom_words)*16/1024:.1f} KB)")
    print(f"  Bias ROM:    {len(biases_flat):,d} entries")
    print(f"  QP ROMs:     {len(m0_flat):,d} entries (m0 + nshift)")
    print(f"  ZP ROMs:     {len(zp_in_flat):,d} entries (in + out)")
    print(f"  SiLU LUT:    {len(silu_flat):,d} entries ({NUM_LAYERS} layers x 256)")
    print(f"  BRAM36 est:  ~{summary['bram36_estimate']}")

    # --- Step 10: Verify ---
    if args.verify:
        ok = verify_weight_rom(rom_words, extracted, c_par)
        if not ok:
            sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
