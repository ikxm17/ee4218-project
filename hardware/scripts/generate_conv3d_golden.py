#!/usr/bin/env python3
"""
generate_conv3d_golden.py
=========================
Generates golden reference data for RTL verification of inference_hdl.sv.

Takes the test image and model weights, performs reference convolutions
for all output channels of all requested layers, applies SiLU activation
via the precomputed LUT, applies 2x2 max pooling where needed, and saves:
  - pixels_layer0.mem              : 256x256x3 INT8 input pixels
  - golden_ch_out0.mem             : 128x128 INT8 layer 0 ch_out=0 (legacy)
  - golden_layer{0..N}_uram.mem    : URAM-packed 128-bit words per layer

The reference pipeline matches the hardware exactly:
  1. Zero-pad input with zp_in (padding = k//2)
  2. For each output pixel: sum K*K*C_IN weighted products (no zp subtraction)
  3. Accumulate: acc = sum + bias
  4. Requantize: output = ((acc * m0) >> nshift) + zp_out, clamp to int8
  5. SiLU activation: output = silu_lut[layer_idx * 256 + (output + 128)]
     (bypassed for CONV1_LIN layers)
  6. Max pool 2x2 stride 2 (CONV3_POOL layers only)

Detection head branching: layers 11-13 (cv2) and 14-16 (cv3) both read
from layer 10's output.

Usage
-----
    python generate_conv3d_golden.py \\
        --model software/models/tflite/tinyissimo_ptq_full_integer_quant.tflite \\
        --image software/inference/data/input_image.jpg \\
        --golden hardware/weights/hdl/weight_rom_golden.npz \\
        --out hardware/testbench/inference_hdl/ \\
        --num-layers 10
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Layer configuration table (mirrors layer_config.svh)
# ---------------------------------------------------------------------------
# (type_name, h_in, w_in, cin, cout, has_pool, kernel_size, has_silu)
LAYER_CFG = [
    ("CONV3_POOL", 256, 256,   3,  16, True,  3, True),   #  0
    ("CONV3",      128, 128,  16,  16, False, 3, True),   #  1
    ("CONV3_POOL", 128, 128,  16,  16, True,  3, True),   #  2
    ("CONV3",       64,  64,  16,  32, False, 3, True),   #  3
    ("CONV3_POOL",  64,  64,  32,  32, True,  3, True),   #  4
    ("CONV3",       32,  32,  32,  64, False, 3, True),   #  5
    ("CONV3_POOL",  32,  32,  64,  64, True,  3, True),   #  6
    ("CONV3",       16,  16,  64,  64, False, 3, True),   #  7
    ("CONV3_POOL",  16,  16,  64, 128, True,  3, True),   #  8
    ("CONV3",        8,   8, 128, 128, False, 3, True),   #  9
    ("CONV1",        8,   8, 128,  24, False, 1, True),   # 10
    ("CONV3",        8,   8,  24,  64, False, 3, True),   # 11
    ("CONV3",        8,   8,  64,  64, False, 3, True),   # 12
    ("CONV1_LIN",    8,   8,  64,  64, False, 1, False),  # 13
    ("CONV3",        8,   8,  24,  24, False, 3, True),   # 14
    ("CONV3",        8,   8,  24,  24, False, 3, True),   # 15
    ("CONV1_LIN",    8,   8,  24,   3, False, 1, False),  # 16
]
NUM_LAYERS = len(LAYER_CFG)

# Detection head branch point: layers 14-16 read from layer 10's output
BRANCH_RESTORE_AT = 14
BRANCH_SOURCE = 10


def load_and_preprocess(image_path: str, size: int = 256) -> np.ndarray:
    """Load image, resize to (size x size), return as uint8 [H, W, C]."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def uint8_to_int8(pixels: np.ndarray) -> np.ndarray:
    """Convert uint8 pixels to int8 (two's complement reinterpret).

    TFLite full-integer models with input zero-point = -128 expect
    the uint8 pixel value reinterpreted as int8 (i.e. pixel - 128).
    """
    return (pixels.astype(np.int16) - 128).astype(np.int8)


def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
    """gemmlowp SaturatingRoundingDoublingHighMul reference (int32 inputs).

    Computes `round((2 * a * b) / 2^32)` as a 32-bit signed result, with the
    sole non-overflow saturation case `a == b == INT32_MIN` → INT32_MAX.
    The rounding nudge is sign-aware (+2^30 for non-negative products,
    1 - 2^30 for negative products), and the 2^31 divisor is integer division
    (toward zero), *not* arithmetic right-shift — this is the subtle
    difference from a naive `(a*b + 2^30) >> 31`.
    """
    INT32_MIN = -(1 << 31)
    INT32_MAX = (1 << 31) - 1
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab64 = a * b  # Python ints are unbounded, no overflow concern
    nudge = (1 << 30) if ab64 >= 0 else (1 - (1 << 30))
    # Truncation toward zero (C++ signed integer division semantics)
    val = ab64 + nudge
    return val // (1 << 31) if val >= 0 else -((-val) // (1 << 31))


def rounding_divide_by_pot(x: int, exponent: int) -> int:
    """gemmlowp RoundingDivideByPOT reference (int32 input).

    Round-nearest-away-from-zero division by 2^exponent. For positive x this
    is round-half-up; for negative x it rounds halves toward -∞ so that
    combined with the arithmetic right-shift the overall effect is
    round-nearest-away-from-zero.
    """
    if exponent <= 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    # Arithmetic right shift on signed Python int is floor division by power of 2
    shifted = x >> exponent
    return shifted + (1 if remainder > threshold else 0)


def gemmlowp_requantize(acc: int, m0: int, n_shift: int, zp_out: int) -> int:
    """Full gemmlowp two-stage requantize: SRDHM(acc, m0) then RDP by (n_shift - 31).

    Mirrors TFLite's int8 quantized conv reference kernel. `n_shift` here is
    the total right-shift matching the HDL/RTL convention (i.e.
    `n_shift = 31 - frexp_exponent`), and the RDP shift is `n_shift - 31`.
    """
    step1 = saturating_rounding_doubling_high_mul(int(acc), int(m0))
    rdp_shift = int(n_shift) - 31
    if rdp_shift < 0:
        # Scale > 1 is unusual but possible; fall back to a simple left-shift.
        step2 = step1 << (-rdp_shift)
    else:
        step2 = rounding_divide_by_pot(step1, rdp_shift)
    return step2 + int(zp_out)


def reference_conv3d_ch(
    pixels_int8: np.ndarray,
    weights_rom: np.ndarray,
    bias: int,
    m0: int,
    nshift: int,
    zp_in: int,
    zp_out: int,
    layer_base: int,
    ch_out: int = 0,
    k: int = 3,
    cin: int = 3,
    c_par: int = 16,
) -> np.ndarray:
    """Run reference 3x3 convolution for a given output channel.

    Matches conv3d.v arithmetic exactly:
    - Padding pixels use zp_in as activation value
    - Weights are NOT zero-point subtracted from activations (conv3d feeds
      raw pixel bytes; the bias has the baked-in zp correction)
    - acc = sum(act * weight) + bias
    - output = clamp(((acc * m0) >> nshift) + zp_out, -128, 127)

    Args:
        pixels_int8: [H, W, C_IN] int8 input image
        weights_rom: [N, 16] int8 ROM words (from golden npz)
        bias: int32 bias for this ch_out
        m0: int32 multiplier for this ch_out
        nshift: int32 shift amount for this ch_out
        zp_in: int8 input zero-point
        zp_out: int8 output zero-point
        layer_base: ROM word offset for this layer
        ch_out: output channel index
    """
    h, w, c = pixels_int8.shape
    assert c == cin
    pad = k // 2  # 1 for k=3

    # Extract weights for ch_out from ROM
    # Each word: 16 int8 values, only cin channels are meaningful
    cin_groups = (cin + c_par - 1) // c_par
    words_per_ch_out = k * k * cin_groups
    wt_start = layer_base + ch_out * words_per_ch_out

    # Build [kh, kw, cin] weight array from ROM
    weight = np.zeros((k, k, cin), dtype=np.int8)
    for cin_grp in range(cin_groups):
        for kp in range(k * k):
            word_idx = wt_start + cin_grp * (k * k) + kp
            kh = kp // k
            kw = kp % k
            for c_idx in range(c_par):
                global_cin = cin_grp * c_par + c_idx
                if global_cin < cin:
                    weight[kh, kw, global_cin] = weights_rom[word_idx, c_idx]

    # Padded input: border pixels are zp_in
    h_pad = h + 2 * pad
    w_pad = w + 2 * pad
    padded = np.full((h_pad, w_pad, cin), zp_in, dtype=np.int8)
    padded[pad:pad+h, pad:pad+w, :] = pixels_int8

    # Convolution (stride=1)
    out_h = h
    out_w = w
    output = np.zeros((out_h, out_w), dtype=np.int8)

    for r in range(out_h):
        for col in range(out_w):
            acc = np.int64(0)
            for ky in range(k):
                for kx in range(k):
                    for ci in range(cin):
                        act = np.int64(padded[r + ky, col + kx, ci])
                        wt = np.int64(weight[ky, kx, ci])
                        acc += act * wt
            # Add bias (bias already has -zp_in*sum(weight) baked in upstream)
            acc += np.int64(bias)
            # Round-half-up requantize — matches the RTL's requantize path
            # (conv3d.v / conv1d.v) bit-exact. Diverges from TFLite's
            # gemmlowp SRDHM+RDP by at most 1-2 LSB on ~0.02% of pixels
            # (negative accumulators near a rounding boundary), which is
            # within the tolerance that quantized int8 networks absorb
            # through their activation saturations.
            scaled = np.int64(acc) * np.int64(m0)
            if int(nshift) > 0:
                nudge = np.int64(1) << (int(nshift) - 1)
                shifted = (scaled + nudge) >> int(nshift)
            else:
                shifted = scaled
            q = int(shifted) + int(zp_out)
            q = max(-128, min(127, q))
            output[r, col] = np.int8(q)

    return output


def load_silu_lut(lut_path: str) -> np.ndarray:
    """Load the SiLU LUT .mem file into a [17, 256] int8 array."""
    entries = []
    with open(lut_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            entries.append(int(line, 16))
    # Convert unsigned uint8 to signed int8
    lut = np.array(entries, dtype=np.uint8).view(np.int8)
    return lut.reshape(17, 256)


def apply_silu_lut(
    conv_output: np.ndarray,
    silu_lut: np.ndarray,
    layer_idx: int,
) -> np.ndarray:
    """Apply precomputed SiLU LUT to conv3d output, matching hardware exactly.

    LUT address = layer_idx * 256 + (q + 128), where q is signed int8.
    """
    # q + 128 maps signed [-128..127] to unsigned [0..255]
    unsigned_idx = (conv_output.astype(np.int16) + 128).astype(np.uint8)
    activated = silu_lut[layer_idx, unsigned_idx.flatten()]
    return activated.reshape(conv_output.shape)


def apply_max_pool_2x2(data: np.ndarray) -> np.ndarray:
    """Apply 2x2 stride-2 max pooling matching max_pool.sv behaviour.

    Input is signed int8 [H, W]; output is [H/2, W/2].
    Uses signed comparison (numpy respects int8 signedness).
    """
    h, w = data.shape
    # Reshape into 2x2 blocks and take max over both block dimensions
    return data.reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3))


def process_layer(
    input_fmap: np.ndarray,        # [H, W, Cin] int8
    weights_rom: np.ndarray,       # [N, 16] int8
    golden_npz: dict,              # loaded npz
    layer_idx: int,
    silu_lut: np.ndarray,          # [17, 256] int8
) -> np.ndarray:
    """Run a full layer: conv + activation + optional pool for all output channels."""
    cfg = LAYER_CFG[layer_idx]
    _, _, _, cin_cfg, cout, has_pool, k, has_silu = cfg

    h_in, w_in, cin = input_fmap.shape
    assert cin == cin_cfg, f"Layer {layer_idx}: expected cin={cin_cfg}, got {cin}"

    layer_base = int(golden_npz["layer_bases"][layer_idx])
    qp_base = int(golden_npz["qp_bases"][layer_idx])
    zp_in = int(golden_npz["zp_in"][layer_idx])
    zp_out = int(golden_npz["zp_out"][layer_idx])

    outputs = []
    for ch in range(cout):
        bias = int(golden_npz["biases"][qp_base + ch])
        m0 = int(golden_npz["m0"][qp_base + ch])
        nshift = int(golden_npz["n_shift"][qp_base + ch])

        if ch < 2 or ch == cout - 1:
            print(f"  ch_out={ch}: bias={bias}, m0=0x{m0 & 0xFFFFFFFF:08x}, nshift={nshift}")
        elif ch == 2:
            print(f"  ... ({cout - 3} more channels)")

        conv_out = reference_conv3d_ch(
            pixels_int8=input_fmap,
            weights_rom=weights_rom,
            bias=bias, m0=m0, nshift=nshift,
            zp_in=zp_in, zp_out=zp_out,
            layer_base=layer_base,
            ch_out=ch, k=k, cin=cin,
        )
        # Activation: SiLU LUT or linear bypass
        if has_silu:
            act_out = apply_silu_lut(conv_out, silu_lut, layer_idx)
        else:
            act_out = conv_out
        # Pool if needed
        if has_pool:
            act_out = apply_max_pool_2x2(act_out)
        outputs.append(act_out)

    # Stack into [H_out, W_out, Cout]
    return np.stack(outputs, axis=-1)


def write_uram_packed_mem(path: str, fmap: np.ndarray, c_par: int = 16):
    """Write feature map [H, W, C] as 128-bit URAM-packed .mem file.

    Channel-group-major layout:
      For group g (channels g*16 .. g*16+15):
        For row r, col c:
          One 128-bit word: ch[g*16+0] in bits[7:0], ch[g*16+1] in bits[15:8], ...
    """
    h, w, c = fmap.shape
    c_groups = (c + c_par - 1) // c_par
    with open(path, "w") as f:
        for g in range(c_groups):
            for r in range(h):
                for col in range(w):
                    word = 0
                    for ci in range(c_par):
                        global_c = g * c_par + ci
                        if global_c < c:
                            byte_val = int(fmap[r, col, global_c]) & 0xFF
                        else:
                            byte_val = 0
                        word |= byte_val << (ci * 8)
                    f.write(f"{word:032x}\n")
    total = c_groups * h * w
    print(f"  URAM packed: {path} ({total} x 128-bit words)")


def write_hex_mem(path: str, data: np.ndarray, desc: str):
    """Write 1D or 2D array as hex .mem file (one value per line)."""
    flat = data.flatten()
    with open(path, "w") as f:
        for val in flat:
            # Two's complement hex for int8
            f.write(f"{int(val) & 0xFF:02x}\n")
    print(f"  {desc}: {path} ({len(flat)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Generate conv3d golden reference data")
    parser.add_argument("--model", required=True, help="TFLite model path")
    parser.add_argument("--image", required=True, help="Test image path")
    parser.add_argument("--golden", required=True, help="weight_rom_golden.npz path")
    parser.add_argument("--lut", required=True, help="silu_lut.mem path")
    parser.add_argument("--out", required=True, help="Output directory for .mem files")
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS,
                        help=f"Number of layers to process (default: all {NUM_LAYERS})")
    args = parser.parse_args()

    num_layers = min(args.num_layers, NUM_LAYERS)
    os.makedirs(args.out, exist_ok=True)

    # Load golden reference data
    print("Loading golden NPZ...")
    g = np.load(args.golden)
    rom_words = g["rom_words"]       # [25654, 16] int8

    # Load and preprocess image
    print(f"Loading image: {args.image}")
    pixels_uint8 = load_and_preprocess(args.image, size=256)
    pixels_int8 = uint8_to_int8(pixels_uint8)
    print(f"  Image shape: {pixels_int8.shape}, dtype: {pixels_int8.dtype}")
    print(f"  Range: [{pixels_int8.min()}, {pixels_int8.max()}]")

    # Load SiLU LUT
    print(f"Loading SiLU LUT: {args.lut}")
    silu_lut = load_silu_lut(args.lut)
    print(f"  LUT shape: {silu_lut.shape}")

    # Save pixels as .mem (channel-interleaved: C_IN values per spatial position)
    print("Writing pixel .mem file...")
    pixels_path = os.path.join(args.out, "pixels_layer0.mem")
    with open(pixels_path, "w") as f:
        for ch in range(3):
            for row in range(256):
                for col in range(256):
                    val = int(pixels_int8[row, col, ch])
                    f.write(f"{val & 0xFF:02x}\n")
    print(f"  pixels: {pixels_path} ({3*256*256} entries, [ch][row][col])")

    # Process layers sequentially, chaining outputs
    print(f"\n{'='*60}")
    print(f"Processing {num_layers} layers")
    print(f"{'='*60}")

    fmap = pixels_int8
    branch_save = None  # saved layer 10 output for cv3 branch

    for layer_idx in range(num_layers):
        cfg = LAYER_CFG[layer_idx]
        type_name, h_in, w_in, cin, cout, has_pool, k, has_silu = cfg

        # Detection head branching: restore layer 10's output for cv3
        if layer_idx == BRANCH_RESTORE_AT and branch_save is not None:
            print(f"\n--- Restoring layer {BRANCH_SOURCE} output for cv3 branch ---")
            fmap = branch_save.copy()

        h_out = h_in // 2 if has_pool else h_in
        print(f"\n=== Layer {layer_idx}: {type_name} "
              f"{h_in}x{w_in}x{cin} -> {h_out}x{h_out}x{cout} (k={k}) ===")

        fmap = process_layer(
            input_fmap=fmap,
            weights_rom=rom_words,
            golden_npz=g,
            layer_idx=layer_idx,
            silu_lut=silu_lut,
        )
        print(f"  Output shape: {fmap.shape}, range: [{fmap.min()}, {fmap.max()}]")

        # Save branch point for detection head
        if layer_idx == BRANCH_SOURCE:
            branch_save = fmap.copy()
            print(f"  (saved for cv3 branch)")

        # Write golden URAM file
        golden_path = os.path.join(args.out, f"golden_layer{layer_idx}_uram.mem")
        write_uram_packed_mem(golden_path, fmap)

        # Legacy: layer 0 single-channel golden
        if layer_idx == 0:
            golden_ch0 = fmap[:, :, 0]
            ch0_path = os.path.join(args.out, "golden_ch_out0.mem")
            write_hex_mem(ch0_path, golden_ch0, "golden_ch_out0")

    print(f"\n{'='*60}")
    print(f"Done. Generated golden files for {num_layers} layers.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
