#!/usr/bin/env python3
"""
generate_conv3d_golden.py
=========================
Generates golden reference data for RTL verification of inference_hdl.sv.

Takes the test image and model weights, performs reference 3x3 convolutions
for all output channels of layers 0 and 1, applies SiLU activation via the
precomputed LUT, applies 2x2 max pooling where needed, and saves:
  - pixels_layer0.mem       : 256x256x3 INT8 input pixels (hex, for $readmemh)
  - golden_ch_out0.mem      : 128x128 INT8 layer 0 ch_out=0 (backwards compat)
  - golden_layer0_uram.mem  : 128x128x16 URAM-packed 128-bit words (layer 0)
  - golden_layer1_uram.mem  : 128x128x16 URAM-packed 128-bit words (layer 1)

The reference pipeline matches the hardware exactly:
  1. Zero-pad input with zp_in (padding=1)
  2. For each output pixel: sum K*K*C_IN weighted products (no zp subtraction)
  3. Accumulate: acc = sum + bias
  4. Requantize: output = ((acc * m0) >> nshift) + zp_out, clamp to int8
  5. SiLU activation: output = silu_lut[layer_idx * 256 + (output + 128)]
  6. Max pool 2x2 stride 2 (CONV3_POOL layers only)

Usage
-----
    python generate_conv3d_golden.py \\
        --model software/models/tflite/tinyissimo_ptq_full_integer_quant.tflite \\
        --image software/inference/data/input_image.jpg \\
        --golden hardware/weights/hdl/weight_rom_golden.npz \\
        --out hardware/testbench/inference_hdl/
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image


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
            # Add bias
            acc += np.int64(bias)
            # Requantize: (acc * m0) >> nshift + zp_out
            scaled = np.int64(acc) * np.int64(m0)
            shifted = scaled >> int(nshift)
            q = shifted + np.int64(zp_out)
            # Clamp to int8
            q = max(-128, min(127, int(q)))
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
    pool: bool = False,
) -> np.ndarray:
    """Run a full layer: conv3d + SiLU + optional pool for all output channels."""
    h_in, w_in, cin = input_fmap.shape
    layer_base = int(golden_npz["layer_bases"][layer_idx])
    qp_base = int(golden_npz["qp_bases"][layer_idx])
    zp_in = int(golden_npz["zp_in"][layer_idx])
    zp_out = int(golden_npz["zp_out"][layer_idx])

    # Layer config (layers 0 and 1 only for this scope)
    layer_cfgs = [
        {"cin": 3, "cout": 16, "h_in": 256},   # layer 0
        {"cin": 16, "cout": 16, "h_in": 128},   # layer 1
    ]
    cout = layer_cfgs[layer_idx]["cout"]

    outputs = []
    for ch in range(cout):
        bias = int(golden_npz["biases"][qp_base + ch])
        m0 = int(golden_npz["m0"][qp_base + ch])
        nshift = int(golden_npz["n_shift"][qp_base + ch])

        print(f"  ch_out={ch}: bias={bias}, m0=0x{m0 & 0xFFFFFFFF:08x}, nshift={nshift}")

        conv_out = reference_conv3d_ch(
            pixels_int8=input_fmap,
            weights_rom=weights_rom,
            bias=bias, m0=m0, nshift=nshift,
            zp_in=zp_in, zp_out=zp_out,
            layer_base=layer_base,
            ch_out=ch, cin=cin,
        )
        # SiLU activation
        act_out = apply_silu_lut(conv_out, silu_lut, layer_idx)
        # Pool if needed
        if pool:
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
    args = parser.parse_args()

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
    # Conv3d reads MAX_PARALLEL channels at each pixel address.
    # Layout: for each spatial position, channels 0..2 then zeros for 3..15
    print("Writing pixel .mem file...")

    # Pixel mem: [channel][spatial] layout matching conv3d's addressing
    # conv3d addr = (round * ACT_SIZE²) + (row * ACT_SIZE + col)
    # For TOTAL_ROUNDS=1, addr = row * 256 + col
    # pixel_bram_data[c*8 +: 8] = pixel for channel c at that address
    # The testbench feeds this, so we save per-channel spatial arrays
    pixels_path = os.path.join(args.out, "pixels_layer0.mem")
    with open(pixels_path, "w") as f:
        for ch in range(3):
            for row in range(256):
                for col in range(256):
                    val = int(pixels_int8[row, col, ch])
                    f.write(f"{val & 0xFF:02x}\n")
    print(f"  pixels: {pixels_path} ({3*256*256} entries, [ch][row][col])")

    # Process Layer 0: CONV3_POOL 256x256x3 -> 128x128x16
    print("\n=== Layer 0: CONV3_POOL 256x256x3 -> 128x128x16 ===")
    layer0_out = process_layer(
        input_fmap=pixels_int8,
        weights_rom=rom_words,
        golden_npz=g,
        layer_idx=0,
        silu_lut=silu_lut,
        pool=True,
    )
    print(f"  Output shape: {layer0_out.shape}")
    print(f"  Output range: [{layer0_out.min()}, {layer0_out.max()}]")

    # Process Layer 1: CONV3 128x128x16 -> 128x128x16
    print("\n=== Layer 1: CONV3 128x128x16 -> 128x128x16 ===")
    layer1_out = process_layer(
        input_fmap=layer0_out,
        weights_rom=rom_words,
        golden_npz=g,
        layer_idx=1,
        silu_lut=silu_lut,
        pool=False,
    )
    print(f"  Output shape: {layer1_out.shape}")
    print(f"  Output range: [{layer1_out.min()}, {layer1_out.max()}]")

    # Write golden output files
    print("\nWriting golden .mem files...")

    # Layer 0 golden: single channel for backwards compat
    golden_ch0 = layer0_out[:, :, 0]
    golden_path = os.path.join(args.out, "golden_ch_out0.mem")
    write_hex_mem(golden_path, golden_ch0, "golden_ch_out0")

    # Layer 0 golden: full URAM packed (all 16 channels)
    golden_l0_path = os.path.join(args.out, "golden_layer0_uram.mem")
    write_uram_packed_mem(golden_l0_path, layer0_out)

    # Layer 1 golden: full URAM packed
    golden_l1_path = os.path.join(args.out, "golden_layer1_uram.mem")
    write_uram_packed_mem(golden_l1_path, layer1_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
