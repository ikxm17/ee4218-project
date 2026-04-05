#!/usr/bin/env python3
"""
generate_conv3d_golden.py
=========================
Generates golden reference data for RTL verification of inference_hdl.sv.

Takes the test image and model weights, performs a reference 3x3 convolution
for output channel 0 of layer 0, applies SiLU activation via the precomputed
LUT, and saves:
  - pixels_layer0.mem   : 256x256x3 INT8 input pixels (hex, for $readmemh)
  - golden_kout0.mem    : 256x256 INT8 expected output after SiLU (hex)

The reference pipeline matches the hardware exactly:
  1. Zero-pad input with zp_in (padding=1)
  2. For each output pixel: sum K*K*C_IN weighted products (no zp subtraction)
  3. Accumulate: acc = sum + bias
  4. Requantize: output = ((acc * m0) >> nshift) + zp_out, clamp to int8
  5. SiLU activation: output = silu_lut[layer_idx * 256 + (output + 128)]

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


def reference_conv3d_kout0(
    pixels_int8: np.ndarray,
    weights_rom: np.ndarray,
    bias: int,
    m0: int,
    nshift: int,
    zp_in: int,
    zp_out: int,
    layer_base: int,
    k: int = 3,
    cin: int = 3,
    c_par: int = 16,
) -> np.ndarray:
    """Run reference 3x3 convolution for output channel 0.

    Matches conv3d.v arithmetic exactly:
    - Padding pixels use zp_in as activation value
    - Weights are NOT zero-point subtracted from activations (conv3d feeds
      raw pixel bytes; the bias has the baked-in zp correction)
    - acc = sum(act * weight) + bias
    - output = clamp(((acc * m0) >> nshift) + zp_out, -128, 127)

    Args:
        pixels_int8: [H, W, C_IN] int8 input image
        weights_rom: [N, 16] int8 ROM words (from golden npz)
        bias: int32 bias for kout=0
        m0: int32 multiplier for kout=0
        nshift: int32 shift amount for kout=0
        zp_in: int8 input zero-point
        zp_out: int8 output zero-point
        layer_base: ROM word offset for layer 0
    """
    h, w, c = pixels_int8.shape
    assert c == cin
    pad = k // 2  # 1 for k=3

    # Extract weights for kout=0 from ROM
    # ROM layout for kout=0: K*K words starting at layer_base
    # Each word: 16 int8 values, only cin channels are meaningful
    cin_groups = (cin + c_par - 1) // c_par
    words_per_kout = k * k * cin_groups
    wt_start = layer_base

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
    layer_bases = g["layer_bases"]   # [17] int32
    qp_bases = g["qp_bases"]         # [17] int32
    biases = g["biases"]             # [827] int32
    m0_arr = g["m0"]                 # [827] int32
    nshift_arr = g["n_shift"]        # [827] int32
    zp_in_arr = g["zp_in"]           # [17] int8
    zp_out_arr = g["zp_out"]         # [17] int8

    # Layer 0, kout=0 parameters
    layer_idx = 0
    kout = 0
    layer_base = int(layer_bases[layer_idx])
    qp_base = int(qp_bases[layer_idx])
    bias = int(biases[qp_base + kout])
    m0 = int(m0_arr[qp_base + kout])
    nshift = int(nshift_arr[qp_base + kout])
    zp_in = int(zp_in_arr[layer_idx])
    zp_out = int(zp_out_arr[layer_idx])

    print(f"Layer {layer_idx}, kout {kout}:")
    print(f"  wt_base={layer_base}, qp_base={qp_base}")
    print(f"  bias={bias}, m0={m0}, nshift={nshift}")
    print(f"  zp_in={zp_in}, zp_out={zp_out}")

    # Load and preprocess image
    print(f"Loading image: {args.image}")
    pixels_uint8 = load_and_preprocess(args.image, size=256)
    pixels_int8 = uint8_to_int8(pixels_uint8)
    print(f"  Image shape: {pixels_int8.shape}, dtype: {pixels_int8.dtype}")
    print(f"  Range: [{pixels_int8.min()}, {pixels_int8.max()}]")

    # Run reference convolution
    print("Running reference convolution (kout=0)...")
    golden_output = reference_conv3d_kout0(
        pixels_int8=pixels_int8,
        weights_rom=rom_words,
        bias=bias,
        m0=m0,
        nshift=nshift,
        zp_in=zp_in,
        zp_out=zp_out,
        layer_base=layer_base,
        k=3,
        cin=3,
        c_par=16,
    )
    print(f"  Conv output shape: {golden_output.shape}")
    print(f"  Conv output range: [{golden_output.min()}, {golden_output.max()}]")

    # Apply SiLU activation via precomputed LUT
    print(f"Loading SiLU LUT: {args.lut}")
    silu_lut = load_silu_lut(args.lut)
    print(f"  LUT shape: {silu_lut.shape}")
    golden_output = apply_silu_lut(golden_output, silu_lut, layer_idx=0)
    print(f"  Activated output range: [{golden_output.min()}, {golden_output.max()}]")

    # Save pixels as .mem (channel-interleaved: C_IN values per spatial position)
    # Conv3d reads MAX_PARALLEL channels at each pixel address.
    # Layout: for each spatial position, channels 0..2 then zeros for 3..15
    print("Writing .mem files...")

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

    # Golden output: flat spatial
    golden_path = os.path.join(args.out, "golden_kout0.mem")
    write_hex_mem(golden_path, golden_output, "golden_kout0")

    # Also dump the first 9 weight ROM words for debug verification
    print("\nFirst 9 weight ROM words (kout=0, cin_grp=0):")
    for i in range(9):
        word = rom_words[layer_base + i]
        hex_str = " ".join(f"{int(b) & 0xFF:02x}" for b in word)
        print(f"  word[{i}]: {hex_str}")

    print(f"\nQP packed ROM entry 0: bias=0x{bias & 0xFFFFFFFF:08x}"
          f" m0=0x{m0 & 0xFFFFFFFF:08x} nshift={nshift}")

    print("\nDone.")


if __name__ == "__main__":
    main()
