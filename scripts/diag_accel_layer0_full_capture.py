"""Diagnostic: capture the FULL layer 0 output range fmap_a[0..16383]
on silicon, before layer 2 overwrites the first ~1024 words.

The existing diag_accel_shift_analysis.py reads only fmap_a[4096..16383]
because the diag waits for inference to complete first, by which point
later layers have overwritten fmap_a[0..1023]. This script polls
LAYER_IDX during inference and reads fmap_a as soon as the FSM advances
past layer 0 (layer_idx >= 1).

Layer 1 writes to fmap_b (PP_BUF_SEL[1]=1), so it does not touch
fmap_a. Layer 2 writes to fmap_a[2048..4095], starting after layer 1
completes. Layer 1's runtime is ~2.6 ms (conv1 on 128x128x16). Reading
16384 fmap_a words via the AXI result window takes ~16 ms via per-word
MMIO but is much faster via the .array bulk path that read_results_raw
uses. We aim to fit the full read inside the layer-1 window so layer 0
output survives untouched.

Strategy:
  1. configure + start, then write 65536 pixels.
  2. Tight Python loop polling LAYER_IDX. When >= 1, layer 0 is done.
  3. Immediately read_window(0, fmap_a, 16384) — slides the result
     window across fmap_a in 320-word chunks via the existing driver.
  4. Save the captured fmap_a as a numpy array, plus layer 0's golden,
     and compute byte-exact / +1 shift hit rates across the FULL
     [0..16383] range.

This is the single most important data point for the +1 shift bug:
it tells us whether the shift is uniform across all 16384 words
(supporting the "compute write address is uniformly off by one"
hypothesis) or only present in some sub-range (which would point at
something address-specific).
"""
import hashlib
import pathlib
import time
import numpy as np
from pynq import Overlay
from software.overlay.drivers.tinyissimoyolo_accelerator import TinyissimoYoloAcceleratorDriver
from software.overlay.tests.checks import load_golden_uram_mem

BIT_PATH = pathlib.Path("hardware/output/playground.bit")
MEM_PATH = pathlib.Path("hardware/testbench/inference_hdl/pixels_layer0.mem")
GOLDEN_DIR = pathlib.Path("hardware/testbench/inference_hdl")

print(f"=== bitstream md5: {hashlib.md5(BIT_PATH.read_bytes()).hexdigest()} ===")

# Load the same image data the diag uses
with open(MEM_PATH) as f:
    mem_bytes = np.array([int(l.strip(), 16) for l in f if l.strip()], dtype=np.uint8)
mem_int8 = mem_bytes.view(np.int8).reshape(3, 256, 256)
image = (mem_int8.astype(np.int16) + 128).astype(np.uint8).transpose(1, 2, 0)

ol = Overlay(str(BIT_PATH), ignore_version=True)
drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)

# Run preload
print("\n=== preload ===")
drv.configure(mode=0)
drv.start()
drv.write_pixels(image)
print(f"  pixel_count={drv.pixel_count}")

# Tight poll for layer 1 (layer 0 done)
print("\n=== polling layer_idx ===")
t0 = time.monotonic()
while drv.layer_idx < 1:
    pass
t_layer1 = time.monotonic() - t0
print(f"  layer_idx hit 1 after {t_layer1*1000:.2f} ms")

# Read fmap_a[0..16383] immediately
print("\n=== reading fmap_a[0..16383] ===")
t0 = time.monotonic()
silicon_full = drv.read_window(0, buf=0, num_words=16384)
t_read = time.monotonic() - t0
print(f"  read shape: {silicon_full.shape}")
print(f"  read time: {t_read*1000:.2f} ms")
print(f"  layer_idx now: {drv.layer_idx}")
print(f"  status: {drv.read_status()}")

# Save the captured snapshot
np.save("layer0_silicon_full.npy", silicon_full)
print(f"  saved to layer0_silicon_full.npy")

# Load golden and compare across the FULL range
print("\n=== comparing against golden_layer0_uram.mem (full 16384 words) ===")
golden = load_golden_uram_mem(
    str(GOLDEN_DIR / "golden_layer0_uram.mem"),
    num_words=16384,
)
print(f"  golden shape: {golden.shape}")

# H1: silicon[k] == golden[k] (no shift)
h1_match = (silicon_full == golden).all(axis=1)
h1_count = int(h1_match.sum())
print(f"\n  H1 (no shift):     {h1_count}/16384 words match")

# H2: silicon[k] == golden[k+1] (+1 shift)
h2_match = (silicon_full[:-1] == golden[1:]).all(axis=1)
h2_count = int(h2_match.sum())
print(f"  H2 (+1 shift):     {h2_count}/16383 words match")

# H3: silicon[k] == golden[k-1] (-1 shift)
h3_match = (silicon_full[1:] == golden[:-1]).all(axis=1)
h3_count = int(h3_match.sum())
print(f"  H3 (-1 shift):     {h3_count}/16383 words match")

# Per-1024-word region histogram
print(f"\n  H1 match histogram (per 1024-word block):")
for blk in range(16):
    n = int(h1_match[blk*1024:(blk+1)*1024].sum())
    bar = "#" * (n // 17)
    print(f"    [{blk*1024:>5}..{(blk+1)*1024-1:>5}]: {n:>5}/1024  {bar}")

print(f"\n  H2 match histogram (per 1024-word block):")
for blk in range(16):
    seg_start = blk*1024
    seg_end = min((blk+1)*1024, 16383)
    seg_len = seg_end - seg_start
    if seg_len <= 0:
        continue
    n = int(h2_match[seg_start:seg_end].sum())
    bar = "#" * (n // 17)
    print(f"    [{seg_start:>5}..{seg_end-1:>5}]: {n:>5}/{seg_len:<4}  {bar}")

# Show first 5 words and the last word
print(f"\n=== sample inspection ===")
for k in [0, 1, 2, 3, 4, 5, 4094, 4095, 4096, 16382, 16383]:
    sil_str = " ".join(f"{x:>4}" for x in silicon_full[k][:6]) + " .."
    gld_str = " ".join(f"{x:>4}" for x in golden[k][:6]) + " .."
    match = "MATCH-H1" if (silicon_full[k] == golden[k]).all() else (
        "MATCH-H2" if k+1 < 16384 and (silicon_full[k] == golden[k+1]).all() else "DIFF"
    )
    print(f"  silicon[{k:>5}] = {sil_str}    golden[{k:>5}] = {gld_str}    {match}")

ol.free()
