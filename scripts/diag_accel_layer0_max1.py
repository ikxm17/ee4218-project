"""Diagnostic: capture the FULL layer 0 output range fmap_a[0..16383]
on silicon by stopping inference after layer 0 via the new max_layers
register (axil_regs.sv ADDR_MAX_LAYERS_R = 0x01C, default 17, set to
1 here).

Bitstream requirement: this script ONLY works against the rebuild that
includes the max_layers_run RTL plumbing (commit hash > 4cd99c2).
Verify by checking the bitstream md5 prints differs from
a7d3e9a47f9519144b4fb3a90e4556b0.

What this answers
-----------------
The pre-rebuild diag could only see fmap_a[4096..16383] of layer 0
because layers 2/4/6/8/10/12/15 overwrite the first 4096 words
before the host can read them (preload via AXI-Lite MMIO is much
slower than the FSM, so by the time we're done writing 65536 pixels,
the FSM has already finished all 17 layers). With max_layers=1 the
FSM stops after layer 0, fmap_a is undisturbed, and we can read
all 16384 words at our leisure.

Two key questions the captured data answers:
  Q1. Is silicon[0..4095] also +1-shifted (uniform shift across the
      whole layer 0 output), or only [4096..16383] (cascade-boundary
      localized)?
  Q2. Does silicon[16383] hold any meaningful value, or is it
      uninitialized? Useful for distinguishing "write addresses are
      systematically -1" from "first data is skipped".

Usage:
    ssh kria-01 'cd ~/workspace/ee4218-project && \\
        echo asdfzxcv | sudo -S XILINX_XRT=/usr PYTHONPATH=. \\
        /opt/ee4218/ee4218-venv/bin/python3 scripts/diag_accel_layer0_max1.py'
"""
import hashlib
import pathlib
import numpy as np
from pynq import Overlay
from software.overlay.drivers.tinyissimoyolo_accelerator import TinyissimoYoloAcceleratorDriver
from software.overlay.tests.checks import load_golden_uram_mem

BIT_PATH = pathlib.Path("hardware/output/playground.bit")
MEM_PATH = pathlib.Path("hardware/testbench/inference_hdl/pixels_layer0.mem")
GOLDEN_DIR = pathlib.Path("hardware/testbench/inference_hdl")

print(f"=== bitstream md5: {hashlib.md5(BIT_PATH.read_bytes()).hexdigest()} ===")

# Load image (matches diag_accel_shift_analysis.py)
with open(MEM_PATH) as f:
    mem_bytes = np.array([int(l.strip(), 16) for l in f if l.strip()], dtype=np.uint8)
mem_int8 = mem_bytes.view(np.int8).reshape(3, 256, 256)
image = (mem_int8.astype(np.int16) + 128).astype(np.uint8).transpose(1, 2, 0)

ol = Overlay(str(BIT_PATH), ignore_version=True)
drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)

# Stop inference after layer 0
print("\n=== set_max_layers(1) — stop after layer 0 ===")
drv.configure(mode=0)
drv.set_max_layers(1)
drv.start()

print("=== write 65536 pixels via FIFO ===")
drv.write_pixels(image)
print(f"  pixel_count={drv.pixel_count}")

print("\n=== wait for inference done (should fire after layer 0 only) ===")
if not drv.wait_done(timeout_s=2.0):
    raise TimeoutError("wait_done timed out — does the rebuild include max_layers_run?")
print(f"  layer_idx after done: {drv.layer_idx}")
print(f"  cycle_count: {drv.cycle_count}")
print(f"  status: {drv.read_status()}")

# Capture fmap_a[0..16383] in full
print("\n=== reading fmap_a[0..16383] ===")
silicon_full = drv.read_window(0, buf=0, num_words=16384)
print(f"  shape: {silicon_full.shape}")
np.save("layer0_silicon_max1.npy", silicon_full)

# Compare against golden
golden = load_golden_uram_mem(
    str(GOLDEN_DIR / "golden_layer0_uram.mem"),
    num_words=16384,
)

# Per-region histograms
def hist(label, mask):
    print(f"\n  {label}:")
    for blk in range(16):
        seg_start = blk*1024
        seg_end = min((blk+1)*1024, mask.shape[0])
        seg_len = seg_end - seg_start
        if seg_len <= 0:
            continue
        n = int(mask[seg_start:seg_end].sum())
        bar = "#" * (n // 17)
        print(f"    [{seg_start:>5}..{seg_end-1:>5}]: {n:>5}/{seg_len:<4}  {bar}")

h1_match = (silicon_full == golden).all(axis=1)
h2_match = (silicon_full[:-1] == golden[1:]).all(axis=1)
h3_match = (silicon_full[1:] == golden[:-1]).all(axis=1)

print(f"\n=== compare against golden_layer0_uram.mem ===")
print(f"  H1 (no shift):     {int(h1_match.sum())}/16384")
print(f"  H2 (+1 shift):     {int(h2_match.sum())}/16383")
print(f"  H3 (-1 shift):     {int(h3_match.sum())}/16383")

hist("H1 histogram per 1024-word block", h1_match)
hist("H2 histogram per 1024-word block", h2_match)

# Verdicts
h1_pct = h1_match.mean()
h2_pct = h2_match.mean()
print(f"\n=== verdict ===")
if h1_pct > 0.99:
    print(f"  H1 dominant ({h1_pct*100:.1f}%) — silicon matches golden bit-exact, NO BUG")
elif h2_pct > 0.99:
    print(f"  H2 dominant ({h2_pct*100:.1f}%) — UNIFORM +1 shift across all 16384 words")
elif h2_pct > 0.5:
    print(f"  H2 partial ({h2_pct*100:.1f}%) — +1 shift in some regions, not uniform")
else:
    print(f"  Neither — H1={h1_pct*100:.1f}% H2={h2_pct*100:.1f}% — needs deeper inspection")

# Sample inspection
print(f"\n=== sample inspection ===")
for k in [0, 1, 2, 3, 127, 128, 4094, 4095, 4096, 8191, 8192, 12287, 12288, 16382, 16383]:
    sil = " ".join(f"{x:>4}" for x in silicon_full[k][:6]) + " .."
    gld = " ".join(f"{x:>4}" for x in golden[k][:6]) + " .."
    if (silicon_full[k] == golden[k]).all():
        verdict = "MATCH-H1"
    elif k+1 < 16384 and (silicon_full[k] == golden[k+1]).all():
        verdict = "MATCH-H2"
    else:
        verdict = "DIFF"
    print(f"  silicon[{k:>5}] = {sil}    golden[{k:>5}] = {gld}    {verdict}")

ol.free()
