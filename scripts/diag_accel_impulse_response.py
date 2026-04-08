"""Impulse-response silicon diagnostic — localize the +1 shift directly.

Write a clean zero-valued input image with a single non-zero "impulse"
pixel at a specific (row, col) position, run layer 0 (via set_max_layers=1),
and read back the layer 0 output. Check where the non-zero 3x3 region lands.

If conv3d is correct, the impulse at image(r, c) produces a non-zero
3x3 region at layer 0 output positions (r-1..r+1, c-1..c+1) clipped to
[0, 127] in the 128x128 layer 0 output (due to the 2x maxpool after
the 3x3 conv). Specifically:
  - For impulse at image(2, 2) in the 256x256 input, the conv produces
    non-zero values at conv_out(1..3, 1..3). After maxpool, the non-zero
    region is at pool(0..1, 0..1) in the 128x128 output.
  - The affected URAM words are at addresses 0, 1, 128, 129 in row-major.

If silicon shows the non-zero region at a different set of addresses,
we can directly measure the 1D displacement in URAM space.

This test completely sidesteps the int8/uint8 zero-point and channel-count
questions because we're only checking WHERE the non-zero values land,
not WHAT they are.

Usage:
    ssh kria-01 'cd ~/workspace/ee4218-project && \\
        echo asdfzxcv | sudo -S XILINX_XRT=/usr PYTHONPATH=. \\
        /opt/ee4218/ee4218-venv/bin/python3 scripts/diag_accel_impulse_response.py'
"""
import hashlib
import pathlib
import numpy as np
from pynq import Overlay
from software.overlay.drivers.tinyissimoyolo_accelerator import TinyissimoYoloAcceleratorDriver

BIT_PATH = pathlib.Path("hardware/output/playground.bit")

# Place the impulse at a distinctive interior position away from boundaries
# so the conv output is NOT affected by padding. Use row=4, col=4 in the
# 256x256 input image. The 3x3 conv window for image position (4, 4) reads
# input(3..5, 3..5), which is all zero except input(4, 4) = impulse.
IMPULSE_ROW = 4
IMPULSE_COL = 4
IMPULSE_VALUE = (255, 255, 255)  # saturated white (max uint8 on all 3 channels)

print(f"=== bitstream md5: {hashlib.md5(BIT_PATH.read_bytes()).hexdigest()} ===")
print(f"=== impulse at image({IMPULSE_ROW}, {IMPULSE_COL}) = RGB{IMPULSE_VALUE} ===")

# Build the input image: all zero-point (uint8 128 = int8 0), with one impulse
# Using uint8 128 ensures the int8 value after write_pixels() conversion is 0,
# which means most conv outputs are the bias term (constant per-channel).
# The non-bias contribution only comes from the single impulse position.
image = np.full((256, 256, 3), 128, dtype=np.uint8)  # int8 0 everywhere
image[IMPULSE_ROW, IMPULSE_COL] = IMPULSE_VALUE  # int8 127 at impulse

ol = Overlay(str(BIT_PATH), ignore_version=True)
drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)

print("\n=== set_max_layers(1) — run only layer 0 ===")
drv.configure(mode=0)
drv.set_max_layers(1)
drv.start()
drv.write_pixels(image)

print("=== wait for done ===")
if not drv.wait_done(timeout_s=3.0):
    raise TimeoutError("Inference timeout")
print(f"  cycle_count: {drv.cycle_count}")

# Read layer 0 output (fmap_a, 16384 words × 16 channels)
print("\n=== reading fmap_a[0..16383] ===")
sil = drv.read_window(base_addr=0, buf=0, num_words=16384)
print(f"  shape: {sil.shape}")

# Compute per-spatial-position "activity" — the max absolute value across
# channels. A zero-input produces a constant output (just the bias), so
# non-zero-relative positions will stand out.
#
# Actually: better — compare each position's values to the "background"
# value at position (0, 0) (which is far from the impulse but still zero-
# input, so its output is the pure bias term).
#
# Note: conv3d outputs 256x256, but layer 0 has 2x maxpool, so the URAM
# holds 128x128 = 16384 pool outputs.

# Since zero-input produces constant "bias" output, "activity" is defined
# as "differs from the bias baseline".
baseline = sil[0]  # (16,) int8 — the output at pool position (0, 0)
print(f"  baseline (pool[0, 0]): {baseline.tolist()}")

# Find positions where the output differs from the baseline
diff_mask = (sil != baseline).any(axis=1)  # (16384,) bool
diff_positions = np.where(diff_mask)[0]
print(f"  positions differing from baseline: {len(diff_positions)}")

if len(diff_positions) == 0:
    print("  NO DIFFERENCE FROM BASELINE — impulse didn't propagate!")
    print("  Either bug or baseline chose a non-baseline position.")
else:
    # Print each differing position with its 2D coordinates (in 128x128 pool space)
    print(f"\n  Differing URAM addresses (first 30):")
    for addr in diff_positions[:30]:
        row = addr // 128
        col = addr % 128
        delta = (sil[addr].astype(int) - baseline.astype(int))
        nonzero_channels = int((delta != 0).sum())
        max_abs_delta = int(np.abs(delta).max())
        print(f"    URAM[{addr:>5}] = pool({row:>3}, {col:>3})  "
              f"chs_differing={nonzero_channels:>2}  max|Δ|={max_abs_delta:>3}")

    # Compute expected positions for the impulse at image(4, 4)
    # 3x3 conv: non-zero conv outputs at image(3..5, 3..5) in 256x256
    # 2x maxpool: pool output at (I, J) = max of conv(2I..2I+1, 2J..2J+1)
    # Non-zero conv positions (3..5, 3..5) affect pool positions:
    #   row 3 → I=1 (since 2*1=2, 2*1+1=3), row 4 → I=2, row 5 → I=2
    #   col 3 → J=1, col 4 → J=2, col 5 → J=2
    # So affected pool positions: (1, 1), (1, 2), (2, 1), (2, 2)
    # In row-major (128 wide): addresses 1*128+1=129, 1*128+2=130, 2*128+1=257, 2*128+2=258
    expected_addrs = {
        1*128+1: "(1,1)",
        1*128+2: "(1,2)",
        2*128+1: "(2,1)",
        2*128+2: "(2,2)",
    }
    print(f"\n  Expected non-baseline URAM addresses for impulse at image(4, 4):")
    for addr, pos in expected_addrs.items():
        in_silicon = addr in diff_positions
        row = addr // 128
        col = addr % 128
        print(f"    URAM[{addr:>5}] = pool({row:>3}, {col:>3}) {pos}  "
              f"{'<-- differing on silicon' if in_silicon else 'NOT differing on silicon'}")

    # Measure the displacement: find the actual centroid of the differing region
    if len(diff_positions) > 0 and len(diff_positions) < 100:
        # Small enough region to analyze
        rows = diff_positions // 128
        cols = diff_positions % 128
        print(f"\n  Actual differing region: rows {rows.min()}..{rows.max()}, "
              f"cols {cols.min()}..{cols.max()}")
        expected_rows = [1, 2]
        expected_cols = [1, 2]
        print(f"  Expected region:         rows 1..2, cols 1..2")
        print(f"  Row displacement:        {rows.min() - 1:+d} to {rows.max() - 2:+d}")
        print(f"  Col displacement:        {cols.min() - 1:+d} to {cols.max() - 2:+d}")

    # Check specifically for the +1 row-major shift
    shifted_expected = {a - 1: p for a, p in expected_addrs.items()}
    print(f"\n  Under '+1 row-major shift' hypothesis, differing URAM addrs should be:")
    for addr, pos in shifted_expected.items():
        if addr < 0:
            addr = 16384 + addr  # wrap
        in_silicon = addr in diff_positions
        row = addr // 128
        col = addr % 128
        print(f"    URAM[{addr:>5}] = pool({row:>3}, {col:>3}) {pos}(-1)  "
              f"{'<-- differing' if in_silicon else 'NOT differing'}")

ol.free()
