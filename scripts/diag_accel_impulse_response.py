"""Impulse-response silicon diagnostic — localize the +1 shift directly.

Run TWO inferences with layer 0 only (set_max_layers=1):
  Reference: all int8 zero (uint8 128)
  Test:      all int8 zero EXCEPT one 'impulse' pixel at image(IR, IC)

Difference = test - reference.  If conv3d is correct, the impulse at
image(IR, IC) affects conv(IR-1..IR+1, IC-1..IC+1) in the 256x256 conv
output. After 2x max-pool, the affected pool positions are {(row, col)
| row in {(IR-1)//2, IR//2, (IR+1)//2}, col in {(IC-1)//2, IC//2,
(IC+1)//2}} clipped to valid range. For IR=IC=4, that's pool(1..2, 1..2)
→ URAM words 129, 130, 257, 258 in row-major 128-wide.

If silicon shows the affected words at DIFFERENT URAM addresses, we
directly measure the shift in row-major units. The two-run design
cancels out all boundary effects (padding, bias, saturation) that
the single-run baseline couldn't distinguish.

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

# Impulse location — interior, away from boundaries
IMPULSE_ROW = 4
IMPULSE_COL = 4
IMPULSE_VAL = 255  # uint8 255 → int8 127 after driver conversion

print(f"=== bitstream md5: {hashlib.md5(BIT_PATH.read_bytes()).hexdigest()} ===")

def run_inference(image_rgb):
    """Run layer 0 only, return fmap_a[0..16383] as (16384, 16) int8."""
    drv.configure(mode=0)
    drv.set_max_layers(1)
    drv.start()
    drv.write_pixels(image_rgb)
    if not drv.wait_done(timeout_s=3.0):
        raise TimeoutError("Inference timeout")
    return drv.read_window(base_addr=0, buf=0, num_words=16384)

ol = Overlay(str(BIT_PATH), ignore_version=True)
drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)

# Reference: all-zero input (uint8 128 → int8 0)
print("\n=== RUN 1: reference (all int8 zero) ===")
ref_image = np.full((256, 256, 3), 128, dtype=np.uint8)
ref_out = run_inference(ref_image)
print(f"  ref_out[0] = {ref_out[0].tolist()}")

# Test: same but with an impulse at (IR, IC)
print(f"\n=== RUN 2: impulse at image({IMPULSE_ROW}, {IMPULSE_COL}) ===")
test_image = np.full((256, 256, 3), 128, dtype=np.uint8)
test_image[IMPULSE_ROW, IMPULSE_COL] = (IMPULSE_VAL, IMPULSE_VAL, IMPULSE_VAL)
test_out = run_inference(test_image)
print(f"  test_out[0] = {test_out[0].tolist()}")

# Diff = test - ref (in int16 to avoid overflow)
diff = test_out.astype(np.int16) - ref_out.astype(np.int16)
nonzero_mask = (diff != 0).any(axis=1)
nonzero_positions = np.where(nonzero_mask)[0]
print(f"\n=== DIFFERENCE ANALYSIS ===")
print(f"  Total URAM positions with nonzero diff: {len(nonzero_positions)}")

# Expected impulse response locations
# For IR=IC=4 in image, conv affects image(3..5, 3..5) in 256x256,
# pool_2x2 maps to pool((3..5)//2, (3..5)//2) = pool(1..2, 1..2)
# URAM addresses in 128-wide row-major: 129, 130, 257, 258
expected_pool_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
expected_uram_addrs = sorted([r * 128 + c for r, c in expected_pool_positions])
print(f"  Expected affected URAM addresses (NO shift): {expected_uram_addrs}")

# Under +1 row-major shift hypothesis: silicon[k] = correct[k+1]
# So correct pool(1, 1) = URAM addr 129 lands at silicon URAM addr 128
shifted_uram_addrs = [(a - 1) % 16384 for a in expected_uram_addrs]
print(f"  Expected affected URAM addresses (+1 shift):  {sorted(shifted_uram_addrs)}")

print(f"\n  First 20 nonzero-diff URAM addresses:")
for addr in nonzero_positions[:20]:
    r, c = addr // 128, addr % 128
    d = diff[addr]
    max_abs = int(np.abs(d).max())
    n_nz = int((d != 0).sum())
    is_exp = addr in expected_uram_addrs
    is_sft = addr in shifted_uram_addrs
    tag = " <== EXPECTED" if is_exp else (" <== +1 SHIFT PREDICTION" if is_sft else "")
    print(f"    URAM[{addr:>5}] pool({r:>3},{c:>3})  n_chs_nz={n_nz:>2} max|Δ|={max_abs:>3}{tag}")

# Check each hypothesis explicitly
no_shift_hits = sum(1 for a in expected_uram_addrs if nonzero_mask[a])
plus1_shift_hits = sum(1 for a in shifted_uram_addrs if nonzero_mask[a])

# Also check: how many of the nonzero positions fall within ±1 of the expected
# set? (To detect partial shifts or alternative patterns.)
print(f"\n  Verdict:")
print(f"    NO shift: {no_shift_hits}/4 expected positions hit")
print(f"    +1 shift: {plus1_shift_hits}/4 expected positions hit")

if no_shift_hits == 4 and plus1_shift_hits < 4:
    print(f"  VERDICT: silicon conv3d writes CORRECTLY (no +1 shift)")
    print(f"           The +1 shift bug must be elsewhere — likely in the read path")
elif plus1_shift_hits == 4 and no_shift_hits < 4:
    print(f"  VERDICT: silicon conv3d writes +1 SHIFTED")
    print(f"           The bug IS in the conv3d write path (or something earlier)")
elif no_shift_hits == 4 and plus1_shift_hits == 4:
    print(f"  AMBIGUOUS: both positions hit, possibly leakage")
else:
    print(f"  UNEXPECTED: neither pattern fully matches")

# Print a 2D ascii heatmap of the affected region (7x7 around impulse's
# pool coordinates (2, 2))
print(f"\n  Pool-space heatmap (7x7 around expected impulse center (2,2)):")
print(f"       " + " ".join(f"c{c:<4}" for c in range(7)))
for r in range(7):
    row_cells = []
    for c in range(7):
        addr = r * 128 + c
        if nonzero_mask[addr]:
            row_cells.append(f"{'X':>4} ")
        else:
            row_cells.append(f"{'.':>4} ")
    print(f"  r{r:<2} " + " ".join(row_cells))

ol.free()
