"""Diagnostic: verify silicon's preload path writes pixels to fmap_b correctly.

The +1 URAM word shift in layer 0's silicon output has been narrowed to
either a write/compute bug (silicon's ram[] differs from sim's ram[])
or a primitive-level sim/silicon discrepancy. To isolate further, this
script verifies that the AXI-Lite preload path correctly writes pixel
data to fmap_b BEFORE inference runs. If preload is correct, the bug
is in conv3d compute. If preload is wrong, the bug is in axil_regs
preload accumulator or fmap_b port A.

Method:
  1. Soft-reset the accelerator
  2. Set MODE=0 (FIFO preload)
  3. Pulse CTRL[0]=START, CTRL[1]=FIFO_RST  →  enters PH_PRELOAD
  4. Write a SMALL number of distinguishable pixels via PIXEL_FIFO
     (fewer than 65536, so PH_PRELOAD doesn't transition to PH_RUN)
  5. Read fmap_b[0..N] via the AXI-Lite result window
  6. Compare against the expected packed-pixel layout
     (4 pixels per 128-bit URAM word, low byte = R, then G, B, pad=0)

Expected layout per inference_top axil_regs.sv:207:
  o_preload_wr_data <= {s_axi_wdata, pixel_accum[95:64], ...}
  → bits [127:96] = pixel 3
    bits [95:64]  = pixel 2
    bits [63:32]  = pixel 1
    bits [31:0]   = pixel 0
  Each 32-bit pixel slot is little-endian {pad=0, B, G, R}.

If silicon[fmap_b word k] matches the expected layout, the preload
path is correct. If silicon[fmap_b word k] is shifted by 1 word (i.e.,
silicon[fmap_b k] == expected[fmap_b k+1]), the preload itself has
the bug. Anything else means the corruption pattern is different.
"""
import pathlib
import numpy as np
from pynq import Overlay
from software.overlay.drivers.tinyissimoyolo_accelerator import TinyissimoYoloAcceleratorDriver

BIT_PATH = pathlib.Path("hardware/output/playground.bit")

# Number of pixels to write. 100 pixels = 25 fmap_b words.
# Stay well below the 65536 PH_PRELOAD→PH_RUN trigger so the FSM
# stays in PH_PRELOAD and we can read fmap_b without inference
# overwriting it.
N_PIXELS    = 100
N_WORDS_OUT = 30  # read a few extra to see the boundary
PIXEL_FIFO_OFFSET = 0x020

ol = Overlay(str(BIT_PATH), ignore_version=True)
drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)

print("=== preload-path verification ===")
print(f"  bitstream: {BIT_PATH}")

# Step 1-2: clean state, FIFO mode
drv.configure(mode=0)

# Step 3: enter PH_PRELOAD
drv.start()

# Step 4: write a known small pattern via the FIFO. We can't use
# write_pixels() because it requires (256,256,3) and writes 65536
# pixels. Use raw MMIO writes instead.
print(f"\n=== writing {N_PIXELS} known pixels via FIFO ===")
expected_words = np.zeros((N_WORDS_OUT, 16), dtype=np.uint8)
for i in range(N_PIXELS):
    R = (i * 3 + 1) & 0xFF
    G = (i * 3 + 2) & 0xFF
    B = (i * 3 + 3) & 0xFF
    word32 = (0 << 24) | (B << 16) | (G << 8) | R
    drv._ip.write(PIXEL_FIFO_OFFSET, word32)
    word_idx = i // 4
    lane     = i % 4
    if word_idx < N_WORDS_OUT:
        expected_words[word_idx, lane * 4 + 0] = R
        expected_words[word_idx, lane * 4 + 1] = G
        expected_words[word_idx, lane * 4 + 2] = B
        expected_words[word_idx, lane * 4 + 3] = 0

# Verify the FIFO swallowed all writes
pixel_cnt = drv.pixel_count
print(f"  pixel_count after writes: {pixel_cnt} (expected {N_PIXELS})")

status = drv.read_status()
print(f"  status: busy={status['busy']} idle={status['idle']} done={status['done']} preload_done={status['preload_done']}")

# Step 5: read fmap_b[0..N_WORDS_OUT] via the result window. Note that
# read_window does its own set_result_window calls inside, so it's
# safe to call directly even mid-PH_PRELOAD (the result region is
# served by axil_regs regardless of phase).
print(f"\n=== reading fmap_b[0..{N_WORDS_OUT-1}] via result window ===")
sil_words = drv.read_window(0, buf=1, num_words=N_WORDS_OUT).view(np.uint8)

# Step 6: compare
print(f"\n=== compare ===")
print(f"  word | silicon[16 bytes]                              | expected[16 bytes]                             | status")
print(f"  ---- | ---------------------------------------------- | ---------------------------------------------- | ------")
mismatches = 0
shift_matches = 0
for w in range(N_WORDS_OUT):
    sil = sil_words[w]
    exp = expected_words[w]
    if np.array_equal(sil, exp):
        status_str = "MATCH"
    else:
        mismatches += 1
        # Test if shifted by 1 word
        if w + 1 < N_WORDS_OUT and np.array_equal(sil, expected_words[w + 1]):
            status_str = "+1 SHIFT"
            shift_matches += 1
        else:
            status_str = "DIFF"
    sil_str = " ".join(f"{b:3d}" for b in sil[:8]) + " .."
    exp_str = " ".join(f"{b:3d}" for b in exp[:8]) + " .."
    print(f"  {w:>4} | {sil_str:<46} | {exp_str:<46} | {status_str}")

print(f"\n=== summary ===")
print(f"  total mismatches: {mismatches}/{N_WORDS_OUT}")
print(f"  +1 shift count:   {shift_matches}/{N_WORDS_OUT}")
if mismatches == 0:
    print(f"  VERDICT: preload path is CORRECT — bug is downstream of fmap_b preload")
elif shift_matches >= N_WORDS_OUT - 2:
    print(f"  VERDICT: preload path itself is +1 SHIFTED — bug is in axil_regs preload accumulator")
else:
    print(f"  VERDICT: preload has mixed corruption — needs deeper inspection")

ol.free()
