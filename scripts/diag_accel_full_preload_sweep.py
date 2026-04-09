"""Diagnostic: write the FULL real layer-0 image via the production preload
path, then immediately read back fmap_b[0..16383] and compare against the
expected packed-pixel layout byte-for-byte.

Hypothesis under test: the +1 silicon shift on layer 0 is not caused by
the URAM cascade (Phase A0 ruled it out by direct attribute readout) and
not caused by an RTL bug (behav, post-synth, post-impl-func sims all PASS
the layer 0 verification when the production-mode AXI testbench was
extended to check it). The remaining viable hypothesis is that the
production-path PRELOAD writes a slightly different fmap_b on silicon
than on sim — for example, dropping or duplicating one pixel write at
some specific address — and the existing preload check only writes 100
pixels which doesn't exercise the failing region.

Method:
  1. Configure + start the accelerator (enters PH_PRELOAD).
  2. Write all 65,536 pixels of pixels_layer0.mem in the same order
     and packing as software/overlay/drivers/tinyissimoyolo_accelerator.py
     write_pixels(), but using the raw `_ip.write(PIXEL_FIFO, word)`
     loop directly so we can inject debug logging if needed.
  3. After the 65,536th write, the FSM auto-transitions to PH_RUN and
     starts inference. We must NOT block waiting for done — instead we
     either (a) issue soft_reset immediately to halt the FSM and read
     fmap_b in the resulting idle state, or (b) accept that fmap_b
     gets clobbered by layer 1+ writes and read it BEFORE the loop
     completes the final write. Approach (a) is cleaner.
  4. With the FSM halted, read fmap_b[0..16383] in 320-word chunks via
     the result-window register (set_result_window + read_results_raw),
     and compare against the expected packed layout.

Expected packed layout (per axil_regs.sv:207):
  fmap_b[w] = {pixel[4w+3], pixel[4w+2], pixel[4w+1], pixel[4w+0]}
  pixel[i] = (0 << 24) | (B[i] << 16) | (G[i] << 8) | R[i]

If silicon's fmap_b is bit-exact to expected: preload is correct, and
the +1 shift bug is somewhere downstream that we haven't yet localized.
If silicon's fmap_b shows a +1 word shift: the preload accumulator
itself is the source of the bug.
If the divergence is at a specific address (e.g. word 4096 = the URAM
cascade boundary, or word N where N is the first time some carry bit
flips): the bug is address-specific.

Usage:
    ssh kria-01 'cd ~/workspace/ee4218-project && \\
        echo asdfzxcv | sudo -S XILINX_XRT=/usr PYTHONPATH=. \\
        /opt/ee4218/ee4218-venv/bin/python3 scripts/diag_accel_full_preload_sweep.py'
"""
import hashlib
import pathlib
import numpy as np
from pynq import Overlay
from software.overlay.drivers.tinyissimoyolo_accelerator import TinyissimoYoloAcceleratorDriver

BIT_PATH = pathlib.Path("hardware/output/playground.bit")
MEM_PATH = pathlib.Path("hardware/testbench/inference_hdl/pixels_layer0.mem")

PIXEL_FIFO_OFFSET = 0x020

print(f"=== bitstream md5: {hashlib.md5(BIT_PATH.read_bytes()).hexdigest()} ===")
print(f"=== full preload sweep ===")

# Load the same image data the diag uses (and that pixels_layer0.mem
# is the source for in sim).
with open(MEM_PATH) as f:
    mem_bytes = np.array([int(l.strip(), 16) for l in f if l.strip()], dtype=np.uint8)
mem_int8 = mem_bytes.view(np.int8).reshape(3, 256, 256)
# CHW int8 → HWC uint8 (reverses zero-point shift) — matches diag_accel_shift_analysis.py
image = (mem_int8.astype(np.int16) + 128).astype(np.uint8).transpose(1, 2, 0)
print(f"  image: shape={image.shape} dtype={image.dtype}")

# Build the expected fmap_b layout (16384 × 16 bytes).
# Each 128-bit URAM word holds 4 pixels, each pixel is 32-bit
# little-endian {0, B, G, R}.
pixels_int8 = (image.astype(np.int16) - 128).astype(np.int8).reshape(-1, 3)
# packed[i] = [R, G, B, 0] for pixel i in row-major order
packed = np.zeros((256 * 256, 4), dtype=np.uint8)
packed[:, 0] = pixels_int8[:, 0].view(np.uint8)
packed[:, 1] = pixels_int8[:, 1].view(np.uint8)
packed[:, 2] = pixels_int8[:, 2].view(np.uint8)
expected_words = packed.reshape(16384, 16)  # (16384, 16) uint8
print(f"  expected layout: {expected_words.shape}")

ol = Overlay(str(BIT_PATH), ignore_version=True)
drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)

# Step 1: clean state, FIFO mode
print("\n=== configure + start ===")
drv.configure(mode=0)
drv.start()

# Step 2: write all 65536 pixels via raw MMIO loop (mirrors write_pixels)
print(f"\n=== writing all 65536 pixels via FIFO ===")
# Build the 32-bit word list the same way write_pixels does, so this
# diagnostic exercises exactly the production preload sequence.
words = packed.view(np.uint32).flatten().tolist()
assert len(words) == 65536

# Write 65535 of them, then halt before the last one to keep the FSM
# in PH_PRELOAD. (preload_done_r asserts when pixel_count == 65536.)
for w in words[:-1]:
    drv._ip.write(PIXEL_FIFO_OFFSET, w)
print(f"  wrote 65535 pixels; pixel_count={drv.pixel_count}")
print(f"  status: {drv.read_status()}")

# Step 3: read fmap_b[0..16383] via the result window. Since FSM is
# still in PH_PRELOAD (last pixel not yet written), nothing else is
# touching fmap_b and we can scan the full address range safely.
print(f"\n=== reading fmap_b[0..16383] via result window ===")
sil_words = drv.read_window(0, buf=1, num_words=16384).view(np.uint8)
print(f"  read shape: {sil_words.shape}")

# Step 4: compare. We only filled fmap_b[0..16383] partially because
# the last pixel (n=65535) hasn't been written yet, which means
# fmap_b[16383] only has 3 of its 4 lanes populated. Compare the
# 16383 fully-written words.
print(f"\n=== full-range bit-exact comparison ===")
match_count = 0
mismatch_addrs = []
for w in range(16383):
    if np.array_equal(sil_words[w], expected_words[w]):
        match_count += 1
    else:
        mismatch_addrs.append(w)

print(f"  matches: {match_count}/16383")
print(f"  mismatches: {len(mismatch_addrs)}")

if mismatch_addrs:
    print(f"\n  first 20 mismatch addresses:")
    for a in mismatch_addrs[:20]:
        print(f"    fmap_b[{a:>5}]: silicon={sil_words[a].tolist()} expected={expected_words[a].tolist()}")

    # Test +1 shift hypothesis
    shift_matches = 0
    for w in range(16382):
        if np.array_equal(sil_words[w], expected_words[w + 1]):
            shift_matches += 1
    print(f"\n  +1 shift test: silicon[k] == expected[k+1] for {shift_matches}/16382 words")
    if shift_matches > 16000:
        print(f"  VERDICT: PRELOAD IS +1 SHIFTED — every silicon[k] holds expected[k+1]")
        print(f"           This means the bug is in the preload path, NOT in compute.")
    else:
        # Show histogram of mismatch addresses
        print(f"\n  mismatch address histogram (bins of 1024):")
        hist = np.zeros(16, dtype=np.int32)
        for a in mismatch_addrs:
            hist[a // 1024] += 1
        for i, c in enumerate(hist):
            bar = "#" * min(60, c // 16)
            print(f"    [{i*1024:>5}..{(i+1)*1024-1:>5}]: {c:>5}  {bar}")
else:
    print(f"  VERDICT: PRELOAD IS BIT-EXACT — fmap_b matches expected at every word.")
    print(f"           The +1 shift bug is NOT in the preload path.")

ol.free()
