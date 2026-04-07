"""Diagnostic: read every surviving URAM fragment for every layer (except 11)
and compare against the corresponding golden_layerN_uram.mem file.

Uses the programmable result window (RESULT_BASE_REG / RESULT_BUF_REG
added in feat(rtl): add programmable result window for layer-by-layer
URAM bisect) to slide the AXI-Lite result region over arbitrary fmap_a
and fmap_b addresses. Walks every layer's surviving fragment after a
full inference and reports per-layer pass/fail + a head-of-table dump
of the earliest divergent layer.

Surviving fragment map (computed by tracing the 17-layer ping-pong):
    layer  buffer    silicon range          golden file slice
    -----  --------  ---------------------  -------------------
      0    fmap_a    [4096..16383]          [4096..16383]
      1    fmap_b    [8192..16383]          [8192..16383]
      2    fmap_a    [2048..4095]           [2048..4095]
      3    fmap_b    [4096..8191]           [4096..8191]
      4    fmap_a    [1024..2047]           [1024..2047]
      5    fmap_b    [1024..4095]           [1024..4095]
      6    fmap_a    [ 640..1023]           [ 640..1023]
      7    fmap_b    [ 640..1023]           [ 640..1023]
      8    fmap_a    [ 128.. 255]           [ 128.. 255]
      9    fmap_b    [   0.. 255]           [   0.. 255]
     10    fmap_a    [   0.. 127]           [   0.. 127]
     11      —       overwritten by 13      n/a
     12    fmap_a    [ 256.. 511]           [   0.. 255]   (WR_OFFSET=256)
     13    fmap_b    [ 256.. 511]           [   0.. 255]   (WR_OFFSET=256)
     14    fmap_b    [ 576.. 639]           [  64.. 127]   (WR_OFFSET=512)
     15    fmap_a    [ 512.. 639]           [   0.. 127]   (WR_OFFSET=512)
     16    fmap_b    [ 512.. 575]           [   0..  63]   (WR_OFFSET=512)

Usage:
    ssh kria-01 'cd ~/workspace/ee4218-project && \\
        echo asdfzxcv | sudo -S XILINX_XRT=/usr PYTHONPATH=. \\
        /opt/ee4218/ee4218-venv/bin/python3 scripts/diag_accel_layer_bisect.py'

History: developed during the Apr 7-8 2026 sim-vs-silicon debug session
to bisect the layer at which silicon first diverges from sim. Result:
all 16 checkable layers diverge — layer 0 is the earliest, and the
divergence is a clean +1 URAM word address shift (see
diag_accel_shift_analysis.py for the smoking-gun quantitative test).
Originally /tmp/diag_accel_bisect2.py.
"""
import hashlib
import pathlib
import numpy as np
from pynq import Overlay
from software.overlay.drivers.tinyissimoyolo_accelerator import TinyissimoYoloAcceleratorDriver
from software.overlay.tests.checks import load_golden_uram_mem

BIT_PATH    = pathlib.Path("hardware/output/playground.bit")
MEM_PATH    = pathlib.Path("hardware/testbench/inference_hdl/pixels_layer0.mem")
GOLDEN_DIR  = pathlib.Path("hardware/testbench/inference_hdl")

# (layer, buf_name, buf_idx, silicon_start, num_words, golden_start)
CHECKPOINTS = [
    (0,  "fmap_a", 0, 4096, 12288, 4096),
    (1,  "fmap_b", 1, 8192,  8192, 8192),
    (2,  "fmap_a", 0, 2048,  2048, 2048),
    (3,  "fmap_b", 1, 4096,  4096, 4096),
    (4,  "fmap_a", 0, 1024,  1024, 1024),
    (5,  "fmap_b", 1, 1024,  3072, 1024),
    (6,  "fmap_a", 0,  640,   384,  640),
    (7,  "fmap_b", 1,  640,   384,  640),
    (8,  "fmap_a", 0,  128,   128,  128),
    (9,  "fmap_b", 1,    0,   256,    0),
    (10, "fmap_a", 0,    0,   128,    0),
    # layer 11 fully overwritten by layer 13 — skipped
    (12, "fmap_a", 0,  256,   256,    0),
    (13, "fmap_b", 1,  256,   256,    0),
    (14, "fmap_b", 1,  576,    64,   64),
    (15, "fmap_a", 0,  512,   128,    0),
    (16, "fmap_b", 1,  512,    64,    0),
]


def load_golden_slice(path: pathlib.Path, golden_start: int, num_words: int) -> np.ndarray:
    """Read golden file [golden_start..golden_start+num_words] as (num_words,16) int8."""
    full = load_golden_uram_mem(str(path), num_words=golden_start + num_words)
    return full[golden_start:]


print(f"=== bitstream md5: {hashlib.md5(BIT_PATH.read_bytes()).hexdigest()} ===")

with open(MEM_PATH) as f:
    mem_bytes = np.array([int(l.strip(), 16) for l in f if l.strip()], dtype=np.uint8)
mem_int8 = mem_bytes.view(np.int8).reshape(3, 256, 256)
image = (mem_int8.astype(np.int16) + 128).astype(np.uint8).transpose(1, 2, 0)

ol = Overlay(str(BIT_PATH), ignore_version=True)
drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)

print("\n=== running inference ===")
result = drv.run(image)
print(f"  cycle_count: {result['cycle_count']}")

print("\n=== full layer bisection (16 of 17 layers) ===")
print(f"  {'lyr':>3} | {'region':<22} | {'match':<5} | {'diffs':>13} | first 3 mismatches")
print(f"  {'-'*3} | {'-'*22} | {'-'*5} | {'-'*13} | {'-'*30}")

earliest_fail = None
results = []
for layer, buf_name, buf_idx, sil_start, n, gold_start in CHECKPOINTS:
    actual = drv.read_window(sil_start, buf_idx, n)
    golden_path = GOLDEN_DIR / f"golden_layer{layer}_uram.mem"
    golden = load_golden_slice(golden_path, gold_start, n)
    match = np.array_equal(actual, golden)
    diffs = int((actual != golden).sum())
    region = f"{buf_name}[{sil_start}..{sil_start+n-1}]"

    results.append((layer, match, diffs, n, actual, golden))
    if not match and earliest_fail is None:
        earliest_fail = layer

    if match:
        first_diff_str = ""
    else:
        diff_idx = np.where(actual != golden)
        sample = list(zip(*diff_idx))[:3]
        first_diff_str = str(sample)

    status = "YES" if match else "NO"
    print(f"  {layer:>3} | {region:<22} | {status:<5} | "
          f"{diffs:>6}/{n*16:<6} | {first_diff_str}")

print()
if earliest_fail is None:
    print("ALL CHECKABLE LAYERS MATCH GOLDEN — bug appears to be gone.")
else:
    last_pass = None
    for layer, match, _, _, _, _ in results:
        if match and (earliest_fail is None or layer < earliest_fail):
            last_pass = layer
        if not match:
            break
    if last_pass is None:
        print(f"EARLIEST DIVERGENT LAYER: {earliest_fail}")
        print(f"  → bug appears at layer {earliest_fail} or earlier")
    else:
        print(f"EARLIEST DIVERGENT LAYER: {earliest_fail}")
        print(f"LAST PASSING LAYER:        {last_pass}")
        print(f"  → bug lives between layer {last_pass} and {earliest_fail}")

# Dump the head of the earliest failing layer for inspection
if earliest_fail is not None:
    print(f"\n=== layer {earliest_fail} head: actual vs golden (first 4 words, all 16 lanes) ===")
    for layer, match, diffs, n, actual, golden in results:
        if layer != earliest_fail:
            continue
        for w in range(min(4, n)):
            a = actual[w].tolist()
            g = golden[w].tolist()
            same = [i for i in range(16) if a[i] == g[i]]
            diff = [i for i in range(16) if a[i] != g[i]]
            print(f"  word {w}:")
            print(f"    actual  : {a}")
            print(f"    golden  : {g}")
            print(f"    same lanes: {same}")
            print(f"    diff lanes: {diff}")
        break

ol.free()
