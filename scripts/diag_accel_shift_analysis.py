"""Diagnostic: quantify the +1 URAM-word shift in layer 0's silicon output.

The smoking-gun script for the Apr 7-8 2026 sim-vs-silicon discrepancy.
After observing that silicon[N] looked suspiciously like golden[N+1] for
the first few words of layer 0's surviving fragment, this script tests
the shift hypothesis across the entire 12,288-word fragment:

  H1: silicon[k] == golden[k]      (no shift)
  H2: silicon[k] == golden[k+1]    (forward shift by 1)
  H3: silicon[k] == golden[k-1]    (backward shift by 1)

On the broken bitstream md5 5198d50145..., this script returned:
    H1:   117/12288 words (  0.95%)    88990/196608 bytes (45.26%)
    H2: 12287/12287 words (100.00%)   196592/196592 bytes (100.00%)
    H3:    96/12287 words (  0.78%)    79100/196592 bytes (40.24%)

The H2 100% match across the full 12,287-word window is the proof that
the silicon's layer 0 output is uniformly shifted by exactly +1 URAM
word relative to sim. This is invisible to all Vivado simulation models
(behavioural, post-synth, post-impl functional, post-impl timing) and
only manifests on real silicon.

Re-run this after any RTL fix attempt: a successful fix should flip the
result to H1 100% and H2 to noise.

Usage:
    ssh kria-01 'cd ~/workspace/ee4218-project && \\
        echo asdfzxcv | sudo -S XILINX_XRT=/usr PYTHONPATH=. \\
        /opt/ee4218/ee4218-venv/bin/python3 scripts/diag_accel_shift_analysis.py'
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

# Read the FULL surviving fragment of layer 0 (silicon fmap_a[4096..16383]).
# Layer 0 wrote fmap_a[0..16383]; only [4096..16383] is preserved through
# the rest of inference because every later fmap_a writer (layers 2,4,6,8,10,12,15)
# stays in addresses < 1024 except for layers 12 and 15 which both stay
# under 640. So [4096..] is frozen layer 0 output.
SIL_START = 4096
N_WORDS   = 12288
print(f"\n=== reading silicon fmap_a[{SIL_START}..{SIL_START+N_WORDS-1}] ===")
silicon = drv.read_window(SIL_START, 0, N_WORDS)
print(f"  silicon shape: {silicon.shape}")

golden = load_golden_uram_mem(
    str(GOLDEN_DIR / "golden_layer0_uram.mem"),
    num_words=SIL_START + N_WORDS,
)
print(f"  golden shape (full): {golden.shape}")

# Edge-word experiment: distinguish "write addresses are -1"
# from "first data item is skipped". Read the word just BELOW
# the layer 0 region (silicon[SIL_START - 1] == silicon[4095])
# and the word at the END of the layer 0 region
# (silicon[SIL_START + N_WORDS - 1] == silicon[16383]).
#
#   Interpretation A (write addresses are -1):
#     silicon[4095]  == golden[4096]     (displaced d_0 landed here)
#     silicon[16383] == stale / init     (never written)
#
#   Interpretation B (first data skipped):
#     silicon[4095]  == stale / init     (nothing written below)
#     silicon[16383] == stale / init     (last cycle never emitted)
#
# Both interpretations agree on silicon[16383]; they diverge on
# silicon[4095]. One read of each, compared against interp-A and
# interp-B predictions, is the cheapest discriminator.
print("\n=== edge-word experiment (A vs B) ===")
edge_below  = drv.read_window(SIL_START - 1, 0, 2)   # fmap_a[4095..4096]
edge_above  = drv.read_window(SIL_START + N_WORDS - 1, 0, 1)  # fmap_a[16383]
print(f"  silicon[4095]  = {edge_below[0].tolist()}")
print(f"  silicon[4096]  = {edge_below[1].tolist()}  (sanity, should match H2 for k=4096)")
print(f"  silicon[16383] = {edge_above[0].tolist()}")
print(f"  golden[4096]   = {golden[4096].tolist()}  (interp-A predicts silicon[4095] == this)")
print(f"  golden[4095]   = {golden[4095].tolist()}  (stale/init baseline, if any)")
print(f"  golden[16383]  = {golden[16383].tolist()}  (H1 expected at silicon[16383] if writes aligned)")
if np.array_equal(edge_below[0], golden[4096]):
    print("  VERDICT: interp-A matches — silicon writes at -1 offset (d_0 landed at base-1)")
elif np.array_equal(edge_below[0], np.zeros(16, dtype=np.int8)):
    print("  VERDICT: silicon[4095] is all-zero — stale/init baseline (interp-B more likely)")
else:
    print("  VERDICT: silicon[4095] is neither interp-A nor zero; novel value — needs inspection")

# Wraparound experiment: if silicon writes at addr-1 uniformly with
# 14-bit FMAP_ADDR_W wraparound, then d_0 (which should land at 0)
# would wrap to 16383. Test: is silicon[16383] == golden[0]?
print("\n=== wraparound experiment ===")
print(f"  silicon[16383] = {edge_above[0].tolist()}")
print(f"  golden[0]      = {golden[0].tolist()}  (wraparound predicts silicon[16383] == this)")
if np.array_equal(edge_above[0], golden[0]):
    print("  VERDICT: wraparound matches — silicon uniformly writes at addr-1 with 14-bit wrap")
else:
    print("  VERDICT: silicon[16383] != golden[0]; not a simple wraparound")

# Per-layer shift hypothesis sweep: run H1 and H2 on every layer's
# surviving URAM fragment. If the +1 shift is layer-0-specific, only
# layer 0 will show H2 near 100% while others fall at H1.
#
# Layer fragments from diag_accel_layer_bisect.py:
LAYER_FRAGMENTS = [
    # (layer_idx, buf_name, start, length, golden_file)
    (0,  "fmap_a", 4096, 12288, "golden_layer0_uram.mem"),
    (2,  "fmap_a", 2048,  2048, "golden_layer2_uram.mem"),
    (4,  "fmap_a", 1024,  1024, "golden_layer4_uram.mem"),
    (6,  "fmap_a",  640,   384, "golden_layer6_uram.mem"),
    (8,  "fmap_a",  128,   128, "golden_layer8_uram.mem"),
    (10, "fmap_a",    0,   128, "golden_layer10_uram.mem"),
    (12, "fmap_a",  256,   256, "golden_layer12_uram.mem"),
    (15, "fmap_a",  512,   128, "golden_layer15_uram.mem"),
]

print("\n=== per-layer H1/H2 shift sweep ===")
print(f"  layer | buf    | region              | H1 words   | H2 words   | verdict")
print(f"  ----- | ------ | ------------------- | ---------- | ---------- | -------")
for lyr, buf_name, start, length, gfile in LAYER_FRAGMENTS:
    try:
        buf = 0 if buf_name == "fmap_a" else 1
        sil = drv.read_window(start, buf, length)
        gld = load_golden_uram_mem(str(GOLDEN_DIR / gfile), num_words=start + length)
        gld_aligned = gld[start:start + length]
        gld_shifted = gld[start + 1:start + length] if start + length <= gld.shape[0] else gld_aligned[:-1]
        h1 = int((sil == gld_aligned).all(axis=1).sum())
        if sil.shape[0] > 1 and gld_shifted.shape[0] > 0:
            h2 = int((sil[:-1] == gld_shifted).all(axis=1).sum())
            h2_denom = sil.shape[0] - 1
        else:
            h2, h2_denom = 0, 1
        verdict = "+1 shift" if h2 > 0.9 * h2_denom else ("no shift" if h1 > 0.9 * length else "other")
        print(f"  {lyr:>5} | {buf_name:<6} | [{start:>5}..{start+length-1:>5}] "
              f"| {h1:>5}/{length:<4} | {h2:>5}/{h2_denom:<4} | {verdict}")
    except Exception as e:
        print(f"  {lyr:>5} | {buf_name:<6} | [{start:>5}..{start+length-1:>5}] | ERROR: {e}")

golden_aligned     = golden[SIL_START:SIL_START + N_WORDS]
silicon_for_shift  = silicon[:N_WORDS - 1]
golden_shifted_fwd  = golden[SIL_START + 1:SIL_START + N_WORDS]
golden_shifted_back = golden[SIL_START - 1:SIL_START - 2 + N_WORDS]


def compare(a, b):
    if a.shape != b.shape:
        return None, None, None
    word_match = (a == b).all(axis=1)
    byte_match = (a == b).sum()
    return int(word_match.sum()), int(byte_match), a.size


def show(label, a, b):
    word, byte, total = compare(a, b)
    if word is None:
        print(f"  {label}: SHAPE MISMATCH ({a.shape} vs {b.shape})")
        return
    n = a.shape[0]
    pct_word = 100.0 * word / n
    pct_byte = 100.0 * byte / total
    print(f"  {label}: {word:>5}/{n} words ({pct_word:5.2f}%)   "
          f"{byte:>6}/{total} bytes ({pct_byte:5.2f}%)")


print("\n=== shift hypothesis comparisons ===")
print(f"  test                            | full-word matches      | byte matches")
print(f"  ------------------------------- | ---------------------- | ----------------------")
show("H1 silicon[k] == golden[k]      ", silicon, golden_aligned)
show("H2 silicon[k] == golden[k+1]    ", silicon_for_shift, golden_shifted_fwd)
show("H3 silicon[k] == golden[k-1]    ", silicon_for_shift, golden_shifted_back)

# H2 longest consecutive run
print("\n=== H2 (silicon[k] == golden[k+1]) consecutive-match analysis ===")
h2_word_match = (silicon_for_shift == golden_shifted_fwd).all(axis=1)
print(f"  total H2 matches: {int(h2_word_match.sum())}/{N_WORDS - 1}")
runs = []
in_run = False
run_start = 0
for i, m in enumerate(h2_word_match):
    if m and not in_run:
        run_start = i
        in_run = True
    elif not m and in_run:
        runs.append((run_start, i - 1, i - run_start))
        in_run = False
if in_run:
    runs.append((run_start, len(h2_word_match) - 1, len(h2_word_match) - run_start))
runs.sort(key=lambda r: -r[2])
print(f"  longest 5 H2 match runs:")
for start, end, length in runs[:5]:
    print(f"    silicon[{SIL_START + start}..{SIL_START + end}] = "
          f"golden[{SIL_START + start + 1}..{SIL_START + end + 1}]   "
          f"({length} consecutive words)")

# Per-word lane match histograms
print("\n=== per-word lane-match histogram (H1: direct compare, no shift) ===")
h1_lane_matches = (silicon == golden_aligned).sum(axis=1)
hist = np.bincount(h1_lane_matches, minlength=17)
for n_lanes in range(17):
    if hist[n_lanes] > 0:
        bar = "#" * min(60, int(hist[n_lanes] * 60 / N_WORDS))
        print(f"  {n_lanes:>2} lanes match: {hist[n_lanes]:>5} words  {bar}")

print("\n=== per-word lane-match histogram (H2: shift by +1) ===")
h2_lane_matches = (silicon_for_shift == golden_shifted_fwd).sum(axis=1)
hist = np.bincount(h2_lane_matches, minlength=17)
for n_lanes in range(17):
    if hist[n_lanes] > 0:
        bar = "#" * min(60, int(hist[n_lanes] * 60 / N_WORDS))
        print(f"  {n_lanes:>2} lanes match: {hist[n_lanes]:>5} words  {bar}")

ol.free()
