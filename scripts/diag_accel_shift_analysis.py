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
