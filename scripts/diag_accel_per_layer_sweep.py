"""Per-layer silicon vs golden sweep — characterize whether the +1 shift
is uniform across the entire pipeline, layer-0 only, or has some other
pattern (cumulative, layer-dependent, growing, etc.).

Loops max_layers from 1..17. For each value N, soft-resets the accel,
sets max_layers=N, runs inference, then reads the URAM region for
layer N-1's output and compares against `golden_layer{N-1}_uram.mem`.

The URAM layout for each layer comes from the authoritative
`tb_inference_hdl.sv` URAM_WORDS / PP_BUF_SEL / WR_OFFSET tables (cited
in the per-layer URAM map I cross-checked):

  Layer | fmap | base | words | notes
    0   |  a   |  0   | 16384 | 128x128x16
    1   |  b   |  0   | 16384 | 128x128x16
    2   |  a   |  0   |  4096 | 64x64x16
    3   |  b   |  0   |  8192 | 64x64x32 = 2 groups
    4   |  a   |  0   |  2048 | 32x32x32
    5   |  b   |  0   |  4096 | 32x32x64
    6   |  a   |  0   |  1024 | 16x16x64
    7   |  b   |  0   |  1024 | 16x16x64
    8   |  a   |  0   |   512 | 8x8x128
    9   |  b   |  0   |   512 | 8x8x128
   10   |  a   |  0   |   128 | 8x8x24
   11   |  b   |  256 |   256 | cv2 head
   12   |  a   |  256 |   256 | cv2
   13   |  b   |  256 |   256 | cv2 final
   14   |  b   |  512 |   128 | cv3 head
   15   |  a   |  512 |   128 | cv3
   16   |  b   |  512 |    64 | cv3 final, only lanes 0..2 valid

This is the diagnostic primitive that answers: "is the +1 shift
layer-0-only, pipeline-wide-uniform, or something else?"

For each layer we report:
  - num matches at shift 0 (no shift)
  - num matches at shift +1 (silicon = golden[k+1])
  - num matches at shift -1 (silicon = golden[k-1])
  - first 4 mismatched indices for visual sanity

Run via:
    ssh kria-01 'cd ~/workspace/ee4218-project && \\
        echo asdfzxcv | sudo -S XILINX_XRT=/usr PYTHONPATH=. \\
        /opt/ee4218/ee4218-venv/bin/python3 scripts/diag_accel_per_layer_sweep.py'

The script saves all 17 silicon snapshots to `per_layer_silicon.npz` on
the board, with keys `layer{N:02d}` for N in 0..16. Use np.load to
post-process if you need fancier analysis.
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

# Authoritative per-layer URAM map (from tb_inference_hdl.sv:139-173 +
# inference_top.sv:442-450 routing rule "buf_sel=0 -> fmap_a is OUTPUT").
LAYER_MAP = [
    # (idx, fmap_buf, base, words, valid_lanes_per_first_group)
    ( 0, 0,   0, 16384, 16),  # 128x128x16
    ( 1, 1,   0, 16384, 16),  # 128x128x16
    ( 2, 0,   0,  4096, 16),  # 64x64x16
    ( 3, 1,   0,  8192, 16),  # 64x64x32 (2 groups)
    ( 4, 0,   0,  2048, 16),  # 32x32x32 (2 groups)
    ( 5, 1,   0,  4096, 16),  # 32x32x64 (4 groups)
    ( 6, 0,   0,  1024, 16),  # 16x16x64 (4 groups)
    ( 7, 1,   0,  1024, 16),  # 16x16x64 (4 groups)
    ( 8, 0,   0,   512, 16),  # 8x8x128  (8 groups)
    ( 9, 1,   0,   512, 16),  # 8x8x128  (8 groups)
    (10, 0,   0,   128, 16),  # 8x8x24   (1 full group + 1 partial; lanes 0..7 in 2nd group)
    (11, 1, 256,   256, 16),  # cv2 8x8x64
    (12, 0, 256,   256, 16),  # cv2 8x8x64
    (13, 1, 256,   256, 16),  # cv2 8x8x64 final
    (14, 1, 512,   128, 16),  # cv3 8x8x24
    (15, 0, 512,   128, 16),  # cv3 8x8x24
    (16, 1, 512,    64,  3),  # cv3 8x8x3 final  (only lanes 0..2 valid)
]


def shift_match(silicon: np.ndarray, golden: np.ndarray, shift: int) -> tuple[int, int]:
    """Count matching ROWS under a circular shift.

    silicon[k] vs golden[(k + shift) mod N]. Returns (matches, total).
    Uses np.roll for the circular semantics — matches the bug
    signature `silicon[k] == golden[(k+1) mod N]` for shift=+1.
    """
    n = silicon.shape[0]
    rolled = np.roll(golden, -shift, axis=0)
    matches = (silicon == rolled).all(axis=1).sum()
    return int(matches), n


def first_diffs(silicon: np.ndarray, golden: np.ndarray, shift: int, k: int = 4) -> list[int]:
    """Return the first k indices where silicon != golden under given shift."""
    rolled = np.roll(golden, -shift, axis=0)
    bad_rows = np.where(~(silicon == rolled).all(axis=1))[0]
    return bad_rows[:k].tolist()


def main():
    print(f"=== bitstream md5: {hashlib.md5(BIT_PATH.read_bytes()).hexdigest()} ===")

    # Load image (matches diag_accel_layer0_max1.py exactly)
    with open(MEM_PATH) as f:
        mem_bytes = np.array([int(l.strip(), 16) for l in f if l.strip()], dtype=np.uint8)
    mem_int8 = mem_bytes.view(np.int8).reshape(3, 256, 256)
    image = (mem_int8.astype(np.int16) + 128).astype(np.uint8).transpose(1, 2, 0)

    ol = Overlay(str(BIT_PATH), ignore_version=True)
    drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)

    silicon_snapshots: dict[str, np.ndarray] = {}

    print(f"\n{'Layer':>5} | {'fmap':>4} | {'base':>4} | {'words':>5} | "
          f"{'shift0':>7} | {'shift+1':>7} | {'shift-1':>7} | "
          f"{'cyc':>9} | verdict")
    print("-" * 100)

    for (lyr, buf, base, words, vlanes) in LAYER_MAP:
        # Fresh state for every run
        drv.configure(mode=0)
        drv.set_max_layers(lyr + 1)
        drv.start()
        drv.write_pixels(image)

        if not drv.wait_done(timeout_s=3.0):
            print(f"  layer {lyr}: TIMEOUT — does the bitstream support max_layers? ABORTING")
            ol.free()
            return

        sil = drv.read_window(base, buf, words)
        silicon_snapshots[f"layer{lyr:02d}"] = sil

        # Mask off invalid lanes for layers where C_valid < 16
        # (only layer 16 currently — lanes 0..2 valid). Compare only valid lanes.
        if vlanes < 16:
            sil_cmp = sil[:, :vlanes]
        else:
            sil_cmp = sil

        gold = load_golden_uram_mem(
            str(GOLDEN_DIR / f"golden_layer{lyr}_uram.mem"),
            num_words=words,
        )
        if vlanes < 16:
            gold_cmp = gold[:, :vlanes]
        else:
            gold_cmp = gold

        m0, _ = shift_match(sil_cmp, gold_cmp, 0)
        mp, _ = shift_match(sil_cmp, gold_cmp, +1)
        mn, _ = shift_match(sil_cmp, gold_cmp, -1)

        # Verdict tag
        pct0 = m0 / words
        pctp = mp / words
        pctn = mn / words
        if pct0 > 0.999:
            verdict = "BIT-EXACT (no shift)"
        elif pctp > 0.999:
            verdict = "+1 SHIFT (silicon=gold[k+1])"
        elif pctn > 0.999:
            verdict = "-1 SHIFT (silicon=gold[k-1])"
        elif pctp > 0.5:
            verdict = f"PARTIAL +1 ({pctp*100:.0f}%)"
        elif pct0 > 0.5:
            verdict = f"PARTIAL match ({pct0*100:.0f}%)"
        else:
            verdict = f"BROKEN (m0={pct0*100:.0f}% mp={pctp*100:.0f}%)"

        print(f"  {lyr:>3} | {'a' if buf == 0 else 'b':>4} | {base:>4} | {words:>5} | "
              f"{m0:>7} | {mp:>7} | {mn:>7} | {drv.cycle_count:>9} | {verdict}")

        # If neither matches, dump first few mismatch indices for inspection
        if pct0 < 0.999 and pctp < 0.999:
            diffs0 = first_diffs(sil_cmp, gold_cmp, 0, k=4)
            diffsp = first_diffs(sil_cmp, gold_cmp, +1, k=4)
            print(f"        first diffs @ shift0: {diffs0}")
            print(f"        first diffs @ shift+1: {diffsp}")

    # Save all snapshots for offline post-processing
    np.savez("per_layer_silicon.npz", **silicon_snapshots)
    print(f"\n=== saved {len(silicon_snapshots)} snapshots to per_layer_silicon.npz ===")
    ol.free()


if __name__ == "__main__":
    main()
