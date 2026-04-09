"""Per-layer silicon vs golden sweep — characterize per-layer silicon
correctness for either the HDL or the HLS inference engine.

Two engine modes, picked by --engine:

  --engine hdl  (default; original behaviour, screenshot-reproducible)
    Loops max_layers from 1..17. For each value N, soft-resets the
    accel, sets max_layers=N, runs inference, then reads the URAM
    region for layer N-1's output and compares against
    `golden_layer{N-1}_uram.mem`.  This works because max_layers is
    wired into inference_hdl.sv's FSM and bounds the per-layer loop.

  --engine hls
    Single-shot capture: max_layers is IGNORED by the HLS engine
    (the HLS C++ at hardware/hls/tinyissimo_layer_top.cpp walks all
    17 layers internally inside one ap_start, with no external
    bound — see hardware/rtl/inference_hls.sv:185-257).  So we run
    HLS exactly once on a freshly-loaded bitstream (URAMs come up
    at zero on bitstream reload — see project_uram_no_init memory
    + the sdp_ram.sv comment in HANDOFF_hls_bugfix.md), then read
    all 17 windows from the post-run state.

    Starting from clean (zero) URAMs is what makes HLS forensics
    unambiguous: any layer-N window that comes back ALL-ZERO means
    the HLS engine never wrote there — there's no HDL leftover to
    masquerade as a passing layer (the "mirage" pattern called out
    in HANDOFF_hls_bugfix.md "HLS silicon failure mode").  Run from
    a fresh bitstream load — do NOT pre-run HDL.

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

For HDL this is the diagnostic primitive that answers: "is the +1
shift layer-0-only, pipeline-wide-uniform, or something else?"

For HLS, the per-layer verdict is computed against the layer's
PRIVATE RANGE — the subset of its output window that no LATER layer
in the pipeline overwrites with the same fmap buffer.  After a
single full-chain run only the private range is solely attributable
to this layer's writes; the rest of the window has been trampled
by downstream layers.  A layer that bit-exact matches its golden
in the private range is provably writing correctly to the addresses
it owns, even if the full-window match count looks bad.

Verdict tags:
  - BIT-EXACT (full window) : layer's full window matches golden
                              (only happens when private == window)
  - PRIVATE BIT-EXACT       : layer's private range matches golden
                              (the rest is downstream layers' writes)
  - PRIVATE BROKEN          : private range does not match golden —
                              the engine wrote this layer's solely-
                              owned addresses incorrectly.  Real bug.
  - NO PRIVATE RANGE        : entire window overwritten by later
                              layers; cannot test in isolation
                              (only L11 in this pipeline)
  - ALL-ZERO                : engine never wrote here, URAMs at
                              boot zero (requires the pre-check
                              to also confirm clean URAMs)
  - +1 / -1 SHIFT           : full-window circular shift match,
                              kept from the original HDL bisection

For each layer we also report:
  - num matches at shift 0/+1/-1 against the FULL window (legacy)
  - private match count vs private size
  - first 4 mismatched indices (only when PRIVATE BROKEN)

The private-range model is robust to whether HDL was run before
HLS or not, because HLS overwrites every URAM address HDL would
have written.  But the pre-check at the top of main() still asserts
URAMs are at boot zero — that's the unambiguous "no leftover" gate
the user asked for after the first HLS run.

Run via:
    ssh ubuntu@100.95.72.121 'cd ~/workspace/ee4218-project && \\
        echo <pw> | sudo -S XILINX_XRT=/usr PYTHONPATH=. \\
        /opt/ee4218/ee4218-venv/bin/python3 \\
        scripts/diag_accel_per_layer_sweep.py [--engine hdl|hls]'

The script saves all 17 silicon snapshots to
`per_layer_silicon_<engine>.npz` on the board, with keys
`layer{N:02d}` for N in 0..16. Use np.load to post-process if you
need fancier analysis.
"""
import argparse
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


def compute_private_mask(layer_idx: int) -> np.ndarray:
    """Return a length-`words` boolean mask: True at offsets in this
    layer's window that NO later layer in LAYER_MAP overwrites with
    the same fmap buffer.

    The "private range" is the only subset of a layer's output window
    we can directly verify after a single full-chain run, because
    later layers freely clobber the earlier sections of fmap_a/fmap_b.
    A layer whose private range bit-exact matches its golden is
    provably writing correctly to the addresses it solely owns —
    even if its window's other addresses now hold downstream layers'
    outputs.  This is what makes single-shot HLS capture sufficient
    for a per-layer verdict.

    Worked example for the failure mode that broke the previous
    "L0 LEFTOVER" tag: L2's window is fmap_a[0..4096), but L4
    later overwrites fmap_a[0..2048).  L2's private mask is therefore
    True only on offsets [2048..4096), and the bit-exact test runs
    over those 2048 words against the second half of golden_layer2.
    """
    _, my_buf, my_base, my_words, _ = LAYER_MAP[layer_idx]
    mask = np.ones(my_words, dtype=bool)
    for (_, l_buf, l_base, l_words, _) in LAYER_MAP[layer_idx + 1:]:
        if l_buf != my_buf:
            continue
        rel_lo = max(0, l_base - my_base)
        rel_hi = min(my_words, (l_base + l_words) - my_base)
        if rel_lo < rel_hi:
            mask[rel_lo:rel_hi] = False
    return mask


def collect_hdl(drv: TinyissimoYoloAcceleratorDriver,
                image: np.ndarray) -> tuple[dict[str, np.ndarray], dict[int, int]]:
    """HDL collection: per-iter max_layers sweep (original behaviour).

    For each layer N, soft-reset, set max_layers=N+1, run, then read
    layer N's URAM window. Returns (snapshots, per_layer_cycles).
    """
    snapshots: dict[str, np.ndarray] = {}
    cycles: dict[int, int] = {}
    for (lyr, buf, base, words, _vlanes) in LAYER_MAP:
        # Fresh state for every run
        drv.configure(mode=0, engine=0)
        drv.set_max_layers(lyr + 1)
        drv.start()
        drv.write_pixels(image)

        if not drv.wait_done(timeout_s=3.0):
            raise TimeoutError(
                f"layer {lyr}: HDL run did not complete within 3s — "
                "does the bitstream support max_layers?"
            )

        snapshots[f"layer{lyr:02d}"] = drv.read_window(base, buf, words)
        cycles[lyr] = drv.cycle_count
    return snapshots, cycles


def collect_hls(drv: TinyissimoYoloAcceleratorDriver,
                image: np.ndarray) -> tuple[dict[str, np.ndarray], int]:
    """HLS collection: single-shot capture.

    The HLS engine ignores max_layers (it walks all 17 layers
    internally inside one ap_start). So we run it ONCE on a freshly
    loaded bitstream — URAMs come up at zero — then read all 17
    windows from the post-run state. Returns (snapshots, cycles).

    Caller MUST have just done Overlay(BIT_PATH) so URAM cells are
    in their bitstream-load (zero) state. Do NOT pre-run HDL here:
    the whole point of starting from clean URAMs is that any "this
    layer is all zero" verdict is unambiguous evidence the engine
    didn't write — the HDL leftover mirage from the session-3 npz
    forensics doesn't apply.
    """
    drv.configure(mode=0, engine=1)
    drv.start()
    drv.write_pixels(image)

    # 5s ceiling: cosim is ~1.68M cycles ≈ 17 ms @ 100 MHz, so 5 s
    # is generous even with AXI-Lite readback Python overhead.
    if not drv.wait_done(timeout_s=5.0):
        raise TimeoutError("HLS run did not complete within 5s")

    cycles = drv.cycle_count
    snapshots: dict[str, np.ndarray] = {}
    for (lyr, buf, base, words, _vlanes) in LAYER_MAP:
        snapshots[f"layer{lyr:02d}"] = drv.read_window(base, buf, words)
    return snapshots, cycles


def report(snapshots: dict[str, np.ndarray],
           per_layer_cycles: dict[int, int] | None,
           hls_total_cycles: int | None) -> None:
    """Per-layer verdict table using private-range analysis.

    The "private range" of a layer is the subset of its output window
    that no LATER layer overwrites.  After a single full-chain run,
    only the private range is solely attributable to this layer's
    writes; the rest of the window has been trampled by downstream
    layers.  Bit-exact match in the private range is the unambiguous
    "this layer wrote correctly" test — robust to whether HDL was
    pre-run or not, since HLS overwrites every URAM address HDL
    would have written.

    The "shift0/+1/-1" columns are kept for HDL backwards-compat:
    in HDL mode every layer is run in isolation via max_layers, so
    the full window has no later-layer pollution and shift0 == words
    for every healthy layer (matches the original screenshot).
    """
    if hls_total_cycles is not None:
        print(f"\nHLS run: {hls_total_cycles} cycles total\n")

    print(f"{'L':>3} | {'fm':>3} | {'base':>4} | {'words':>5} | "
          f"{'shift0':>7} | {'shift+1':>7} | {'shift-1':>7} | "
          f"{'priv match':>13} | {'cyc':>9} | verdict")
    print("-" * 115)

    for lyr_idx, (lyr, buf, base, words, vlanes) in enumerate(LAYER_MAP):
        sil = snapshots[f"layer{lyr:02d}"]

        # Mask off invalid lanes for layers where C_valid < 16
        # (only layer 16 currently — lanes 0..2 valid).
        sil_cmp = sil[:, :vlanes] if vlanes < 16 else sil

        gold = load_golden_uram_mem(
            str(GOLDEN_DIR / f"golden_layer{lyr}_uram.mem"),
            num_words=words,
        )
        gold_cmp = gold[:, :vlanes] if vlanes < 16 else gold

        m0, _ = shift_match(sil_cmp, gold_cmp, 0)
        mp, _ = shift_match(sil_cmp, gold_cmp, +1)
        mn, _ = shift_match(sil_cmp, gold_cmp, -1)
        pct0 = m0 / words
        pctp = mp / words
        pctn = mn / words

        # Private-range analysis: subset of words that no later layer
        # overwrites with the same fmap buffer.  Bit-exact match in
        # the private range is the unambiguous "this layer wrote
        # correctly" test, regardless of clean-vs-dirty URAM state.
        priv_mask = compute_private_mask(lyr_idx)
        priv_size = int(priv_mask.sum())
        if priv_size > 0:
            priv_rows_match = (sil_cmp[priv_mask] == gold_cmp[priv_mask]).all(axis=1)
            priv_match = int(priv_rows_match.sum())
        else:
            priv_rows_match = np.zeros(0, dtype=bool)
            priv_match = 0

        is_all_zero = not sil.any()

        if priv_size == 0:
            # Only L11 hits this — its entire window is overwritten
            # by L13. Cannot test in isolation; trust the L13 verdict
            # as proxy for cv2-head correctness.
            verdict = "NO PRIVATE RANGE (overwritten by later layers)"
        elif priv_match == priv_size:
            if priv_size == words:
                verdict = "BIT-EXACT (full window)"
            else:
                verdict = f"PRIVATE BIT-EXACT ({priv_size}/{words} solely owned)"
        elif pctp > 0.999:
            verdict = "+1 SHIFT (silicon=gold[k+1])"
        elif pctn > 0.999:
            verdict = "-1 SHIFT (silicon=gold[k-1])"
        elif is_all_zero:
            verdict = "ALL-ZERO (engine never wrote here)"
        else:
            verdict = (f"PRIVATE BROKEN "
                       f"({priv_match}/{priv_size} priv words match)")

        priv_str = f"{priv_match}/{priv_size}"
        cyc_str = (f"{per_layer_cycles[lyr]:>9}"
                   if per_layer_cycles is not None else f"{'—':>9}")
        print(f"  {lyr:>3} | {'a' if buf == 0 else 'b':>3} | {base:>4} | {words:>5} | "
              f"{m0:>7} | {mp:>7} | {mn:>7} | "
              f"{priv_str:>13} | {cyc_str} | {verdict}")

        # Detail dump only for the actually-broken case: print the
        # first few private-range mismatch indices in window-relative
        # coordinates, since those are the ones that point at a real
        # bug in this layer's writes.
        if priv_size > 0 and priv_match < priv_size:
            priv_idx = np.where(priv_mask)[0]
            priv_bad_local = np.where(~priv_rows_match)[0][:4]
            priv_bad_global = priv_idx[priv_bad_local].tolist()
            print(f"        first priv mismatches @ window indices: "
                  f"{priv_bad_global}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--engine", choices=["hdl", "hls"], default="hdl",
        help="which inference engine to sweep (default: hdl). HLS uses "
             "single-shot capture since the HLS top ignores max_layers."
    )
    parser.add_argument(
        "--out", default=None,
        help="output npz path (default: per_layer_silicon_<engine>.npz)"
    )
    args = parser.parse_args()
    out_path = args.out or f"per_layer_silicon_{args.engine}.npz"

    print(f"=== bitstream md5: {hashlib.md5(BIT_PATH.read_bytes()).hexdigest()} ===")
    print(f"=== engine: {args.engine.upper()} ===")

    # Load image (matches diag_accel_layer0_max1.py exactly)
    with open(MEM_PATH) as f:
        mem_bytes = np.array([int(l.strip(), 16) for l in f if l.strip()], dtype=np.uint8)
    mem_int8 = mem_bytes.view(np.int8).reshape(3, 256, 256)
    image = (mem_int8.astype(np.int16) + 128).astype(np.uint8).transpose(1, 2, 0)

    # Hard bitstream reload — URAM cells come up at zero, which is
    # the precondition the HLS forensic verdicts depend on. Cheap
    # enough that the HDL path also gets a clean baseline for free.
    ol = Overlay(str(BIT_PATH), ignore_version=True)
    drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)

    try:
        # Pre-check: confirm URAMs are at boot-zero before any
        # inference runs.  PYNQ's Overlay() calls download() which
        # triggers a hard PROG_B (FPGA reprogram), and Xilinx URAM
        # cells reset to zero on PROG_B by spec.  If anything non-
        # zero shows up here, either Overlay() did not actually
        # reload the bitstream (some PYNQ/FPGA-manager configs cache)
        # or there is residual state from another agent — power-
        # cycle the board and retry.
        #
        # The private-range verdict in report() is technically robust
        # to dirty URAMs (HLS overwrites every address HDL would
        # have written, so the m0/private analysis is invariant),
        # but the user explicitly asked for direct positive evidence
        # that no leftover is present, and this is the cheapest way
        # to give it.
        fmap_a_pre = drv.read_window(0, 0, 16384)
        fmap_b_pre = drv.read_window(0, 1, 16384)
        nz_a = int((fmap_a_pre != 0).sum())
        nz_b = int((fmap_b_pre != 0).sum())
        total_bytes = 16384 * 16
        print(f"=== pre-run URAM check: "
              f"fmap_a non-zero bytes = {nz_a}/{total_bytes}, "
              f"fmap_b non-zero bytes = {nz_b}/{total_bytes} ===")
        if nz_a or nz_b:
            print("WARNING: URAMs are NOT at boot zero. The "
                  "private-range verdict is still valid (HLS "
                  "overwrites everything HDL would write), but "
                  "the ALL-ZERO tag becomes meaningless. Power-"
                  "cycle the board and retry for a clean baseline.")
        else:
            print("       URAMs confirmed clean — no HDL leftover, "
                  "no residual state from prior runs.")

        if args.engine == "hdl":
            snapshots, per_layer_cycles = collect_hdl(drv, image)
            report(snapshots, per_layer_cycles, hls_total_cycles=None)
        else:
            snapshots, total_cycles = collect_hls(drv, image)
            report(snapshots, per_layer_cycles=None, hls_total_cycles=total_cycles)

        np.savez(out_path, **snapshots)
        print(f"\n=== saved {len(snapshots)} snapshots to {out_path} ===")
    finally:
        ol.free()


if __name__ == "__main__":
    main()
