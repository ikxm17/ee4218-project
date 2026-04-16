"""Per-layer cycle count from live silicon, sweeping max_layers_run = 1..17.

For each N in 1..17, configure the accelerator, set max_layers_run = N, run
a single inference using the same pixels_layer0.mem input the testbench
uses, and read cycle_count after the FSM goes idle. Per-layer cycles are
the deltas between successive N. Emits CSV with columns layer_idx,total
matching the sim's cycle_breakdown.csv schema, so cycle_breakdown_report.py
can overlay simulated and silicon totals.

Run on the board:
    ssh kria-01 'cd ~/workspace/ee4218-project && \\
        echo asdfzxcv | sudo -S XILINX_XRT=/usr PYTHONPATH=. \\
        /opt/ee4218/ee4218-venv/bin/python3 \\
        scripts/diag_cycle_breakdown_silicon.py --output silicon_cycles.csv'
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pynq import Overlay  # noqa: E402

from software.overlay.drivers.tinyissimoyolo_accelerator import (  # noqa: E402
    TinyissimoYoloAcceleratorDriver,
)


DEFAULT_PIXELS_MEM = REPO_ROOT / "hardware" / "testbench" / "inference_hdl" / "pixels_layer0.mem"
DEFAULT_BITSTREAM = REPO_ROOT / "software" / "overlay" / "playground.bit"


def image_from_pixels_mem(mem_path: pathlib.Path) -> np.ndarray:
    """Reconstruct the (256, 256, 3) uint8 RGB image the testbench loads."""
    with mem_path.open() as f:
        mem_bytes = np.array(
            [int(line.strip(), 16) for line in f if line.strip()],
            dtype=np.uint8,
        )
    if mem_bytes.size != 3 * 256 * 256:
        raise RuntimeError(
            f"{mem_path}: expected {3 * 256 * 256} hex words, got {mem_bytes.size}"
        )
    mem_int8 = mem_bytes.view(np.int8).reshape(3, 256, 256)
    image_hwc = (mem_int8.astype(np.int16) + 128).astype(np.uint8).transpose(1, 2, 0)
    return image_hwc


def measure_cycles_for_n_layers(drv, image: np.ndarray, n: int) -> int:
    """Run a single inference stopped after layer n-1, return cycle_count."""
    drv.configure(mode=0, engine=0)
    drv.set_max_layers(n)
    drv.start()
    drv.write_pixels(image)
    if not drv.wait_done(timeout_s=2.0):
        raise TimeoutError(f"Inference for max_layers={n} did not complete in 2 s")
    return drv.cycle_count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bitstream", default=str(DEFAULT_BITSTREAM),
                    help="Path to .bit overlay (defaults to playground.bit)")
    ap.add_argument("--pixels-mem", default=str(DEFAULT_PIXELS_MEM),
                    help="Path to pixels_layer0.mem (the testbench input)")
    ap.add_argument("--output", default="silicon_cycles.csv",
                    help="Output CSV path")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Number of repeats per N to confirm determinism")
    args = ap.parse_args()

    image = image_from_pixels_mem(pathlib.Path(args.pixels_mem))
    print(f"[silicon] loaded image from {args.pixels_mem}")

    overlay = Overlay(args.bitstream)
    print(f"[silicon] loaded overlay {args.bitstream}")

    drv = TinyissimoYoloAcceleratorDriver(overlay.tinyissimoyolo_accel_0)

    cumulative: list[int] = []
    for n in range(1, 18):
        runs = [measure_cycles_for_n_layers(drv, image, n) for _ in range(args.repeats)]
        if len(set(runs)) > 1:
            print(f"[silicon] WARN: max_layers={n} not deterministic across "
                  f"{args.repeats} runs: {runs}")
        cycles = runs[0]
        cumulative.append(cycles)
        print(f"[silicon] max_layers={n:2d}: cycle_count = {cycles:>10d}")

    out_path = pathlib.Path(args.output)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer_idx", "total"])
        prev = 0
        for i, total in enumerate(cumulative):
            per_layer = total - prev
            w.writerow([i, per_layer])
            prev = total
    print(f"[silicon] CSV written to {out_path.resolve()}")
    print(f"[silicon] full-network cycle_count = {cumulative[-1]} "
          f"(expected ~2,936,665)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
