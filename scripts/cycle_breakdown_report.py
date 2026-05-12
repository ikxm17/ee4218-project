"""Render a per-layer cycle breakdown from cycle_monitor's CSV.

Prints a Markdown table to stdout (totals, primary buckets as %-of-layer,
overlap-pipeline activity as %-of-layer-total) and writes a stacked
horizontal bar chart to cycle_breakdown.png. Optionally overlays an
analytical estimate (H * W * ceil(cin/16) * cout) and a silicon-side
totals column from diag_cycle_breakdown_silicon.py.

Run:
    conda run -n claude-utils python scripts/cycle_breakdown_report.py \\
        path/to/cycle_breakdown.csv [--silicon-csv silicon_cycles.csv] \\
        [--png cycle_breakdown.png]
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
import pathlib
import shutil
import sys


LAYER_TYPE_NAME = {0: "CONV3", 1: "CONV3+POOL", 2: "CONV1+SiLU", 3: "CONV1_LIN"}

PRIMARY_COLS = ["weight_load", "compute", "next_chout", "next_layer"]
PARALLEL_COLS = ["act_active", "pool_active", "rmw_s0_active", "rmw_wr_active"]


def load_sim(csv_path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "layer_idx":  int(r["layer_idx"]),
                "layer_type": int(r["layer_type"]),
                "h_in":       int(r["h_in"]),
                "cin":        int(r["cin"]),
                "cout":       int(r["cout"]),
                "total":      int(r["total"]),
                **{c: int(r[c]) for c in PRIMARY_COLS + PARALLEL_COLS},
            })
    rows.sort(key=lambda r: r["layer_idx"])
    return rows


def load_silicon(csv_path: pathlib.Path) -> dict[int, int]:
    out: dict[int, int] = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            out[int(r["layer_idx"])] = int(r["total"])
    return out


def analytical(row: dict, c_par: int = 16) -> int:
    """Naive lower bound: H_out * W_out * ceil(Cin/C_par) * Cout cycles.

    H_out depends on whether maxpool is fused (CONV3_POOL halves H/W).
    Treats the convolver as a 1-cycle-per-output-pixel-per-cin-batch core.
    """
    h_in = row["h_in"]
    h_out = h_in // 2 if row["layer_type"] == 1 else h_in  # CONV3_POOL = 1
    return h_out * h_out * math.ceil(row["cin"] / c_par) * row["cout"]


def render_markdown(rows: list[dict], silicon: dict[int, int] | None) -> str:
    total_all = sum(r["total"] for r in rows)
    headers = ["L", "type", "HxW", "Cin", "Cout", "cycles", "%total",
               "load%", "comp%", "ovh%", "act%", "pool%", "rmw%", "model"]
    if silicon:
        headers.append("silicon")
        headers.append("Δ")

    def pct(n: int, d: int) -> str:
        return f"{(100.0 * n / d):.1f}" if d else "-"

    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        tot = r["total"]
        ovh = r["next_chout"] + r["next_layer"]
        cells = [
            str(r["layer_idx"]),
            LAYER_TYPE_NAME.get(r["layer_type"], "?"),
            f"{r['h_in']}",
            str(r["cin"]),
            str(r["cout"]),
            f"{tot:,}",
            pct(tot, total_all),
            pct(r["weight_load"], tot),
            pct(r["compute"], tot),
            pct(ovh, tot),
            pct(r["act_active"], tot),
            pct(r["pool_active"], tot),
            pct(r["rmw_wr_active"], tot),
            f"{analytical(r):,}",
        ]
        if silicon:
            sili = silicon.get(r["layer_idx"])
            cells.append(f"{sili:,}" if sili is not None else "-")
            cells.append(f"{(sili - tot):+d}" if sili is not None else "-")
        out.append("| " + " | ".join(cells) + " |")

    out.append("")
    out.append(f"**Total:** {total_all:,} cycles "
               f"({total_all / 1e6:.2f} M @ 100 MHz = {total_all / 1e5:.2f} ms)")
    if silicon:
        sil_tot = sum(silicon.values())
        out.append(f"**Silicon total:** {sil_tot:,} cycles "
                   f"(Δ vs sim = {sil_tot - total_all:+d})")
    return "\n".join(out)


def render_png(rows: list[dict], png_path: pathlib.Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[report] matplotlib not installed, skipping PNG: {png_path}",
              file=sys.stderr)
        return

    layers = [r["layer_idx"] for r in rows]
    load = [r["weight_load"] for r in rows]
    comp = [r["compute"] for r in rows]
    ch = [r["next_chout"] for r in rows]
    ly = [r["next_layer"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    left = [0] * len(rows)
    for vals, label, color in [
        (load, "S_LOAD",       "#4c72b0"),
        (comp, "S_COMPUTE",    "#55a868"),
        (ch,   "S_NEXT_CHOUT", "#c44e52"),
        (ly,   "S_NEXT_LAYER", "#8172b2"),
    ]:
        ax.barh(layers, vals, left=left, label=label, color=color)
        left = [a + b for a, b in zip(left, vals)]

    # Overlay analytical estimate as a black tick per row
    for r in rows:
        est = analytical(r)
        ax.plot([est, est], [r["layer_idx"] - 0.4, r["layer_idx"] + 0.4],
                color="black", linewidth=1.5)

    ax.set_yticks(layers)
    ax.invert_yaxis()
    ax.set_xlabel("Cycles")
    ax.set_ylabel("Layer index")
    ax.set_title("TinyissimoYOLO HDL inference — per-layer cycle breakdown\n"
                 "(black tick = H·W·⌈Cin/16⌉·Cout analytical estimate)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f"[report] wrote {png_path}")


def snapshot_to_history(csv_path: pathlib.Path, round_name: str,
                        rows: list[dict]) -> None:
    """Copy csv_path into notes/cycle-breakdowns/<round_name>.csv and print
    a ready-to-paste SUMMARY.md table row."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    dest_dir = repo_root / "notes" / "cycle-breakdowns"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{round_name}.csv"
    shutil.copy2(csv_path, dest)
    print(f"\n[snapshot] copied {csv_path} → {dest}", file=sys.stderr)

    total = sum(r["total"] for r in rows)
    fps = 100e6 / total  # cycles/sec / cycles/frame
    hot = max(rows, key=lambda r: r["total"])
    hot_name = LAYER_TYPE_NAME.get(hot["layer_type"], "?")
    hot_pct = 100.0 * hot["total"] / total
    today = _dt.date.today().isoformat()

    row = (f"| {round_name} | {today} | {total:,} | ~{fps:.0f} | — | — | "
           f"L{hot['layer_idx']} {hot_name} {hot_pct:.1f}% | <fill> | <hash> |")
    print("[snapshot] SUMMARY.md row (edit Δ and commit hash, then paste):",
          file=sys.stderr)
    print(row, file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=pathlib.Path,
                    help="Path to cycle_breakdown.csv")
    ap.add_argument("--silicon-csv", type=pathlib.Path,
                    help="Optional silicon CSV from diag_cycle_breakdown_silicon.py")
    ap.add_argument("--png", type=pathlib.Path,
                    default=pathlib.Path("cycle_breakdown.png"),
                    help="PNG output path (default: cycle_breakdown.png)")
    ap.add_argument("--snapshot", metavar="ROUND_NAME",
                    help="Also archive CSV into notes/cycle-breakdowns/<name>.csv "
                         "and print a SUMMARY.md table row")
    args = ap.parse_args()

    rows = load_sim(args.csv)
    silicon = load_silicon(args.silicon_csv) if args.silicon_csv else None

    print(render_markdown(rows, silicon))
    render_png(rows, args.png)
    if args.snapshot:
        snapshot_to_history(args.csv, args.snapshot, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
