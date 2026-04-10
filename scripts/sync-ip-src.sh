#!/usr/bin/env bash
#
# scripts/sync-ip-src.sh
#
# Mirror Verilog/SystemVerilog source files from hardware/rtl/ into
# hardware/ip_repo/src/ so the next Vivado "Package IP" run picks up the
# latest RTL edits.
#
# Why this script exists
# ----------------------
# The Vivado IP packaging flow uses hardware/ip_repo/src/ as its
# canonical packaging source — when you "Package IP", Vivado bundles the
# files from that directory into the IP, NOT from hardware/rtl/. This
# creates a three-level cache hazard:
#
#     hardware/rtl/                         (canonical, where you edit)
#         |
#         v   (this script — manual sync)
#     hardware/ip_repo/src/                 (packaging source)
#         |
#         v   (Vivado "Package IP")
#     hardware/vivado/.../ipshared/<hash>/  (synth-facing copy)
#
# Edits in rtl/ are invisible to synthesis until BOTH lower caches
# refresh. This script handles the first hop. After running it, you
# still need to re-run "Package IP" in Vivado, then re-run synthesis.
# See notes/insights/ or memory project_vivado_ip_cache_hazard.md.
#
# Scope
# -----
# Two source roots are mirrored into hardware/ip_repo/src/:
#
# 1. ALL files at the TOP LEVEL of hardware/rtl/, excluding dotfiles
#    (.gitkeep). This covers:
#      - .v   hand-written Verilog AND Vitis-HLS-generated Verilog
#             (the latter mirrored in by hardware/scripts/sync_hls_to_rtl.sh)
#      - .sv  SystemVerilog
#      - .dat HLS ROM init files (referenced by *_ROM_AUTO_1R.v via
#             $readmemh)
#      - any other top-level files added in the future
#    Subdirectories are NOT recursed.
#
# 2. An EXPLICIT list of files from hardware/weights/hdl/ that the
#    accelerator IP needs at runtime:
#      - layer_config.svh   global header used by inference_top /
#                           inference_hdl / axil_regs
#      - weight_rom.mem     URAM init for the conv weights
#      - qp_packed_rom.mem  BRAM init for the packed quant params
#      - silu_lut.mem       LUT init for the SiLU activation
#    Other files under hardware/weights/hdl/ (bias_rom.mem, m0_rom.mem,
#    nshift_rom.mem, weight_rom_golden.npz, .coe variants, etc.) are
#    NOT used by the current packed-QP path and are intentionally not
#    mirrored. Add to WEIGHTS_FILES below if that ever changes.
#
# Still NOT covered:
#   - testbenches (canonical: hardware/testbench/) — by design, the
#     IP should never bundle its own testbenches.
#
# Limitation
# ----------
# This script only COPIES files. It does NOT update component.xml's
# file groups. After running this script you must re-run the IP
# packaging step (either via the Tcl flow at
#   hardware/scripts/package_accelerator_ip.tcl
# or via Vivado GUI -> Tools -> Create and Package New IP) so the file
# groups get repopulated from the now-current ip_repo/src/. Otherwise
# new files will sit in src/ but not appear in the IP descriptor and
# Vivado will silently ignore them at synth time.
#
# Usage
# -----
#   bash scripts/sync-ip-src.sh
#
# Or, if marked executable:
#   ./scripts/sync-ip-src.sh

set -euo pipefail

# Resolve repo root from script location so the script works regardless
# of the caller's CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RTL_DIR="$REPO_ROOT/hardware/rtl"
WEIGHTS_DIR="$REPO_ROOT/hardware/weights/hdl"
SRC_DIR="$REPO_ROOT/hardware/ip_repo/src"

# Explicit list of accelerator-needed files under hardware/weights/hdl/.
# Edit this if the accelerator's runtime ROM mix changes.
WEIGHTS_FILES=(
    "layer_config.svh"
    "weight_rom.mem"
    "qp_packed_rom.mem"
    "silu_lut.mem"
    "zp_in_rom.mem"
    "zp_out_rom.mem"
)

if [[ ! -d "$RTL_DIR" ]]; then
    echo "ERROR: $RTL_DIR does not exist" >&2
    exit 1
fi
if [[ ! -d "$WEIGHTS_DIR" ]]; then
    echo "ERROR: $WEIGHTS_DIR does not exist" >&2
    exit 1
fi
if [[ ! -d "$SRC_DIR" ]]; then
    echo "ERROR: $SRC_DIR does not exist" >&2
    echo "       Run Vivado IP packaging at least once to create it." >&2
    exit 1
fi

shopt -s nullglob

# ─── Cleanup pass ────────────────────────────────────────────────────
# Remove files from ip_repo/src/ that don't belong: golden .mem,
# testbenches, BD artifacts, constraint files, sim-only data, and
# subdirectories.  These accumulate from earlier manual packaging
# operations and pollute the IP if left in place.

removed=0

cleanup_one() {
    local path="$1"
    echo "  REMOVED: $(basename "$path")"
    rm -rf "$path"
    removed=$((removed + 1))
}

echo "Cleaning ip_repo/src/ ..."

# Remove subdirectories (BD IP instances that don't belong)
for d in "$SRC_DIR"/*/; do
    [[ -d "$d" ]] && cleanup_one "$d"
done

# Remove known pollution patterns
for f in "$SRC_DIR"/golden_*.mem \
         "$SRC_DIR"/tb_*.sv "$SRC_DIR"/tb_*.v \
         "$SRC_DIR"/bd_*.v \
         "$SRC_DIR"/playground*.v \
         "$SRC_DIR"/pixels*.mem \
         "$SRC_DIR"/*.xdc; do
    [[ -f "$f" ]] && cleanup_one "$f"
done

if (( removed > 0 )); then
    echo "  Removed $removed polluting file(s)/dir(s)."
else
    echo "  No pollution found."
fi
echo

# ─── Sync pass ───────────────────────────────────────────────────────
copied=0
skipped=0

# sync_one <src> — copies src into $SRC_DIR if missing or stale.
# Updates the global counters $copied and $skipped.
sync_one() {
    local src="$1"
    [[ -f "$src" ]] || return 0
    local base
    base="$(basename "$src")"
    # skip dotfiles (.gitkeep etc.)
    [[ "$base" == .* ]] && return 0
    local dst="$SRC_DIR/$base"

    if [[ -f "$dst" ]] && cmp -s "$src" "$dst"; then
        skipped=$((skipped + 1))
        return 0
    fi

    cp -p "$src" "$dst"
    echo "  synced: $base"
    copied=$((copied + 1))
}

# 1. Mirror ALL top-level files in hardware/rtl/
for src in "$RTL_DIR"/*; do
    sync_one "$src"
done

# 2. Mirror the explicit weight-tree files
for name in "${WEIGHTS_FILES[@]}"; do
    src="$WEIGHTS_DIR/$name"
    if [[ ! -f "$src" ]]; then
        echo "WARNING: $src not found — re-run hardware/scripts/generate_hdl_weights.py?" >&2
        continue
    fi
    sync_one "$src"
done

echo
echo "Done. $copied file(s) updated, $skipped already in sync."

if (( copied > 0 )); then
    cat <<'EOF'

Next steps:
  1. Re-package the IP so component.xml's file groups pick up the new
     files. This is REQUIRED — sync-ip-src.sh only copies files; it
     does NOT touch component.xml. Without re-packaging, new files
     sit in src/ but Vivado will not see them at synth time.

     Recommended (Tcl, reproducible):
       vivado -mode batch -source hardware/scripts/package_accelerator_ip.tcl

     Or via the GUI:
       Tools → Create and Package New IP → Package your current project
       (use the dedicated packaging project at
        hardware/vivado/accelerator_pkg/, NOT the main tinyissimoyolo.xpr —
        the main project includes the camera pipeline IPs which trigger
        IP_Flow 19-11716 on v_proc_ss.)

  2. Re-run synthesis (or rebuild_bitstream.tcl) to compile the new
     sources.

  3. Verify ipshared/<hash>/src/ matches ip_repo/src/ — if not, the
     IP packaging step did not pick up the new files.
EOF
fi
