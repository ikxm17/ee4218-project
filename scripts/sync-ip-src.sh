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
# Only .v and .sv files in hardware/rtl/ (top level). Does NOT touch:
#   - .mem / .coe init files (sourced from hardware/weights/hdl/)
#   - layer_config.svh (sourced from hardware/weights/hdl/)
#   - testbenches (sourced from hardware/testbench/ during packaging)
#   - golden output .mem (testbench refs)
#
# These have their own canonical sources outside rtl/ — extending this
# script to cover them would require additional source mappings. Keep
# it minimal for now.
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
SRC_DIR="$REPO_ROOT/hardware/ip_repo/src"

if [[ ! -d "$RTL_DIR" ]]; then
    echo "ERROR: $RTL_DIR does not exist" >&2
    exit 1
fi
if [[ ! -d "$SRC_DIR" ]]; then
    echo "ERROR: $SRC_DIR does not exist" >&2
    echo "       Run Vivado IP packaging at least once to create it." >&2
    exit 1
fi

shopt -s nullglob

copied=0
skipped=0
for src in "$RTL_DIR"/*.v "$RTL_DIR"/*.sv; do
    [[ -f "$src" ]] || continue
    base="$(basename "$src")"
    dst="$SRC_DIR/$base"

    if [[ -f "$dst" ]] && cmp -s "$src" "$dst"; then
        skipped=$((skipped + 1))
        continue
    fi

    cp -p "$src" "$dst"
    echo "  synced: $base"
    copied=$((copied + 1))
done

echo
echo "Done. $copied file(s) updated, $skipped already in sync."

if (( copied > 0 )); then
    cat <<'EOF'

Next steps:
  1. In Vivado: re-run "Package IP" to refresh the IP catalog.
     - Vivado may also need to re-detect the file changes; if it
       does not, close and re-open the IP packager project.
  2. Re-run synthesis to compile the new sources.
  3. Verify ipshared/<hash>/src/ matches rtl/ — if not, the IP
     packaging step did not pick up the new files.
EOF
fi
