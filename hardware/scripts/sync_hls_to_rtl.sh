#!/usr/bin/env bash
#
# sync_hls_to_rtl.sh
#
# Copies the Vitis-HLS-generated .v and .dat files from
#   hardware/hls/tinyissimo_layer/tinyissimo_layer/hls/impl/ip/hdl/verilog/
# into hardware/rtl/ so the IP packager can find every accelerator
# source — HDL pipeline + HLS engine + ROM init data — in one
# directory. This is the "loose Verilog" path: the HLS output is
# treated as plain RTL alongside the hand-written Verilog/SystemVerilog,
# so the IP packaging GUI can be pointed at hardware/rtl/ as a single
# source root.
#
# All HLS files are prefixed `tinyissimo_layer_top_*` so there are no
# filename collisions with the existing hardware/rtl/ files (verified
# 2026-04-08).
#
# Run from the repo root:
#   bash hardware/scripts/sync_hls_to_rtl.sh
#
# Idempotent: cp -f overwrites existing files. Safe to re-run after
# any HLS regeneration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HLS_DIR="$REPO_ROOT/hardware/hls/tinyissimo_layer/tinyissimo_layer/hls/impl/ip/hdl/verilog"
RTL_DIR="$REPO_ROOT/hardware/rtl"

if [[ ! -d "$HLS_DIR" ]]; then
    echo "ERROR: $HLS_DIR does not exist." >&2
    echo "       Re-run Vitis HLS export to populate it." >&2
    exit 1
fi
if [[ ! -d "$RTL_DIR" ]]; then
    echo "ERROR: $RTL_DIR does not exist." >&2
    exit 1
fi

shopt -s nullglob

v_count=0
dat_count=0

for f in "$HLS_DIR"/*.v; do
    cp -f "$f" "$RTL_DIR/"
    v_count=$((v_count + 1))
done

for f in "$HLS_DIR"/*.dat; do
    cp -f "$f" "$RTL_DIR/"
    dat_count=$((dat_count + 1))
done

echo "Copied $v_count .v file(s) and $dat_count .dat file(s) from"
echo "  $HLS_DIR"
echo "to"
echo "  $RTL_DIR"
