#!/usr/bin/env bash
#
# scripts/vivado-setup.sh
#
# Prepares the Vivado project for use. Fixes common problems
# automatically — stale IP caches, missing files, duplicate
# component.xml, polluted ip_repo/src/.
#
# After this script completes, the project is ready to open in the
# Vivado GUI or use with rebuild_bitstream.tcl / run_sim_headless.tcl.
#
# Steps:
#   1. CLEAN + SYNC  — purge junk from ip_repo/src/, copy rtl/ sources
#   2. PACKAGE       — run package_accelerator_ip.tcl (Vivado batch)
#   3. VALIDATE      — run validate-ip-repo.sh
#   4. RECREATE      — run recreate_project.tcl (Vivado batch)
#
# Each step checks whether work is needed and skips if up-to-date.
# Use --force to bypass freshness checks and rebuild everything.
#
# Usage:
#   ./scripts/vivado-setup.sh          # setup (skip what's fresh)
#   ./scripts/vivado-setup.sh --force  # force full rebuild

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

XPR="$REPO_ROOT/hardware/vivado/tinyissimoyolo/tinyissimoyolo.xpr"
IP_REPO="$REPO_ROOT/hardware/ip_repo"

# ─── Parse arguments ─────────────────────────────────────────────────
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $(basename "$0")"
    echo ""
    echo "Prepares the Vivado project for use by syncing sources,"
    echo "packaging the IP, validating, and recreating the .xpr."
    echo "Always runs all steps unconditionally."
    exit 0
fi

cd "$REPO_ROOT"

echo "============================================"
echo " Vivado Project Setup"
echo "============================================"
echo

# ─── Step 1: CLEAN + SYNC ────────────────────────────────────────────
echo "── Step 1: Sync RTL → IP repo ──"
bash "$SCRIPT_DIR/sync-ip-src.sh"
echo

# ─── Step 2: PACKAGE IP ──────────────────────────────────────────────
echo "── Step 2: Package accelerator IP ──"

# Auto-fix: remove duplicate component.xml (stale root copy alongside
# the subdirectory package).
if [[ -f "$IP_REPO/component.xml" ]] && \
   [[ -f "$IP_REPO/tinyissimoyolo_accelerator_v1_0/component.xml" ]]; then
    echo "  Auto-fix: removing stale root component.xml (keeping subdirectory version)"
    rm -f "$IP_REPO/component.xml"
    rm -rf "$IP_REPO/xgui" "$IP_REPO/sim"
fi

echo "  Running package_accelerator_ip.tcl ..."
vivado -mode batch -nojournal -nolog \
    -source "$REPO_ROOT/hardware/scripts/package_accelerator_ip.tcl"
echo "  IP packaged successfully"
echo

# ─── Step 3: VALIDATE ────────────────────────────────────────────────
echo "── Step 3: Validate IP repo ──"
if ! bash "$SCRIPT_DIR/validate-ip-repo.sh"; then
    echo ""
    echo "ERROR: IP repo validation failed after packaging."
    echo "       Check the errors above and fix manually."
    exit 1
fi
echo

# ─── Step 4: RECREATE PROJECT ────────────────────────────────────────
# Always recreate — the .xpr is a derived artifact and recreation only
# takes ~30s.  Timestamp-based skipping is fragile (a stale .xpr can
# look "newer" than its sources after Vivado touches it).
echo "── Step 4: Recreate Vivado project ──"
echo "  Running recreate_project.tcl ..."
vivado -mode batch -nojournal -nolog \
    -source "$REPO_ROOT/scripts/recreate_project.tcl"
echo "  Project recreated successfully"
echo

# ─── Done ─────────────────────────────────────────────────────────────
echo "============================================"
echo " Setup complete."
echo " Project: $XPR"
echo ""
echo " Next steps:"
echo "   vivado $XPR"
echo "   vivado -mode batch -source scripts/rebuild_bitstream.tcl"
echo "   vivado -mode batch -source scripts/run_sim_headless.tcl"
echo "============================================"
