#!/usr/bin/env bash
#
# scripts/vivado-setup.sh
#
# Unified entry point for the Vivado project workflow.
# Handles the full pipeline from any state to a working project:
#
#   1. CLEAN   — remove junk from ip_repo/src/
#   2. SYNC    — copy rtl/ → ip_repo/src/
#   3. PACKAGE — run package_accelerator_ip.tcl (Vivado batch)
#   4. VALIDATE — run validate-ip-repo.sh
#   5. RECREATE — run recreate_project.tcl (Vivado batch)
#   6. ACTION  — open GUI / build / simulate
#
# Each step checks whether work is needed and skips if up-to-date.
# Use --force to bypass freshness checks and rebuild everything.
#
# Usage:
#   ./scripts/vivado-setup.sh                # setup + open GUI
#   ./scripts/vivado-setup.sh --batch        # setup only, no GUI
#   ./scripts/vivado-setup.sh --build        # setup + synth/impl/bitstream
#   ./scripts/vivado-setup.sh --sim [flow]   # setup + simulation (default: behav)
#   ./scripts/vivado-setup.sh --force        # force full rebuild
#   ./scripts/vivado-setup.sh --force --build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

XPR="$REPO_ROOT/hardware/vivado/tinyissimoyolo/tinyissimoyolo.xpr"
IP_REPO="$REPO_ROOT/hardware/ip_repo"
BD_FILE="$REPO_ROOT/hardware/vivado/tinyissimoyolo/tinyissimoyolo.srcs/sources_1/bd/playground/playground.bd"
HLS_DIR="$REPO_ROOT/hardware/hls/tinyissimo_layer/tinyissimo_layer/hls/impl/ip/hdl/verilog"

# ─── Parse arguments ─────────────────────────────────────────────────
MODE="gui"
FORCE=false
SIM_FLOW="behav"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch)   MODE="batch"; shift ;;
        --build)   MODE="build"; shift ;;
        --sim)
            MODE="sim"
            shift
            if [[ $# -gt 0 ]] && [[ "$1" != --* ]]; then
                SIM_FLOW="$1"; shift
            fi
            ;;
        --force)   FORCE=true; shift ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--batch|--build|--sim [flow]] [--force]"
            echo ""
            echo "Modes:"
            echo "  (default)   setup + open Vivado GUI"
            echo "  --batch     setup only, no GUI"
            echo "  --build     setup + full synth/impl/bitstream"
            echo "  --sim [f]   setup + simulation (behav|post_synth|post_impl|post_timing)"
            echo ""
            echo "Options:"
            echo "  --force     bypass freshness checks, rebuild everything"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

cd "$REPO_ROOT"

echo "============================================"
echo " Vivado Project Setup"
echo "  Mode:  $MODE"
echo "  Force: $FORCE"
echo "============================================"
echo

# ─── Freshness helpers ───────────────────────────────────────────────
# Returns 0 (true) if $1 is newer than all files matching $2 glob pattern
newer_than_all() {
    local ref="$1"
    shift
    [[ ! -f "$ref" ]] && return 1
    for f in "$@"; do
        [[ -f "$f" ]] && [[ "$f" -nt "$ref" ]] && return 1
    done
    return 0
}

# Find the active component.xml
find_component_xml() {
    local pkg_xml="$IP_REPO/tinyissimoyolo_accelerator_v1_0/component.xml"
    local root_xml="$IP_REPO/component.xml"
    if [[ -f "$pkg_xml" ]]; then
        echo "$pkg_xml"
    elif [[ -f "$root_xml" ]]; then
        echo "$root_xml"
    else
        echo ""
    fi
}

# ─── Step 1: CLEAN + SYNC ────────────────────────────────────────────
echo "── Step 1: Sync RTL → IP repo ──"

need_sync=false
if $FORCE; then
    need_sync=true
elif [[ ! -d "$IP_REPO/src" ]]; then
    need_sync=true
else
    # Check if any rtl/ file is newer (content-wise) than ip_repo/src/
    shopt -s nullglob
    for f in "$REPO_ROOT"/hardware/rtl/*.{v,sv,dat}; do
        base="$(basename "$f")"
        ip_f="$IP_REPO/src/$base"
        if [[ ! -f "$ip_f" ]] || ! cmp -s "$f" "$ip_f"; then
            need_sync=true
            break
        fi
    done
    shopt -u nullglob
fi

if $need_sync; then
    echo "  Running sync-ip-src.sh ..."
    bash "$SCRIPT_DIR/sync-ip-src.sh"
else
    echo "  ip_repo/src/ is up to date — skipping sync"
fi
echo

# ─── Step 2: PACKAGE IP ──────────────────────────────────────────────
echo "── Step 2: Package accelerator IP ──"

# Auto-fix: remove duplicate component.xml (stale root copy alongside
# the subdirectory package) BEFORE checking freshness.
if [[ -f "$IP_REPO/component.xml" ]] && \
   [[ -f "$IP_REPO/tinyissimoyolo_accelerator_v1_0/component.xml" ]]; then
    echo "  Auto-fix: removing stale root component.xml (keeping subdirectory version)"
    rm -f "$IP_REPO/component.xml"
    rm -rf "$IP_REPO/xgui" "$IP_REPO/sim"
fi

need_package=false
component_xml=$(find_component_xml)

if $FORCE; then
    need_package=true
elif [[ -z "$component_xml" ]]; then
    need_package=true
elif [[ ! -d "$HLS_DIR" ]]; then
    echo "  WARNING: HLS output directory not found at $HLS_DIR"
    echo "           Re-export from Vitis HLS if needed."
    need_package=true
else
    # Check if any source file is newer than component.xml
    shopt -s nullglob
    for f in "$REPO_ROOT"/hardware/rtl/*.{v,sv,dat} "$HLS_DIR"/*.{v,dat}; do
        if [[ -f "$f" ]] && [[ "$f" -nt "$component_xml" ]]; then
            need_package=true
            break
        fi
    done
    shopt -u nullglob
fi

if $need_package; then
    echo "  Running package_accelerator_ip.tcl ..."
    vivado -mode batch -nojournal -nolog \
        -source "$REPO_ROOT/hardware/scripts/package_accelerator_ip.tcl"
    echo "  IP packaged successfully"
else
    echo "  component.xml is up to date — skipping packaging"
fi
echo

# ─── Step 3: VALIDATE ────────────────────────────────────────────────
echo "── Step 3: Validate IP repo ──"
if ! bash "$SCRIPT_DIR/validate-ip-repo.sh"; then
    if $FORCE; then
        echo ""
        echo "ERROR: IP repo validation failed even with --force."
        echo "       Check the errors above and fix manually."
        exit 1
    else
        echo ""
        echo "Validation failed — retrying with forced repackaging..."
        echo ""
        # Force a full repackage and re-validate
        echo "  Running package_accelerator_ip.tcl ..."
        vivado -mode batch -nojournal -nolog \
            -source "$REPO_ROOT/hardware/scripts/package_accelerator_ip.tcl"
        echo ""
        if ! bash "$SCRIPT_DIR/validate-ip-repo.sh"; then
            echo ""
            echo "ERROR: IP repo validation still failing after repackaging."
            echo "       Try: ./scripts/vivado-setup.sh --force"
            exit 1
        fi
    fi
fi
echo

# ─── Step 4: RECREATE PROJECT ────────────────────────────────────────
echo "── Step 4: Recreate Vivado project ──"

need_recreate=false
component_xml=$(find_component_xml)

if $FORCE; then
    need_recreate=true
elif [[ ! -f "$XPR" ]]; then
    need_recreate=true
elif [[ -n "$component_xml" ]] && [[ "$component_xml" -nt "$XPR" ]]; then
    need_recreate=true
elif [[ "$BD_FILE" -nt "$XPR" ]]; then
    need_recreate=true
fi

if $need_recreate; then
    echo "  Running recreate_project.tcl ..."
    vivado -mode batch -nojournal -nolog \
        -source "$REPO_ROOT/scripts/recreate_project.tcl"
    echo "  Project recreated successfully"
else
    echo "  .xpr is up to date — skipping recreation"
fi
echo

# ─── Step 5: MODE-DEPENDENT ACTION ───────────────────────────────────
case "$MODE" in
    batch)
        echo "============================================"
        echo " Setup complete (batch mode)."
        echo " Project: $XPR"
        echo "============================================"
        ;;
    gui)
        echo "============================================"
        echo " Opening Vivado GUI ..."
        echo "============================================"
        vivado "$XPR" &
        ;;
    build)
        echo "============================================"
        echo " Starting full build (synth + impl + bitstream) ..."
        echo "============================================"
        vivado -mode batch -nojournal -nolog \
            -source "$REPO_ROOT/scripts/rebuild_bitstream.tcl"
        ;;
    sim)
        echo "============================================"
        echo " Starting simulation (flow: $SIM_FLOW) ..."
        echo "============================================"
        vivado -mode batch -nojournal -nolog \
            -source "$REPO_ROOT/scripts/run_sim_headless.tcl" \
            -tclargs "$SIM_FLOW"
        ;;
esac
