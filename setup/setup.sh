#!/usr/bin/env bash
# Kria KV260 on-board setup orchestrator
# Usage: sudo bash setup.sh [--skip-tailscale] [--skip-system-base] ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
LOG_FILE="$HOME/kria-setup.log"

# --- Parse skip flags ---
declare -A SKIP=()
for arg in "$@"; do
    case "$arg" in
        --skip-*)
            name="${arg#--skip-}"
            SKIP["$name"]=1
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: sudo bash setup.sh [--skip-preflight] [--skip-system-base] [--skip-tailscale] [--skip-verify]"
            exit 1
            ;;
    esac
done

# --- Root check ---
if [ "$(id -u)" -ne 0 ]; then
    echo "Error: this script must be run as root (sudo)."
    exit 1
fi

# --- Shared constants ---
export KRIA_SETUP_DIR="$SCRIPT_DIR"

# --- Logging ---
exec > >(tee -a "$LOG_FILE") 2>&1
echo ""
echo "========================================"
echo " Kria KV260 Setup — $(date)"
echo "========================================"
echo "Log file: $LOG_FILE"
echo ""

# --- Script name mapping for skip flags ---
# Extract friendly name from script filename: 01-system-base.sh → system-base
script_skip_name() {
    local base
    base="$(basename "$1" .sh)"
    echo "${base#[0-9][0-9]-}"
}

# --- Run scripts in order ---
FAILED_STEP=""
for script in "$SCRIPTS_DIR"/[0-9]*.sh; do
    [ -f "$script" ] || continue

    name="$(script_skip_name "$script")"

    if [ "${SKIP[$name]+set}" = "set" ]; then
        echo "--- Skipping: $name (--skip-$name) ---"
        echo ""
        continue
    fi

    echo "--- Running: $(basename "$script") ---"
    if bash "$script"; then
        echo "--- Done: $(basename "$script") ---"
    else
        FAILED_STEP="$(basename "$script")"
        echo ""
        echo "FAILED at: $FAILED_STEP"
        echo "To re-run this step individually:"
        echo "  sudo bash $script"
        echo ""
        echo "To resume from the next step, re-run with:"
        echo "  sudo bash $SCRIPT_DIR/setup.sh --skip-$name"
        exit 1
    fi
    echo ""
done

echo "========================================"
echo " Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Authenticate Tailscale:  sudo tailscale up"
echo "  2. Verify setup:            sudo bash $SCRIPTS_DIR/99-verify.sh"
