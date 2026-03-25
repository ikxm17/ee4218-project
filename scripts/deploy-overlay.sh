#!/usr/bin/env bash
# Deploy a Vivado overlay (.xsa) to the Kria board.
# Extracts .bit + .hwh from the .xsa archive, then rsyncs them to the
# board's hardware/output/ directory.
#
# Usage:
#   bash scripts/deploy-overlay.sh [--xsa <path>] [--board <host>] [--name <name>] [--local-only]
#
# Options:
#   --xsa          Path to .xsa file (default: auto-detect from hardware/output/*.xsa)
#   --board        SSH host alias (default: kria-01)
#   --name         Overlay base name for .bit/.hwh (default: basename of .xsa)
#   --local-only   Extract .xsa locally without deploying to board

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_DIR/hardware/output"

# --- Defaults ---
XSA=""
BOARD="kria-01"
NAME=""
LOCAL_ONLY=false

usage() {
    cat <<EOF
Usage: bash scripts/deploy-overlay.sh [options]

Extracts .bit + .hwh from a Vivado .xsa archive and deploys to the board.

Options:
  --xsa PATH       Path to .xsa file (default: auto-detect from hardware/output/*.xsa)
  --board HOST     SSH host alias (default: kria-01)
  --name NAME      Overlay base name (default: basename of .xsa without extension)
  --local-only     Extract locally only, skip deployment to board
  --help           Show this help message

Examples:
  # Auto-detect .xsa from hardware/output/
  bash scripts/deploy-overlay.sh

  # Explicit .xsa with custom name
  bash scripts/deploy-overlay.sh --xsa path/to/design.xsa --name camera_pipeline

  # Extract only (no board deployment)
  bash scripts/deploy-overlay.sh --xsa path/to/design.xsa --local-only
EOF
    exit 0
}

needs_value() {
    if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: $1 requires a value."
        exit 1
    fi
}

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --xsa)        needs_value "$@"; XSA="$2";   shift 2 ;;
        --board)      needs_value "$@"; BOARD="$2";  shift 2 ;;
        --name)       needs_value "$@"; NAME="$2";   shift 2 ;;
        --local-only) LOCAL_ONLY=true;               shift   ;;
        --help)       usage ;;
        *)
            echo "Error: unknown argument: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# --- Locate .xsa ---
if [ -z "$XSA" ]; then
    shopt -s nullglob
    xsa_files=("$OUTPUT_DIR"/*.xsa)
    shopt -u nullglob

    if [ ${#xsa_files[@]} -eq 0 ]; then
        echo "Error: no .xsa files found in $OUTPUT_DIR"
        echo "Either export from Vivado into hardware/output/ or specify --xsa <path>."
        exit 1
    elif [ ${#xsa_files[@]} -gt 1 ]; then
        echo "Error: multiple .xsa files found in $OUTPUT_DIR:"
        printf "  %s\n" "${xsa_files[@]}"
        echo "Specify one with --xsa <path>."
        exit 1
    fi

    XSA="${xsa_files[0]}"
    echo "Auto-detected: $XSA"
fi

if [ ! -f "$XSA" ]; then
    echo "Error: .xsa file not found: $XSA"
    exit 1
fi

# --- Derive overlay name ---
if [ -z "$NAME" ]; then
    NAME="$(basename "$XSA" .xsa)"
fi

# --- Extract .bit and .hwh from .xsa ---
echo "Extracting $XSA -> $OUTPUT_DIR/$NAME.{bit,hwh}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

unzip -q -o "$XSA" -d "$TMPDIR"

# Find .bit file inside the archive
BIT_FILE="$(find "$TMPDIR" -name '*.bit' -type f | head -1)"
if [ -z "$BIT_FILE" ]; then
    echo "Error: no .bit file found inside $XSA"
    exit 1
fi

# Find top-level .hwh file inside the archive.
# The .xsa contains multiple .hwh files: the top-level design plus sub-IP
# .hwh files (e.g., for SmartConnect, CSI-2 RX subsystem).  The top-level
# .hwh is the shortest filename (no sub-IP suffix) and the one PYNQ uses.
HWH_FILE="$(find "$TMPDIR" -name '*.hwh' -type f \
    | awk '{ print length($0), $0 }' | sort -n | head -1 | cut -d' ' -f2-)"
if [ -z "$HWH_FILE" ]; then
    echo "Error: no .hwh file found inside $XSA"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cp "$BIT_FILE" "$OUTPUT_DIR/$NAME.bit"
cp "$HWH_FILE" "$OUTPUT_DIR/$NAME.hwh"

echo "Extracted:"
echo "  $OUTPUT_DIR/$NAME.bit ($(du -h "$OUTPUT_DIR/$NAME.bit" | cut -f1))"
echo "  $OUTPUT_DIR/$NAME.hwh ($(du -h "$OUTPUT_DIR/$NAME.hwh" | cut -f1))"

if [ "$LOCAL_ONLY" = true ]; then
    echo ""
    echo "Done (local-only mode)."
    exit 0
fi

# --- Deploy to board ---
echo ""
echo "Deploying to $BOARD..."

# Verify board is reachable
if ! ssh -o ConnectTimeout=5 "$BOARD" true 2>/dev/null; then
    echo "Error: cannot reach $BOARD via SSH."
    echo "Check your SSH config and that the board is powered on."
    exit 1
fi

# Get project path on board
BOARD_PROJECT_DIR="$(ssh "$BOARD" 'cd ~/workspace/ee4218-project 2>/dev/null && pwd' || true)"
if [ -z "$BOARD_PROJECT_DIR" ]; then
    echo "Error: project directory not found on $BOARD at ~/workspace/ee4218-project"
    exit 1
fi

BOARD_OUTPUT_DIR="$BOARD_PROJECT_DIR/hardware/output"
ssh "$BOARD" "mkdir -p '$BOARD_OUTPUT_DIR'"

# --- Compile .dtbo if .dts exists ---
DTS_FILE="$OUTPUT_DIR/$NAME.dts"
if [ -f "$DTS_FILE" ]; then
    echo "Compiling device tree overlay: $DTS_FILE"
    if dtc -@ -I dts -O dtb -o "$OUTPUT_DIR/$NAME.dtbo" "$DTS_FILE" 2>&1 | grep -i error; then
        echo "Error: dtc compilation failed."
        exit 1
    fi
    echo "  $OUTPUT_DIR/$NAME.dtbo ($(du -h "$OUTPUT_DIR/$NAME.dtbo" | cut -f1))"
fi

# Build file list for rsync
DEPLOY_FILES=("$OUTPUT_DIR/$NAME.bit" "$OUTPUT_DIR/$NAME.hwh")
if [ -f "$OUTPUT_DIR/$NAME.dtbo" ]; then
    DEPLOY_FILES+=("$OUTPUT_DIR/$NAME.dtbo")
fi

rsync -avz --progress \
    "${DEPLOY_FILES[@]}" \
    "$BOARD:$BOARD_OUTPUT_DIR/"

# --- Smoke test: verify .hwh parses ---
echo ""
echo "Verifying .hwh on board..."
VERIFY_CMD="python3 -c \"
from xml.etree import ElementTree
tree = ElementTree.parse('$BOARD_OUTPUT_DIR/$NAME.hwh')
root = tree.getroot()
print(f'  HWH valid: {root.tag}, {len(list(root))} top-level elements')
\""

if ssh "$BOARD" "$VERIFY_CMD" 2>/dev/null; then
    echo "Verification passed."
else
    echo "Warning: .hwh verification failed (file may still be usable)."
fi

echo ""
echo "Deploy complete. Load in Python with:"
echo "  from pynq import Overlay"
echo "  ol = Overlay('$BOARD_OUTPUT_DIR/$NAME.bit')"
echo "  print(ol.ip_dict.keys())"
