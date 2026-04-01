#!/usr/bin/env bash
# Inspect a Vivado .xsa or .hwh file and print the IP address map.
#
# Useful for verifying that overlay driver code (software/overlay/) uses
# the correct IP instance names after a block design rebuild.
#
# Usage:
#   bash scripts/inspect-hwh.sh                       # auto-detect from hardware/output/
#   bash scripts/inspect-hwh.sh path/to/design.xsa    # from .xsa archive
#   bash scripts/inspect-hwh.sh path/to/design.hwh    # from .hwh directly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_DIR/hardware/output"

INPUT="${1:-}"
CLEANUP=""

# --- Locate input file ---
if [ -z "$INPUT" ]; then
    # Auto-detect: prefer .hwh, fall back to .xsa
    shopt -s nullglob
    hwh_files=("$OUTPUT_DIR"/*.hwh)
    xsa_files=("$OUTPUT_DIR"/*.xsa)
    shopt -u nullglob

    if [ ${#hwh_files[@]} -ge 1 ]; then
        INPUT="${hwh_files[0]}"
    elif [ ${#xsa_files[@]} -ge 1 ]; then
        INPUT="${xsa_files[0]}"
    else
        echo "Error: no .hwh or .xsa files found in $OUTPUT_DIR" >&2
        echo "Usage: bash scripts/inspect-hwh.sh [<file.xsa|file.hwh>]" >&2
        exit 1
    fi
    echo "Auto-detected: $INPUT"
fi

if [ ! -f "$INPUT" ]; then
    echo "Error: file not found: $INPUT" >&2
    exit 1
fi

# --- If .xsa, extract .hwh to temp dir ---
HWH_FILE="$INPUT"
if [[ "$INPUT" == *.xsa ]]; then
    TMPDIR="$(mktemp -d)"
    CLEANUP="$TMPDIR"
    trap 'rm -rf "$CLEANUP"' EXIT

    unzip -q -o "$INPUT" -d "$TMPDIR"

    # The .xsa contains multiple .hwh files: top-level design + sub-IP .hwh
    # files (e.g., for SmartConnect, CSI-2 RX subsystem).  The top-level
    # .hwh is the shortest filename (no sub-IP suffix) and the one PYNQ uses.
    HWH_FILE="$(find "$TMPDIR" -name '*.hwh' -type f \
        | awk '{ print length($0), $0 }' | sort -n | head -1 | cut -d' ' -f2-)"
    if [ -z "$HWH_FILE" ]; then
        echo "Error: no .hwh found inside $INPUT" >&2
        exit 1
    fi
    echo "Using top-level .hwh: $(basename "$HWH_FILE")"
fi

# --- Parse .hwh and print IP map ---
HWH_FILE_ESC="$HWH_FILE" python3 << 'PYEOF'
import os
from xml.etree import ElementTree

hwh_path = os.environ["HWH_FILE_ESC"]
tree = ElementTree.parse(hwh_path)
root = tree.getroot()

# Collect addressable IPs
ips = []
for module in root.iter("MODULE"):
    instance = module.get("INSTANCE", "")
    vlnv = module.get("VLNV", "")
    ip_type = vlnv.split(":")[-2] if ":" in vlnv else vlnv

    # Find base address from MEMORYMAP or PARAMETERS
    base_addr = None
    high_addr = None

    for addr_block in module.iter("ADDRESSBLOCK"):
        base = addr_block.get("BASEVALUE")
        high = addr_block.get("HIGHVALUE")
        if base:
            base_addr = base
            high_addr = high
            break

    if not base_addr:
        for param in module.iter("PARAMETER"):
            pname = param.get("NAME", "")
            # Standard AXI: C_BASEADDR / C_HIGHADDR
            # HLS IPs: C_S_AXI_CTRL_BASEADDR / C_S_AXI_CTRL_HIGHADDR
            if pname in ("C_BASEADDR", "C_S_AXI_CTRL_BASEADDR"):
                base_addr = param.get("VALUE")
            elif pname in ("C_HIGHADDR", "C_S_AXI_CTRL_HIGHADDR"):
                high_addr = param.get("VALUE")

    ips.append((instance, ip_type, base_addr, high_addr))

# Print header
print()
print(f"{'Instance Name':<30} {'IP Type':<25} {'Base Address':<14} {'High Address':<14}")
print("-" * 83)

# Addressable IPs first, then infrastructure
addressable = [(n, t, b, h) for n, t, b, h in ips if b]
infra = [(n, t, b, h) for n, t, b, h in ips if not b]

for name, ip_type, base, high in sorted(addressable, key=lambda x: int(x[2], 0) if x[2] else 0):
    print(f"{name:<30} {ip_type:<25} {base:<14} {high or '':<14}")

if infra:
    print()
    print("Infrastructure (no address map):")
    for name, ip_type, _, _ in sorted(infra, key=lambda x: x[0]):
        print(f"  {name:<28} {ip_type}")

# Print overlay driver cross-reference
print()
print("Driver cross-reference (software/overlay/camera.py):")
driver_names = {
    "IP_DEMOSAIC":     "v_demosaic_0",
    "IP_GAMMA":        "v_gamma_lut_0",
    "IP_VDMA":         "axi_vdma_0",
    "IP_MULTI_SCALER": "v_multi_scaler_0",
}
all_instances = {n for n, _, _, _ in ips}
for const, expected in driver_names.items():
    status = "OK" if expected in all_instances else "MISSING"
    print(f"  {const:<20} -> {expected:<25} [{status}]")

PYEOF
