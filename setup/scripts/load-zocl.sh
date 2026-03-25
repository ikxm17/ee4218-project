#!/usr/bin/env bash
# Load the ZOCL device tree overlay to create /dev/dri/renderD128.
#
# PYNQ on Kria needs the XRT render device to program the FPGA, but the
# render device only exists after the ZOCL DT node is applied.  This
# script applies a minimal ZOCL-only overlay as a bootstrap step.
#
# Run this once after each board reboot, before using PYNQ:
#   sudo bash setup/scripts/load-zocl.sh
#
# The overlay persists until reboot or manual removal.

set -euo pipefail

OVERLAY_NAME="zocl"
OVERLAY_DIR="/sys/kernel/config/device-tree/overlays/$OVERLAY_NAME"

# Check if already loaded
if [ -d "$OVERLAY_DIR" ] && [ -s "$OVERLAY_DIR/dtbo" ]; then
    if [ -e /dev/dri/renderD128 ]; then
        echo "ZOCL overlay already loaded (/dev/dri/renderD128 exists)."
        exit 0
    fi
    # Stale overlay — remove and re-apply
    rmdir "$OVERLAY_DIR" 2>/dev/null || true
fi

# Create and apply the ZOCL-only overlay inline
DTS=$(cat <<'DTS_EOF'
/dts-v1/;
/plugin/;
&amba {
    zyxclmm_drm {
        compatible = "xlnx,zocl";
        status = "okay";
    };
};
DTS_EOF
)

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "$DTS" > "$TMPDIR/zocl.dts"
dtc -@ -I dts -O dtb -o "$TMPDIR/zocl.dtbo" "$TMPDIR/zocl.dts" 2>/dev/null

mkdir -p "$OVERLAY_DIR"
cat "$TMPDIR/zocl.dtbo" > "$OVERLAY_DIR/dtbo"

# Verify
if [ -e /dev/dri/renderD128 ]; then
    echo "ZOCL overlay loaded. /dev/dri/renderD128 ready."
else
    echo "Warning: ZOCL overlay applied but renderD128 not found."
    ls -la /dev/dri/
    exit 1
fi
