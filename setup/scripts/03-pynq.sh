#!/usr/bin/env bash
# Install PYNQ framework and create shared Python venv
#
# Adapts the system-level prerequisites from Xilinx/Kria-PYNQ install.sh
# (apt packages, Xilinx PPA, libcma, xclbinutil, device tree overlay) but
# installs Python packages into an isolated venv instead of system-wide.
# Skips Jupyter, notebooks, MicroBlaze compiler, DPU, and composable pipeline.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
APT_OPTS=(-y -o Dpkg::Options::="--force-confold")

VENV_DIR="/opt/ee4218/ee4218-venv"
PROFILE_SCRIPT="/etc/profile.d/ee4218.sh"
XOCL_FILE="/etc/xocl.txt"
PYNQ_DTS_DIR="/usr/local/share/pynq-dts"
BOARD_USER="ubuntu"
PYNQ_BRANCH="v3.0.1"
PYNQ_CLONE="/tmp/pynq-src"
PYNQ_BINARIES_URL="https://www.xilinx.com/bin/public/openDownload?filename=pynq-v3.0-binaries.tar.gz"

echo "=== PYNQ framework setup ==="

# ── Xilinx PPA (for libdrm-xlnx-dev) ────────────────────────────────
echo "Adding Xilinx PPA..."
apt-key adv --recv-keys --keyserver hkp://keyserver.ubuntu.com:80 \
    803DDF595EA7B6644F9B96B752150A179A9E84C9 2>/dev/null || true
echo "deb http://ppa.launchpad.net/ubuntu-xilinx/updates/ubuntu jammy main" \
    > /etc/apt/sources.list.d/xilinx-gstreamer.list

# ── System dependencies ──────────────────────────────────────────────
# Mirrors the apt-get install from kria-pynq install.sh, minus packages
# already in 01-system-base.sh (build-essential, python3-dev, python3-pip,
# python3.10-venv, i2c-tools)
echo "Installing system dependencies..."
apt-get update -qq
apt-get "${APT_OPTS[@]}" install \
    python3-cffi \
    libffi-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libdrm-xlnx-dev \
    libboost-all-dev \
    device-tree-compiler

# ── Clone PYNQ source (for libcma and DTS) ───────────────────────────
if [ ! -f /usr/include/libxlnk_cma.h ] || [ ! -f "$PYNQ_DTS_DIR/pynq.dtbo" ]; then
    if [ ! -d "$PYNQ_CLONE" ]; then
        echo "Cloning PYNQ ${PYNQ_BRANCH} source..."
        git clone --depth 1 --branch "$PYNQ_BRANCH" \
            https://github.com/Xilinx/PYNQ.git "$PYNQ_CLONE"
    fi
fi

# ── libcma (CMA allocator library + header) ──────────────────────────
# PYNQ's C extensions need libxlnk_cma.h at build time and libcma.so at
# runtime. Pre-built .so is in the repo; make -t skips recompilation,
# make install copies the .so and .h to system paths.
if [ ! -f /usr/include/libxlnk_cma.h ]; then
    echo "Installing libcma..."
    pushd "$PYNQ_CLONE/sdbuild/packages/libsds/libcma" > /dev/null
    make -t
    DESTDIR="" make install
    ldconfig
    popd > /dev/null
else
    echo "libcma already installed, skipping."
fi

# ── xclbinutil (working version from pynq-v3.0-binaries) ────────────
# The XRT 2.13 bundled xclbinutil segfaults on aarch64. Download the
# working version from Xilinx's pre-built PYNQ binaries.
if ! /usr/local/bin/xclbinutil --version &>/dev/null; then
    echo "Installing working xclbinutil..."
    wget -q "$PYNQ_BINARIES_URL" -O /tmp/pynq-v3.0-binaries.tar.gz
    tar xzf /tmp/pynq-v3.0-binaries.tar.gz -C /tmp pynq-v3.0-binaries/xrt/xclbinutil
    cp /tmp/pynq-v3.0-binaries/xrt/xclbinutil /usr/local/bin/xclbinutil
    chmod +x /usr/local/bin/xclbinutil
    rm -rf /tmp/pynq-v3.0-binaries /tmp/pynq-v3.0-binaries.tar.gz
else
    echo "xclbinutil already installed, skipping."
fi

# ── pynq.dtbo (device tree overlay for ZOCL + AFI + UIO) ────────────
# Required for PYNQ device discovery. Creates the ZOCL DRM device
# (renderD128) that PYNQ uses to program the FPGA.
if [ ! -f "$PYNQ_DTS_DIR/pynq.dtbo" ]; then
    echo "Compiling pynq device tree overlay..."
    mkdir -p "$PYNQ_DTS_DIR"
    # Use the DTS from kria-pynq (inlined — it's 30 lines)
    cat > "$PYNQ_DTS_DIR/pynq.dts" << 'DTS'
/dts-v1/;
/plugin/;
/ {
    fragment@1 {
        target = <&amba>;
        overlay1: __overlay__ {
            afi0: afi0 {
                compatible = "xlnx,afi-fpga";
                config-afi = <0 0>, <1 0>, <2 0>, <3 0>, <4 0>, <5 0>, <6 0>,
                             <7 0>, <8 0>, <9 0>, <10 0>, <11 0>, <12 0>,
                             <13 0>, <14 0>, <15 0>;
            };
        };
    };
    fragment@2 {
        target = <&amba>;
        overlay2: __overlay__ {
            zocl: zyxclmm_drm {
                compatible = "xlnx,zocl";
                status = "okay";
            };
        };
    };
    fragment@3 {
        target = <&amba>;
        overlay3: __overlay__ {
            fabric: fabric@A0000000 {
                interrupts = <0x00 0x59 0x04>;
                interrupt-parent = <&gic>;
                compatible = "generic-uio";
                reg = <0x00 0xa0000000 0x00 0x10000>;
            };
        };
    };
};
DTS
    dtc -I dts -O dtb -o "$PYNQ_DTS_DIR/pynq.dtbo" "$PYNQ_DTS_DIR/pynq.dts" 2>/dev/null || true
else
    echo "pynq.dtbo already compiled, skipping."
fi

# Insert dtbo if not already active
if [ ! -d /sys/kernel/config/device-tree/overlays/pynq ]; then
    echo "Inserting pynq device tree overlay..."
    modprobe zocl 2>/dev/null || true
    mkdir -p /sys/kernel/config/device-tree/overlays/pynq
    cat "$PYNQ_DTS_DIR/pynq.dtbo" > /sys/kernel/config/device-tree/overlays/pynq/dtbo
fi

# ── Cleanup PYNQ source clone ────────────────────────────────────────
if [ -d "$PYNQ_CLONE" ]; then
    echo "Cleaning up PYNQ source clone..."
    rm -rf "$PYNQ_CLONE"
fi

# ── Python venv ──────────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/python3" ]; then
    echo "Creating Python venv at $VENV_DIR..."
    mkdir -p "$(dirname "$VENV_DIR")"
    python3 -m venv "$VENV_DIR"
else
    echo "Python venv already exists at $VENV_DIR, skipping creation."
fi

echo "Upgrading pip..."
# Pin setuptools<=80: pynqutils 0.1.2 requires setuptools<=80
"$VENV_DIR/bin/pip" install --upgrade pip "setuptools<=80" wheel

# ── PYNQ + pinned dependencies ───────────────────────────────────────
# Pin pynq==3.0.1: v3.1.2 requires external pyxrt (not in XRT 2.13)
# Pin pycparser<3: v3.0 removed plyparser module that PYNQ needs
echo "Installing PYNQ..."
"$VENV_DIR/bin/pip" install "pynq==3.0.1" "pycparser<3" "numpy==1.26.4" "smbus2"

# ── Ownership ────────────────────────────────────────────────────────
echo "Setting venv ownership to $BOARD_USER..."
chown -R "$BOARD_USER:$BOARD_USER" /opt/ee4218

# ── Systemd service for pynq.dtbo at boot ────────────────────────────
# Load ZOCL + AFI device tree overlay early so PYNQ can program the
# FPGA via XRT.  This replaces the profile.d approach which only ran
# on interactive login and had a broken sudo redirect.
echo "Installing pynq-dtbo systemd service..."
cat > /etc/systemd/system/pynq-dtbo.service << 'UNIT'
[Unit]
Description=Load PYNQ device tree overlay (ZOCL + AFI)
After=sys-kernel-config.mount
ConditionPathIsDirectory=!/sys/kernel/config/device-tree/overlays/pynq

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStartPre=/sbin/modprobe zocl
ExecStart=/bin/bash -c '\
    mkdir -p /sys/kernel/config/device-tree/overlays/pynq && \
    cat /usr/local/share/pynq-dts/pynq.dtbo > /sys/kernel/config/device-tree/overlays/pynq/dtbo'

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable pynq-dtbo.service

# Start it now if not already loaded
if [ ! -d /sys/kernel/config/device-tree/overlays/pynq ]; then
    systemctl start pynq-dtbo.service
fi

# ── Environment variables ────────────────────────────────────────────
# Board identification and XRT path for PYNQ device discovery.
echo "Writing $PROFILE_SCRIPT..."
cat > "$PROFILE_SCRIPT" << 'PROFILE'
export BOARD=KV260
export XILINX_XRT=/usr
PROFILE

# ── Board identification ─────────────────────────────────────────────
echo "KV260" > "$XOCL_FILE"

# ── Post-install sanity checks ───────────────────────────────────────
# Export board vars for PYNQ device discovery (profile was just written
# above but not sourced in the current shell).
export BOARD=KV260
export XILINX_XRT=/usr

echo "Running PYNQ sanity checks..."

# Import check — if this fails, the pip install is broken
"$VENV_DIR/bin/python3" -c "from pynq import Overlay; print('  import check: OK')"

# Device enumeration — verifies ZOCL driver and device tree overlay
"$VENV_DIR/bin/python3" -c "
from pynq import Device
dev = Device.active_device
if dev is not None:
    print(f'  device enumeration: {dev.name}')
else:
    print('  device enumeration: no active device (may need reboot)')
" || echo "  device enumeration: skipped (non-critical)"

# NOTE: Full overlay-load validation (Overlay('.bit')) deferred until
# project bitstream (.bit + .hwh) is available.

echo "PYNQ framework setup complete."
