#!/usr/bin/env bash
# Install PYNQ framework and create shared Python venv
#
# Adapts the system-level prerequisites from Xilinx/Kria-PYNQ install.sh
# (apt packages, Xilinx PPA, libcma) but installs Python packages into an
# isolated venv instead of system-wide. Skips Jupyter, notebooks, MicroBlaze
# compiler, DPU, and composable pipeline.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
APT_OPTS=(-y -o Dpkg::Options::="--force-confold")

VENV_DIR="/opt/ee4218/venv"
PROFILE_SCRIPT="/etc/profile.d/ee4218.sh"
XOCL_FILE="/etc/xocl.txt"
BOARD_USER="ubuntu"
PYNQ_BRANCH="v3.0.1"
PYNQ_CLONE="/tmp/pynq-src"

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

# ── libcma (CMA allocator library + header) ──────────────────────────
# PYNQ's C extensions need libxlnk_cma.h at build time and libcma.so at
# runtime. The kria-pynq script gets these from pynq/sdbuild/packages/libsds.
# Pre-built .so is in the repo; make -t skips recompilation, make install
# copies the .so and .h to system paths.
if [ ! -f /usr/include/libxlnk_cma.h ]; then
    echo "Installing libcma..."
    if [ ! -d "$PYNQ_CLONE" ]; then
        git clone --depth 1 --branch "$PYNQ_BRANCH" \
            https://github.com/Xilinx/PYNQ.git "$PYNQ_CLONE"
    fi
    pushd "$PYNQ_CLONE/sdbuild/packages/libsds/libcma" > /dev/null
    make -t
    DESTDIR="" make install
    ldconfig
    popd > /dev/null
else
    echo "libcma already installed, skipping."
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
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

# ── PYNQ ─────────────────────────────────────────────────────────────
echo "Installing PYNQ..."
"$VENV_DIR/bin/pip" install pynq numpy

# ── Cleanup ──────────────────────────────────────────────────────────
if [ -d "$PYNQ_CLONE" ]; then
    echo "Cleaning up PYNQ source clone..."
    rm -rf "$PYNQ_CLONE"
fi

# ── Ownership ────────────────────────────────────────────────────────
echo "Setting venv ownership to $BOARD_USER..."
chown -R "$BOARD_USER:$BOARD_USER" /opt/ee4218

# ── Environment variables ────────────────────────────────────────────
echo "Writing $PROFILE_SCRIPT..."
cat > "$PROFILE_SCRIPT" << 'PROFILE'
export BOARD=KV260
export XILINX_XRT=/usr
export VIRTUAL_ENV=/opt/ee4218/venv
export PATH="/opt/ee4218/venv/bin:$PATH"
PROFILE

# ── Board identification ─────────────────────────────────────────────
echo "KV260" > "$XOCL_FILE"

echo "PYNQ framework setup complete."
