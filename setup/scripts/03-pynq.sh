#!/usr/bin/env bash
# Install PYNQ framework and create shared Python venv
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
APT_OPTS=(-y -o Dpkg::Options::="--force-confold")

VENV_DIR="/opt/ee4218/venv"
PROFILE_SCRIPT="/etc/profile.d/ee4218.sh"
XOCL_FILE="/etc/xocl.txt"
BOARD_USER="ubuntu"

echo "=== PYNQ framework setup ==="

# ── System dependencies ──────────────────────────────────────────────
echo "Installing system dependencies..."
apt-get update -qq
apt-get "${APT_OPTS[@]}" install \
    python3-cffi \
    libffi-dev \
    libdrm-dev \
    device-tree-compiler

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
