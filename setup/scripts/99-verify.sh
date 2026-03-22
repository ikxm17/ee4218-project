#!/usr/bin/env bash
# Smoke tests — verify setup is correct
set -uo pipefail

FAIL=0

pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; FAIL=1; }
info() { echo "[INFO] $1"; }

echo "=== Verification ==="

# Architecture
if [ "$(uname -m)" = "aarch64" ]; then
    pass "Architecture is aarch64"
else
    fail "Architecture is $(uname -m), expected aarch64"
fi

# Python 3.10
if python3 --version 2>/dev/null | grep -q "3.10"; then
    pass "Python $(python3 --version 2>&1 | awk '{print $2}') available"
else
    fail "Python 3.10 not found"
fi

# Static IP on eth0
ETH0_IP=$(ip -4 addr show eth0 2>/dev/null | grep -oP 'inet \K[0-9.]+' | head -1)
if [ -n "$ETH0_IP" ]; then
    pass "Static IP set on eth0: $ETH0_IP"
else
    fail "No IPv4 address on eth0"
fi

# Gateway reachable
GATEWAY=$(ip route show default 2>/dev/null | awk '{print $3}' | head -1)
if [ -n "$GATEWAY" ] && ping -c1 -W3 "$GATEWAY" >/dev/null 2>&1; then
    pass "Gateway reachable ($GATEWAY)"
else
    fail "Gateway not reachable (${GATEWAY:-none found})"
fi

# Tailscale installed
if command -v tailscale >/dev/null 2>&1; then
    TS_VER=$(tailscale --version 2>/dev/null | head -1)
    pass "Tailscale installed ($TS_VER)"
else
    fail "Tailscale not installed"
fi

# Tailscale status (informational only)
if tailscale status >/dev/null 2>&1; then
    info "Tailscale connected"
else
    info "Tailscale not connected (run: sudo tailscale up)"
fi

# SSH running
if systemctl is-active --quiet ssh 2>/dev/null || systemctl is-active --quiet sshd 2>/dev/null; then
    pass "SSH service running"
else
    fail "SSH service not running"
fi

# Disk space
FREE_KB=$(df --output=avail / | tail -1 | tr -d ' ')
FREE_GB=$(( FREE_KB / 1048576 ))
info "Disk free: ${FREE_GB} GB on /"

# IP summary
info "eth0 IP: ${ETH0_IP:-N/A}"
TS_IP=$(tailscale ip -4 2>/dev/null || true)
if [ -n "$TS_IP" ]; then
    info "Tailscale IP: $TS_IP"
fi

# PYNQ venv
VENV_DIR="/opt/ee4218/venv"
if [ -x "$VENV_DIR/bin/python3" ]; then
    pass "Python venv exists at $VENV_DIR"
else
    fail "Python venv not found at $VENV_DIR"
fi

# PYNQ importable
if "$VENV_DIR/bin/python3" -c "from pynq import Overlay" 2>/dev/null; then
    pass "PYNQ importable"
else
    fail "PYNQ import failed"
fi

# FPGA manager
if [ -d /sys/class/fpga_manager/ ]; then
    pass "FPGA manager sysfs accessible"
else
    fail "FPGA manager sysfs not found"
fi

# /dev/mem
if [ -c /dev/mem ]; then
    pass "/dev/mem character device exists"
else
    fail "/dev/mem not found"
fi

# BOARD env var (needs re-login to take effect)
if [ "${BOARD:-}" = "KV260" ]; then
    pass "BOARD=KV260 set"
else
    info "BOARD variable not set (re-login or: source /etc/profile.d/ee4218.sh)"
fi

echo ""
if [ "$FAIL" -ne 0 ]; then
    echo "Some checks FAILED."
    exit 1
fi
echo "All checks passed."
