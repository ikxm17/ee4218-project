#!/usr/bin/env bash
# Install base system packages
set -euo pipefail

echo "=== System base packages ==="

echo "Updating package lists..."
apt-get update

echo "Upgrading existing packages..."
apt-get upgrade -y

echo "Installing base packages..."
apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    python3-dev \
    python3-pip \
    python3.10-venv \
    net-tools \
    openssh-server \
    i2c-tools \
    v4l-utils \
    vim \
    tmux \
    gh

echo "Enabling SSH..."
systemctl enable --now ssh

echo "Base system packages installed."
