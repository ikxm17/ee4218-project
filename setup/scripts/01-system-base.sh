#!/usr/bin/env bash
# Install base system packages
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
APT_OPTS=(-y -o Dpkg::Options::="--force-confold")

echo "=== System base packages ==="

echo "Updating package lists..."
apt-get update

echo "Upgrading existing packages..."
apt-get "${APT_OPTS[@]}" upgrade

echo "Installing base packages..."
apt-get "${APT_OPTS[@]}" install \
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
