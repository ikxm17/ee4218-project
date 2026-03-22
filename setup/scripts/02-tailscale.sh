#!/usr/bin/env bash
# Install and enable Tailscale
set -euo pipefail

echo "=== Tailscale installation ==="

# Add Tailscale GPG key and repository
echo "Adding Tailscale repository..."
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg \
    | tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null

curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.tailscale-keyring.list \
    | tee /etc/apt/sources.list.d/tailscale.list >/dev/null

export DEBIAN_FRONTEND=noninteractive

echo "Installing Tailscale..."
apt-get update
apt-get install -y tailscale

echo "Enabling tailscaled service..."
systemctl enable --now tailscaled

echo ""
echo "Tailscale installed. To authenticate, run:"
echo "  sudo tailscale up"
echo ""
echo "This requires interactive access — follow the URL it prints to log in."
