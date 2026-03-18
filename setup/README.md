# Kria KV260 Setup — Ubuntu 22.04

Environment setup for Kria KV260 boards. Configures static IP networking, base packages, and Tailscale for remote access.

## Prerequisites

- Xilinx Kria KV260 Vision AI Starter Kit
- Power supply (12V barrel jack)
- microSD card (32 GB+)
- Ethernet cable + router
- Host PC with SD card reader

## 1. Flash SD Card

Download the Ubuntu 22.04 image for Kria from [Canonical](https://ubuntu.com/download/amd-xilinx) and flash it with [Raspberry Pi Imager](https://www.raspberrypi.com/software/) or `dd`:

```bash
# Example with dd (replace /dev/sdX with your SD card device)
xzcat ubuntu-22.04-preinstalled-server-arm64+xlnx-zynqmp.img.xz | sudo dd of=/dev/sdX bs=4M status=progress
sync
```

## 2. Prep SD Card (Host-side)

Run on your PC after flashing. This pre-configures static IP and SSH so the board is headless-ready on first boot.

```bash
sudo bash setup/prep-sd.sh --device /dev/sdX --board-num <N> --gateway <GATEWAY_IP>
```

**Arguments:**

| Flag | Description | Example |
|------|-------------|---------|
| `--device` | SD card block device | `/dev/sdb` |
| `--board-num` | Board number (determines IP: gateway_base + 100 + N) | `1` → `.101` |
| `--gateway` | Router IP address | `192.168.1.1` |
| `--ssh-key` | Path to SSH public key (optional, auto-detects `~/.ssh/id_*.pub`) | `~/.ssh/id_ed25519.pub` |

**Example — 3 boards on a 192.168.1.x network:**

```bash
sudo bash setup/prep-sd.sh --device /dev/sdb --board-num 1 --gateway 192.168.1.1  # → 192.168.1.101
sudo bash setup/prep-sd.sh --device /dev/sdb --board-num 2 --gateway 192.168.1.1  # → 192.168.1.102
sudo bash setup/prep-sd.sh --device /dev/sdb --board-num 3 --gateway 192.168.1.1  # → 192.168.1.103
```

## 3. Boot and SSH In

1. Insert the SD card into the KV260
2. Connect Ethernet and power
3. Wait ~60 seconds for boot
4. SSH in:

```bash
ssh ubuntu@192.168.1.101   # default password: ubuntu
```

## 4. Run Setup (On-board)

```bash
sudo bash setup/setup.sh
```

This runs all setup scripts in order:

| Script | Purpose |
|--------|---------|
| `00-preflight.sh` | Checks arch, OS, disk, network |
| `01-system-base.sh` | apt update/upgrade + essential packages |
| `02-tailscale.sh` | Installs Tailscale VPN |
| `99-verify.sh` | Smoke tests |

**Skip a step:**

```bash
sudo bash setup/setup.sh --skip-tailscale
```

**Re-run a single step:**

```bash
sudo bash setup/scripts/01-system-base.sh
```

## 5. Tailscale Authentication

After setup completes, authenticate each board with Tailscale:

```bash
sudo tailscale up
```

Follow the printed URL to log in. Each board appears in your Tailscale network.

## 6. Verify

```bash
sudo bash setup/scripts/99-verify.sh
```

All checks should show `[PASS]`. Tailscale connection is `[INFO]` (passes if authenticated).

## Adding Packages Later

Add new numbered scripts to `setup/scripts/`:

```
scripts/03-pynq.sh
scripts/04-tflite.sh
```

The orchestrator (`setup.sh`) runs `scripts/[0-9]*.sh` in sorted order. Each script:
- Is self-contained and can be run independently
- Should use `set -euo pipefail`
- Gets a `--skip-<name>` flag automatically (derived from filename)

## File Structure

```
setup/
├── README.md                      # This file
├── setup.sh                       # On-board orchestrator
├── prep-sd.sh                     # Host-side SD card prep
├── scripts/
│   ├── 00-preflight.sh            # Pre-flight checks
│   ├── 01-system-base.sh          # Base packages
│   ├── 02-tailscale.sh            # Tailscale VPN
│   └── 99-verify.sh               # Smoke tests
└── config/
    └── netplan-static.yaml.tpl    # Netplan template
```
