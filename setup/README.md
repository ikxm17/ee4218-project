# Kria KV260 Setup — Ubuntu 22.04

Environment setup for Kria KV260 boards. Configures static IP networking, base packages, and Tailscale for remote access.

## Prerequisites

- Xilinx Kria KV260 Vision AI Starter Kit
- Power supply (12V barrel jack)
- microSD card (32 GB+)
- Ethernet cable + router
- Host PC with SD card reader

## 1. Flash + Prep SD Card

One command downloads Ubuntu 22.04, flashes it to the SD card, and configures static IP + SSH:

```bash
bash setup/prep-sd.sh --device /dev/sdX --board-num <N> --gateway <GATEWAY_IP>
```

The script prompts for `sudo` only when needed (flashing, mounting). The image (~2 GB) is cached in `~/.cache/kria-setup/` after the first download, so subsequent boards don't re-download.

**Arguments:**

| Flag | Description | Example |
|------|-------------|---------|
| `--device` | SD card block device | `/dev/sdb` |
| `--board-num` | Board number (determines IP: gateway_base + 100 + N) | `1` → `.101` |
| `--gateway` | Router IP address | `192.168.1.1` |
| `--ssh-key` | Path to SSH public key (optional, auto-detects `~/.ssh/id_*.pub`) | `~/.ssh/id_ed25519.pub` |
| `--no-flash` | Skip download + flash (config only on an already-flashed card) | |
| `--image` | Use a local `.img.xz` file instead of downloading | `~/Downloads/kria.img.xz` |
| `--clean-cache` | Remove cached image after flashing | |

**Example — 3 boards on a 192.168.1.x network:**

```bash
bash setup/prep-sd.sh --device /dev/sdb --board-num 1 --gateway 192.168.1.1  # → 192.168.1.101
bash setup/prep-sd.sh --device /dev/sdb --board-num 2 --gateway 192.168.1.1  # → 192.168.1.102
bash setup/prep-sd.sh --device /dev/sdb --board-num 3 --gateway 192.168.1.1  # → 192.168.1.103
```

**Re-configure an already-flashed card** (e.g., change board number):

```bash
bash setup/prep-sd.sh --device /dev/sdb --board-num 4 --gateway 192.168.1.1 --no-flash
```

## 2. Boot and SSH In

1. Insert the SD card into the KV260
2. Connect Ethernet and power
3. Wait ~60 seconds for boot
4. SSH in:

```bash
ssh ubuntu@192.168.1.101   # default password: ubuntu
```

## 3. Run Setup (On-board)

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

## 4. Tailscale Authentication

After setup completes, authenticate each board with Tailscale:

```bash
sudo tailscale up
```

Follow the printed URL to log in. Each board appears in your Tailscale network.

## 5. Verify

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
