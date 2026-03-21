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
bash prep-sd.sh --device /dev/sdX --board-num <N> --gateway <GATEWAY_IP>
```

The script prompts for `sudo` only when needed (flashing, mounting). The image (~2 GB) is cached in `~/.cache/kria-setup/` after the first download, so subsequent boards don't re-download.

**Arguments:**

| Flag | Description | Example |
|------|-------------|---------|
| `--device` | SD card block device | `/dev/sdb` |
| `--board-num` | Board number (determines IP: gateway_base + 100 + N) | `1` → `.101` |
| `--gateway` | Router IP address | `192.168.1.1` |
| `--ssh-key` | Path to SSH public key (optional, auto-detects `~/.ssh/id_*.pub`) | `~/.ssh/id_ed25519.pub` |
| `--password` | Custom password for `ubuntu` user (optional, default: `ubuntu`) | `mypass` |
| `--no-flash` | Skip download + flash (config only on an already-flashed card) | |
| `--image` | Use a local `.img.xz` file instead of downloading | `~/Downloads/kria.img.xz` |
| `--clean-cache` | Remove cached image after flashing | |

Each board gets a hostname derived from its board number: `kria-01`, `kria-02`, etc. The forced password change on first login is disabled — boards boot ready for SSH with no interactive prompts.

**Example — 3 boards on a 192.168.1.x network:**

```bash
bash prep-sd.sh --device /dev/sdb --board-num 1 --gateway 192.168.1.1  # → 192.168.1.101
bash prep-sd.sh --device /dev/sdb --board-num 2 --gateway 192.168.1.1  # → 192.168.1.102
bash prep-sd.sh --device /dev/sdb --board-num 3 --gateway 192.168.1.1  # → 192.168.1.103
```

**Re-configure an already-flashed card** (e.g., change board number):

```bash
bash prep-sd.sh --device /dev/sdb --board-num 4 --gateway 192.168.1.1 --no-flash
```

## 2. Boot and SSH In

1. Insert the SD card into the KV260
2. Connect Ethernet and power
3. Wait ~60 seconds for boot
4. SSH in:

```bash
ssh ubuntu@192.168.1.101   # password: ubuntu (or custom if --password was used)
```

No forced password change on first login — the board is ready to use immediately.

## 3. Run Setup (On-board)

```bash
sudo bash setup.sh
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
sudo bash setup.sh --skip-tailscale
```

**Re-run a single step:**

```bash
sudo bash scripts/01-system-base.sh
```

## 4. Tailscale Authentication

After setup completes, authenticate each board with Tailscale:

```bash
sudo tailscale up
```

Follow the printed URL to log in. Each board appears in your Tailscale network.

## 5. Verify

```bash
sudo bash scripts/99-verify.sh
```

All checks should show `[PASS]`. Tailscale connection is `[INFO]` (passes if authenticated).

## SSH Config (Host-side)

Add the following to `~/.ssh/config` on your host machine for automatic local/Tailscale failover:

```ssh-config
Match host kria exec "nc -z -w1 <LOCAL_IP> 22 2>/dev/null"
    Hostname <LOCAL_IP>

Host kria
    Hostname <TAILSCALE_IP>
    User ubuntu
```

This probes the local IP first — if reachable, it connects directly. Otherwise it falls back to the Tailscale IP. Replace `<LOCAL_IP>` with the board's static IP (e.g. `192.168.1.101`) and `<TAILSCALE_IP>` with the IP shown after running `sudo tailscale up` on the board.

For multiple boards, repeat the pattern with different host aliases and IPs:

```ssh-config
Match host kria-01 exec "nc -z -w1 <LOCAL_IP_1> 22 2>/dev/null"
    Hostname <LOCAL_IP_1>

Host kria-01
    Hostname <TAILSCALE_IP_1>
    User ubuntu

Match host kria-02 exec "nc -z -w1 <LOCAL_IP_2> 22 2>/dev/null"
    Hostname <LOCAL_IP_2>

Host kria-02
    Hostname <TAILSCALE_IP_2>
    User ubuntu
```

Then connect with:

```bash
ssh kria       # single board
ssh kria-01    # multi-board
```

## Adding Packages Later

Add new numbered scripts to `scripts/`:

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
