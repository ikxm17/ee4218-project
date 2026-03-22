# Kria KV260 Setup — Ubuntu 22.04

Environment setup for Kria KV260 boards. Configures static IP networking, base packages, and Tailscale for remote access.

## Prerequisites

- Xilinx Kria KV260 Vision AI Starter Kit
- Power supply (12V barrel jack)
- microSD card (32 GB+)
- Ethernet cable + router
- Host PC with SD card reader

## Host-side Setup

### Flash + Prep SD Card

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
| `--no-flash` | Skip download + flash (config only on an already-flashed card) | |
| `--image` | Use a local `.img.xz` file instead of downloading | `~/Downloads/kria.img.xz` |
| `--clean-cache` | Remove cached image after flashing | |

Each board gets a hostname derived from its board number: `kria-01`, `kria-02`, etc.

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

### SSH Config

Add the following to `~/.ssh/config` on your host machine for automatic local/Tailscale failover:

```ssh-config
Match host kria exec "nc -z -w1 <LOCAL_IP> 22 2>/dev/null"
    Hostname <LOCAL_IP>

Host kria
    Hostname <TAILSCALE_IP>
    User ubuntu
```

This probes the local IP first — if reachable, it connects directly. Otherwise it falls back to the Tailscale IP. Replace `<LOCAL_IP>` with the board's static IP (e.g. `192.168.1.101`) and `<TAILSCALE_IP>` with the IP shown after running `sudo tailscale up` on the board.

> The Tailscale fallback requires `tailscale up` on your host machine too (same Tailnet).

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

## On-board Setup

### Boot and SSH In

1. Insert the SD card into the KV260
2. Connect Ethernet and power
3. Wait ~60 seconds for boot
4. SSH in:

```bash
ssh ubuntu@<LOCAL_IP>   # password: ubuntu
```

On first login you will be prompted to change the default password.

### Run Setup

```bash
sudo bash setup.sh
```

Scripts follow the naming convention `<NN>-<name>.sh`, where `<NN>` is a two-digit number controlling execution order and `<name>` is the step name used for skip flags.

This runs all setup scripts in order:

| Script | Purpose |
|--------|---------|
| `00-preflight.sh` | Checks arch, OS, disk, network |
| `01-system-base.sh` | apt update/upgrade + essential packages |
| `02-tailscale.sh` | Installs Tailscale VPN |
| `03-pynq.sh` | PYNQ framework + shared Python venv at `/opt/ee4218/ee4218-venv` |
| `04-tflite.sh` | TFLite Runtime for software inference on A53 |
| `99-verify.sh` | Smoke tests |

**Skip a step** (by name or number):

```bash
sudo bash setup.sh --skip <name>
sudo bash setup.sh --skip <NN>
```

**Re-run a single step:**

```bash
sudo bash scripts/01-system-base.sh
```

### Tailscale Authentication

After setup completes, authenticate each board with Tailscale:

```bash
sudo tailscale up
```

Follow the printed URL to log in. Each board appears in your Tailscale network.

### Verify

```bash
sudo bash scripts/99-verify.sh
```

All checks should show `[PASS]`. Tailscale connection is `[INFO]` (passes if authenticated).

## Using the PYNQ Environment

PYNQ and dependencies are installed in a Python venv at `/opt/ee4218/ee4218-venv`. Board environment variables (`BOARD`, `XILINX_XRT`) and the PYNQ device tree overlay are set up automatically on interactive login via `/etc/profile.d/ee4218.sh`.

### Activate the venv

```bash
source /opt/ee4218/ee4218-venv/bin/activate
```

### Run PYNQ scripts

PYNQ requires root for `/dev/mem` access (MMIO) and FPGA programming. Use the full venv path with `sudo` (`sudo` resets `PATH`, so `sudo python3` would use system Python):

```bash
sudo /opt/ee4218/ee4218-venv/bin/python3 <script>.py
```

### Loading a bitstream (WIP)

> Repo structure for bitstreams not finalised yet. General pattern below.

> **TODO:** Once the project `.bit` + `.hwh` are available, add an overlay-load smoke test to `03-pynq.sh` to validate the full PYNQ → FPGA pipeline end-to-end. Current validation covers import and device enumeration only.

Export `.bit` + `.hwh` from Vivado (filenames must match), copy to the board:

```python
from pynq import Overlay, allocate
import numpy as np

ol = Overlay("<design>.bit", download=True)
ol.<ip_name>.mmio.write(0x0, 0x1)
buf = allocate(shape=(64,), dtype=np.uint32)
```

### Install additional packages

```bash
source /opt/ee4218/ee4218-venv/bin/activate
pip install <package>
```

### After a reboot

The PYNQ device tree overlay (`pynq.dtbo`) and ZOCL kernel module are re-loaded automatically on the first interactive login via `/etc/profile.d/ee4218.sh`. For non-interactive use:

```bash
sudo bash -c 'source /etc/profile.d/ee4218.sh && /opt/ee4218/ee4218-venv/bin/python3 <script>.py'
```

## Using TFLite Runtime

TFLite Runtime (`tflite-runtime==2.14.0`) is installed in the same shared venv. It provides the interpreter for running `.tflite` models on the A53 cores using XNNPACK with int8 NEON-optimized kernels.

### Run inference

```bash
sudo /opt/ee4218/ee4218-venv/bin/python3 <inference_script>.py
```

```python
import tflite_runtime.interpreter as tflite
import numpy as np

interpreter = tflite.Interpreter(model_path="<model>_int8.tflite", num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()
detections = interpreter.get_tensor(output_details[0]['index'])
```

## Adding Packages Later

The orchestrator (`setup.sh`) runs `scripts/[0-9]*.sh` in sorted order. Each script:
- Is self-contained and can be run independently
- Should use `set -euo pipefail`
- Can be skipped with `--skip <name>` or `--skip <NN>` (derived from filename)

```
scripts/03-pynq.sh
scripts/04-tflite.sh
```

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
│   ├── 03-pynq.sh                 # PYNQ framework + Python venv
│   ├── 04-tflite.sh               # TFLite Runtime
│   └── 99-verify.sh               # Smoke tests
└── config/
    └── netplan-static.yaml.tpl    # Netplan template
```
