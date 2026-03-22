#!/usr/bin/env bash
# Host-side SD card preparation for Kria KV260
# Downloads + flashes Ubuntu 22.04, then configures static IP, hostname,
# and SSH.
#
# Usage:
#   bash prep-sd.sh --device /dev/sdX --board-num <N> --gateway <GATEWAY_IP>
#
# Options:
#   --device       SD card block device (e.g., /dev/sdb)
#   --board-num    Board number (integer); IP = gateway_base + 100 + N
#   --gateway      Router/gateway IP address
#   --ssh-key      Path to public SSH key (default: ~/.ssh/id_*.pub)
#   --no-flash     Skip download + flash (config only on an already-flashed card)
#   --image        Path to a local .img.xz file (skip download, still flash)
#   --clean-cache  Remove cached image after flashing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/config/netplan-static.yaml.tpl"

# --- Image download constants ---
IMAGE_URL="https://people.canonical.com/~platform/images/xilinx/kria-ubuntu-22.04/iot-limerick-kria-classic-desktop-2204-20240304-165.img.xz"
IMAGE_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/kria-setup"
IMAGE_FILENAME="$(basename "$IMAGE_URL")"

# --- Defaults ---
DEVICE=""
BOARD_NUM=""
GATEWAY=""
SSH_KEY=""
NO_FLASH=false
LOCAL_IMAGE=""
CLEAN_CACHE=false
HOSTNAME_PREFIX="kria"

usage() {
    cat <<EOF
Usage: bash prep-sd.sh --device /dev/sdX --board-num <N> --gateway <GATEWAY_IP> [options]

Options:
  --device        SD card block device (e.g., /dev/sdb)
  --board-num     Board number (integer); IP = gateway_base + 100 + N
  --gateway       Router/gateway IP address
  --ssh-key       Path to SSH public key (default: auto-detect ~/.ssh/id_*.pub)
  --no-flash      Skip download + flash (config only on an already-flashed card)
  --image PATH    Use a local .img.xz file instead of downloading
  --clean-cache   Remove cached image after flashing
  --help          Show this help message
EOF
    exit 0
}

# --- Parse arguments ---
needs_value() {
    if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: $1 requires a value."
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)      needs_value "$@"; DEVICE="$2";      shift 2 ;;
        --board-num)   needs_value "$@"; BOARD_NUM="$2";   shift 2 ;;
        --gateway)     needs_value "$@"; GATEWAY="$2";     shift 2 ;;
        --ssh-key)     needs_value "$@"; SSH_KEY="$2";     shift 2 ;;
        --no-flash)    NO_FLASH=true;    shift ;;
        --image)       needs_value "$@"; LOCAL_IMAGE="$2"; shift 2 ;;
        --clean-cache) CLEAN_CACHE=true; shift ;;
        --help)        usage ;;
        *)
            echo "Error: unknown argument: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# --- Validate required args ---
MISSING=()
[ -z "$DEVICE" ]    && MISSING+=("--device")
[ -z "$BOARD_NUM" ] && MISSING+=("--board-num")
[ -z "$GATEWAY" ]   && MISSING+=("--gateway")
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Error: missing required argument(s): ${MISSING[*]}"
    echo "Run with --help for usage."
    exit 1
fi

# --- Normalize partition path to base device ---
# /dev/sda1 → /dev/sda, /dev/nvme0n1p2 → /dev/nvme0n1
if [[ "$DEVICE" =~ ^(/dev/[a-z]+)[0-9]+$ ]]; then
    BASE="${BASH_REMATCH[1]}"
elif [[ "$DEVICE" =~ ^(/dev/[a-z0-9]+)p[0-9]+$ ]]; then
    BASE="${BASH_REMATCH[1]}"
else
    BASE="$DEVICE"
fi

if [ "$BASE" != "$DEVICE" ]; then
    echo "Note: $DEVICE looks like a partition; using $BASE instead."
    DEVICE="$BASE"
fi

# --- Validate --image path ---
if [ -n "$LOCAL_IMAGE" ] && [ ! -f "$LOCAL_IMAGE" ]; then
    echo "Error: image file not found: $LOCAL_IMAGE"
    exit 1
fi

# --- Validate device exists ---
DEVICE_BASE="$(basename "$DEVICE")"
if [ ! -b "$DEVICE" ]; then
    echo "Error: $DEVICE is not a block device (does it exist?)."
    echo "Plug in your SD card and check with: lsblk"
    exit 1
fi

# --- Safety: refuse devices with mounted filesystems ---
if ! MOUNTED_PARTS="$(lsblk -ln -o MOUNTPOINT "$DEVICE" | awk 'NF')"; then
    echo "Error: failed to query mount status for $DEVICE."
    exit 1
fi
if [ -n "$MOUNTED_PARTS" ]; then
    echo "Error: $DEVICE has mounted filesystems:"
    lsblk -o NAME,SIZE,MOUNTPOINT "$DEVICE"
    echo ""
    echo "Refusing to flash a device with active mounts (could be your system drive)."
    echo "If this is really your SD card, unmount with:"
    echo "  sudo umount ${DEVICE}*"
    echo ""
    echo "Note: Don't use the file explorer eject — that powers off the device"
    echo "and you'll need to re-plug it."
    exit 1
fi

# --- Safety: refuse non-removable devices ---
REMOVABLE_PATH="/sys/block/$DEVICE_BASE/removable"

if [ ! -f "$REMOVABLE_PATH" ]; then
    echo "Error: cannot determine if $DEVICE is removable (no sysfs entry at $REMOVABLE_PATH)."
    echo "Make sure you specified the correct block device (e.g., /dev/sdb)."
    exit 1
fi

if [ "$(cat "$REMOVABLE_PATH")" != "1" ]; then
    echo "WARNING: $DEVICE does not appear to be a removable device."
    echo "This safety check prevents accidental writes to your system drive."
    read -rp "Are you SURE you want to continue? Type 'yes' to proceed: " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Aborted."
        exit 1
    fi
fi

# --- Compute board IP ---
GATEWAY_BASE="${GATEWAY%.*}"
BOARD_IP="${GATEWAY_BASE}.$(( 100 + BOARD_NUM ))"

# --- Compute hostname ---
HOSTNAME=$(printf '%s-%02d' "$HOSTNAME_PREFIX" "$BOARD_NUM")

echo ""
echo "=== SD Card Prep ==="
echo "  Device:     $DEVICE"
echo "  Board #:    $BOARD_NUM"
echo "  Hostname:   $HOSTNAME"
echo "  Gateway:    $GATEWAY"
echo "  Board IP:   $BOARD_IP"
echo "  Password:   ubuntu (change on first login)"
if [ "$NO_FLASH" = true ]; then
    echo "  Flash:      skipped (--no-flash)"
elif [ -n "$LOCAL_IMAGE" ]; then
    echo "  Image:      $LOCAL_IMAGE (local)"
else
    echo "  Image:      $IMAGE_FILENAME (download/cache)"
fi
echo ""

# --- Download + Flash ---
if [ "$NO_FLASH" = true ]; then
    # Config-only mode: softer confirmation
    read -rp "This will modify the rootfs on $DEVICE. Continue? [y/N] " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
else
    # Determine image path
    if [ -n "$LOCAL_IMAGE" ]; then
        IMAGE_PATH="$LOCAL_IMAGE"
    else
        # Download image (with resume support)
        mkdir -p "$IMAGE_CACHE_DIR"
        IMAGE_PATH="$IMAGE_CACHE_DIR/$IMAGE_FILENAME"
        if [ -f "$IMAGE_PATH" ]; then
            echo "Using cached image: $IMAGE_PATH"
        else
            echo "Downloading Ubuntu 22.04 image for Kria..."
            echo "  URL: $IMAGE_URL"
            echo "  Destination: $IMAGE_PATH"
            echo ""
            wget -c -O "$IMAGE_PATH" "$IMAGE_URL"
            echo ""
        fi
    fi

    IMAGE_SIZE="$(du -h "$IMAGE_PATH" | cut -f1)"
    echo "Image: $IMAGE_PATH ($IMAGE_SIZE)"
    echo ""

    # Destructive confirmation
    read -rp "This will ERASE $DEVICE and flash Ubuntu 22.04. Continue? [y/N] " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi

    echo ""
    echo "Flashing $IMAGE_PATH → $DEVICE ..."
    xzcat "$IMAGE_PATH" | sudo dd of="$DEVICE" bs=4M status=progress conv=fsync
    sudo sync
    sudo partprobe "$DEVICE"
    echo "Flash complete."

    if [ "$CLEAN_CACHE" = true ] && [ -z "$LOCAL_IMAGE" ]; then
        echo "Cleaning cached image..."
        rm -f "$IMAGE_CACHE_DIR/$IMAGE_FILENAME"
        rmdir "$IMAGE_CACHE_DIR" 2>/dev/null || true
    fi
    echo ""
fi

# --- Mount rootfs (partition 2) ---
MOUNT_DIR="$(mktemp -d /tmp/kria-rootfs.XXXXXX)"
PARTITION="${DEVICE}2"

# Handle NVMe-style partition naming (e.g., /dev/nvme0n1p2)
if [[ "$DEVICE" =~ [0-9]$ ]]; then
    PARTITION="${DEVICE}p2"
fi

echo "Mounting $PARTITION to $MOUNT_DIR..."
sudo mount "$PARTITION" "$MOUNT_DIR"

cleanup() {
    echo "Unmounting $MOUNT_DIR..."
    sudo umount "$MOUNT_DIR" 2>/dev/null || true
    rmdir "$MOUNT_DIR" 2>/dev/null || true
}
trap cleanup EXIT

# --- 1. Static IP via netplan ---
echo "Configuring static IP..."
NETPLAN_DIR="$MOUNT_DIR/etc/netplan"
sudo mkdir -p "$NETPLAN_DIR"

# Remove default cloud-init netplan config to avoid conflicts
sudo rm -f "$NETPLAN_DIR/50-cloud-init.yaml"

sed -e "s|__IP_ADDRESS__|$BOARD_IP|g" \
    -e "s|__GATEWAY__|$GATEWAY|g" \
    "$TEMPLATE" | sudo tee "$NETPLAN_DIR/01-static.yaml" > /dev/null

sudo chmod 600 "$NETPLAN_DIR/01-static.yaml"

# --- 2. Disable cloud-init network config ---
echo "Disabling cloud-init network management..."
CLOUD_CFG_DIR="$MOUNT_DIR/etc/cloud/cloud.cfg.d"
sudo mkdir -p "$CLOUD_CFG_DIR"
echo "network: {config: disabled}" | sudo tee "$CLOUD_CFG_DIR/99-disable-network-config.cfg" > /dev/null

# --- 3. Hostname ---
echo "Setting hostname to $HOSTNAME..."
echo "$HOSTNAME" | sudo tee "$MOUNT_DIR/etc/hostname" > /dev/null
cat <<EOF | sudo tee "$MOUNT_DIR/etc/hosts" > /dev/null
127.0.0.1 localhost
127.0.1.1 $HOSTNAME

# The following lines are desirable for IPv6 capable hosts
::1 ip6-localhost ip6-loopback
fe00::0 ip6-localnet
ff00::0 ip6-mcastprefix
ff02::1 ip6-allnodes
ff02::2 ip6-allrouters
ff02::3 ip6-allhosts
EOF

# --- 4. Cloud-init: preserve hostname ---
echo "Configuring cloud-init (hostname)..."
cat <<EOF | sudo tee "$CLOUD_CFG_DIR/99-kria-setup.cfg" > /dev/null
preserve_hostname: true
EOF

# --- 5. SSH key ---
echo "Setting up SSH authorized key..."
SSH_DIR="$MOUNT_DIR/home/ubuntu/.ssh"
sudo mkdir -p "$SSH_DIR"

if [ -n "$SSH_KEY" ]; then
    # Use specified key
    if [ ! -f "$SSH_KEY" ]; then
        echo "Error: SSH key not found at $SSH_KEY"
        exit 1
    fi
    sudo cp "$SSH_KEY" "$SSH_DIR/authorized_keys"
else
    # Auto-detect from ~/.ssh/id_*.pub
    KEY_FILES=( "$HOME"/.ssh/id_*.pub )
    if [ ${#KEY_FILES[@]} -eq 0 ] || [ ! -f "${KEY_FILES[0]}" ]; then
        echo "Warning: no SSH public key found in $HOME/.ssh/"
        echo "  You can add one later or re-run with --ssh-key <path>"
    else
        cat "${KEY_FILES[@]}" | sudo tee "$SSH_DIR/authorized_keys" > /dev/null
        echo "  Added ${#KEY_FILES[@]} key(s) from $HOME/.ssh/"
    fi
fi

# Set ownership — the image ships /home/ubuntu as root-owned and
# cloud-init does not fix it on first boot.
sudo chown 1000:1000 "$MOUNT_DIR/home/ubuntu"
sudo chown -R 1000:1000 "$SSH_DIR"
sudo chmod 700 "$SSH_DIR"
sudo chmod 600 "$SSH_DIR/authorized_keys" 2>/dev/null || true

echo ""
echo "=== Done ==="
echo ""
echo "Summary:"
echo "  Board #$BOARD_NUM → IP: $BOARD_IP"
echo "  Hostname: $HOSTNAME"
echo "  Gateway: $GATEWAY"
echo "  Credentials: ubuntu / ubuntu (change on first login)"
if [ "$NO_FLASH" = false ] && [ -z "$LOCAL_IMAGE" ]; then
    echo "  Cached image: $IMAGE_CACHE_DIR/$IMAGE_FILENAME"
fi
echo ""
echo "After inserting the SD card and powering on the board (~60s boot):"
echo "  ssh ubuntu@$BOARD_IP"
echo ""
echo "Then run the on-board setup:"
echo "  sudo bash setup/setup.sh"
