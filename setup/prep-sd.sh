#!/usr/bin/env bash
# Host-side SD card preparation for Kria KV260
# Run on your PC after flashing the Ubuntu 22.04 image to the SD card.
#
# Usage:
#   sudo bash prep-sd.sh --device /dev/sdX --board-num <N> --gateway <GATEWAY_IP>
#
# Options:
#   --device    SD card block device (e.g., /dev/sdb)
#   --board-num Board number (integer); IP = gateway_base + 100 + N
#   --gateway   Router/gateway IP address
#   --ssh-key   Path to public SSH key (default: ~/.ssh/id_*.pub)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/config/netplan-static.yaml.tpl"

# --- Defaults ---
DEVICE=""
BOARD_NUM=""
GATEWAY=""
SSH_KEY=""

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)   DEVICE="$2";    shift 2 ;;
        --board-num) BOARD_NUM="$2"; shift 2 ;;
        --gateway)  GATEWAY="$2";   shift 2 ;;
        --ssh-key)  SSH_KEY="$2";   shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: sudo bash prep-sd.sh --device /dev/sdX --board-num <N> --gateway <GATEWAY_IP> [--ssh-key <path>]"
            exit 1
            ;;
    esac
done

# --- Validate required args ---
if [ -z "$DEVICE" ] || [ -z "$BOARD_NUM" ] || [ -z "$GATEWAY" ]; then
    echo "Error: --device, --board-num, and --gateway are all required."
    echo "Usage: sudo bash prep-sd.sh --device /dev/sdX --board-num <N> --gateway <GATEWAY_IP>"
    exit 1
fi

# --- Root check ---
if [ "$(id -u)" -ne 0 ]; then
    echo "Error: this script must be run as root (sudo)."
    exit 1
fi

# --- Safety: refuse non-removable devices ---
DEVICE_BASE="$(basename "$DEVICE")"
# Strip partition suffix to get base device name
DEVICE_BASE="${DEVICE_BASE%%[0-9]*}"
REMOVABLE_PATH="/sys/block/$DEVICE_BASE/removable"

if [ ! -f "$REMOVABLE_PATH" ]; then
    echo "Error: cannot determine if $DEVICE is removable (no sysfs entry at $REMOVABLE_PATH)."
    echo "Make sure you specified the correct device (e.g., /dev/sdb, not /dev/sdb1)."
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

echo ""
echo "=== SD Card Prep ==="
echo "  Device:     $DEVICE"
echo "  Board #:    $BOARD_NUM"
echo "  Gateway:    $GATEWAY"
echo "  Board IP:   $BOARD_IP"
echo ""

# --- Confirm before proceeding ---
read -rp "This will modify the rootfs on $DEVICE. Continue? [y/N] " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# --- Mount rootfs (partition 2) ---
MOUNT_DIR="$(mktemp -d /tmp/kria-rootfs.XXXXXX)"
PARTITION="${DEVICE}2"

# Handle NVMe-style partition naming (e.g., /dev/nvme0n1p2)
if [[ "$DEVICE" =~ [0-9]$ ]]; then
    PARTITION="${DEVICE}p2"
fi

echo "Mounting $PARTITION to $MOUNT_DIR..."
mount "$PARTITION" "$MOUNT_DIR"

cleanup() {
    echo "Unmounting $MOUNT_DIR..."
    umount "$MOUNT_DIR" 2>/dev/null || true
    rmdir "$MOUNT_DIR" 2>/dev/null || true
}
trap cleanup EXIT

# --- 1. Static IP via netplan ---
echo "Configuring static IP..."
NETPLAN_DIR="$MOUNT_DIR/etc/netplan"
mkdir -p "$NETPLAN_DIR"

sed -e "s|__IP_ADDRESS__|$BOARD_IP|g" \
    -e "s|__GATEWAY__|$GATEWAY|g" \
    "$TEMPLATE" > "$NETPLAN_DIR/01-static.yaml"

chmod 600 "$NETPLAN_DIR/01-static.yaml"

# --- 2. Disable cloud-init network config ---
echo "Disabling cloud-init network management..."
CLOUD_CFG_DIR="$MOUNT_DIR/etc/cloud/cloud.cfg.d"
mkdir -p "$CLOUD_CFG_DIR"
echo "network: {config: disabled}" > "$CLOUD_CFG_DIR/99-disable-network-config.cfg"

# --- 3. SSH key ---
echo "Setting up SSH authorized key..."
SSH_DIR="$MOUNT_DIR/home/ubuntu/.ssh"
mkdir -p "$SSH_DIR"

if [ -n "$SSH_KEY" ]; then
    # Use specified key
    if [ ! -f "$SSH_KEY" ]; then
        echo "Error: SSH key not found at $SSH_KEY"
        exit 1
    fi
    cat "$SSH_KEY" > "$SSH_DIR/authorized_keys"
else
    # Auto-detect from ~/.ssh/id_*.pub (use the invoking user's home)
    REAL_HOME="$(eval echo "~${SUDO_USER:-$USER}")"
    KEY_FILES=( "$REAL_HOME"/.ssh/id_*.pub )
    if [ ${#KEY_FILES[@]} -eq 0 ] || [ ! -f "${KEY_FILES[0]}" ]; then
        echo "Warning: no SSH public key found in $REAL_HOME/.ssh/"
        echo "  You can add one later or re-run with --ssh-key <path>"
    else
        cat "${KEY_FILES[@]}" > "$SSH_DIR/authorized_keys"
        echo "  Added ${#KEY_FILES[@]} key(s) from $REAL_HOME/.ssh/"
    fi
fi

# Set ownership to ubuntu user (UID/GID 1000 is default for first user)
chown -R 1000:1000 "$SSH_DIR"
chmod 700 "$SSH_DIR"
chmod 600 "$SSH_DIR/authorized_keys" 2>/dev/null || true

echo ""
echo "=== Done ==="
echo ""
echo "Summary:"
echo "  Board #$BOARD_NUM → IP: $BOARD_IP"
echo "  Gateway: $GATEWAY"
echo ""
echo "After inserting the SD card and powering on the board (~60s boot):"
echo "  ssh ubuntu@$BOARD_IP"
echo ""
echo "Default password (first login): ubuntu"
echo "Then run the on-board setup:"
echo "  sudo bash setup/setup.sh"
