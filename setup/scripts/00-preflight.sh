#!/usr/bin/env bash
# Pre-flight checks — fail fast on unsupported environments
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
FAIL=0

check() {
    local label="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "[PASS] $label"
    else
        echo "[FAIL] $label"
        FAIL=1
    fi
}

echo "=== Pre-flight checks ==="

# Architecture
check "Architecture is aarch64" test "$(uname -m)" = "aarch64"

# Ubuntu 22.04
check "Ubuntu 22.04 detected" bash -c '
    . /etc/os-release
    [ "$ID" = "ubuntu" ] && [[ "$VERSION_ID" == 22.04* ]]
'

# Disk space (at least 4 GB free on /)
FREE_KB=$(df --output=avail / | tail -1 | tr -d ' ')
FREE_GB=$(( FREE_KB / 1048576 ))
check "At least 4 GB free disk (${FREE_GB} GB available)" test "$FREE_GB" -ge 4

# Network connectivity
check "Network connectivity (ping 1.1.1.1)" ping -c1 -W5 1.1.1.1

echo ""
if [ "$FAIL" -ne 0 ]; then
    echo "Pre-flight checks FAILED. Fix issues above before continuing."
    exit 1
fi
echo "All pre-flight checks passed."
