"""Shared verification primitives for driver tests and pipeline diagnostics.

Used by both off-board and on-board tests, and by /pipeline-debug scripts.
"""


def check_register_readback(ip, offset: int, value: int) -> int:
    """Write a value to a register and read it back.

    Returns the readback value. Raises AssertionError on mismatch.
    """
    ip.write(offset, value)
    readback = ip.read(offset)
    assert readback == value, (
        f"Register 0x{offset:04X}: wrote 0x{value:08X}, "
        f"read back 0x{readback:08X}"
    )
    return readback


def check_ap_ctrl_running(ip, ap_ctrl_offset: int = 0x00) -> dict:
    """Verify an HLS IP reports running after start.

    Returns decoded AP_CTRL bits.
    """
    val = ip.read(ap_ctrl_offset)
    idle = bool(val & 0x04)
    result = {
        "started": bool(val & 0x01),
        "auto_restart": bool(val & 0x80),
        "idle": idle,
        "raw": val,
    }
    assert result["started"] or result["auto_restart"], (
        f"AP_CTRL at 0x{ap_ctrl_offset:04X} = 0x{val:02X}: "
        f"IP not started (no START or AUTO_RESTART bit)"
    )
    return result


def check_register_constants_sane(
    offsets: dict[str, int],
    addr_space: int,
    alignment: int = 4,
) -> None:
    """Verify register offset constants are sane.

    Checks: non-negative, aligned, within address space, no overlaps.

    Args:
        offsets: {name: offset} mapping.
        addr_space: Maximum address space size in bytes.
        alignment: Required byte alignment (default 4).
    """
    seen = {}
    for name, off in offsets.items():
        assert off >= 0, f"{name}: offset 0x{off:X} is negative"
        assert off % alignment == 0, (
            f"{name}: offset 0x{off:X} not {alignment}-byte aligned"
        )
        assert off < addr_space, (
            f"{name}: offset 0x{off:X} >= address space 0x{addr_space:X}"
        )
        if off in seen:
            assert False, (
                f"{name} and {seen[off]} share offset 0x{off:X}"
            )
        seen[off] = name
