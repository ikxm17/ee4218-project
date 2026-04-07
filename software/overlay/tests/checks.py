"""Shared verification primitives for driver tests and pipeline diagnostics.

Used by both off-board and on-board tests, and by /pipeline-debug scripts.
"""

import pathlib

import numpy as np

from software.overlay.drivers import TinyissimoYoloAcceleratorDriver


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


def find_accel_bitstream(hw_dir: str) -> str | None:
    """Find a .bit in `hw_dir` whose sibling .hwh references the accelerator.

    Reuses the same .hwh XML parser the audit system uses
    (`software.overlay.drivers._parse_hwh`) so a stale comment or a
    parameter-value string referencing the VLNV doesn't produce a
    false positive. Returns None if no match, so callers can
    `pytest.skip` cleanly.

    Args:
        hw_dir: Directory containing .bit and .hwh pairs (e.g.
            `hardware/output/`).

    Returns:
        Absolute path to the matching .bit, or None.
    """
    from software.overlay.drivers import _parse_hwh

    target_vlnv = TinyissimoYoloAcceleratorDriver.IP_VLNV
    hw_path = pathlib.Path(hw_dir)
    for bit in sorted(hw_path.glob("*.bit")):
        hwh = bit.with_suffix(".hwh")
        if not hwh.exists():
            continue
        try:
            modules = _parse_hwh(str(hwh))
        except Exception:
            continue
        if any(info["vlnv"] == target_vlnv for info in modules.values()):
            return str(bit)
    return None


def load_golden_uram_mem(path: str, num_words: int) -> np.ndarray:
    """Parse a golden_layer*_uram.mem file into a (num_words, 16) int8 array.

    The .mem files used by the inference_hdl testbench are 128-bit
    URAM-packed words, one 32-hex-character word per line. The on-wire
    byte order is LSB-first per axil_regs.sv:255 (lane = `rd_addr[3:2]`),
    so we extract lane `i` as bits `[i*8 +: 8]` of each line.

    Each lane byte is interpreted as signed int8 (two's complement),
    matching how the HDL accelerator stores int8 activations and how
    `TinyissimoYoloAcceleratorDriver.read_results_raw()` reconstructs
    them on the PS side.

    Args:
        path: Path to the .mem file (e.g.
            `hardware/testbench/inference_hdl/golden_layer16_uram.mem`).
        num_words: Number of 128-bit words to read. Files may be longer
            than `num_words`; surplus lines are ignored. Layer 13 stores
            256 cv2 words, layer 16 stores 64 cv3 words.

    Returns:
        np.int8 array of shape (num_words, 16).
    """
    out = np.zeros((num_words, 16), dtype=np.int8)
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= num_words:
                break
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            word = int(line, 16)
            for lane in range(16):
                byte = (word >> (lane * 8)) & 0xFF
                out[i, lane] = np.int8(byte if byte < 128 else byte - 256)
    return out
