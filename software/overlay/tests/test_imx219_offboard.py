"""Off-board tests for IMX219 register definitions.

Tests the register table constants and structure in _imx219_regs.py
(NOT the driver class, since it imports smbus2). No hardware required.
"""

import pytest

from software.overlay.drivers._imx219_regs import (
    INIT_TABLE_1080P30_RAW10_2LANE,
    REG_MODE_SELECT,
    REG_SOFTWARE_RESET,
    REG_MODEL_ID_H,
    REG_MODEL_ID_L,
    MODEL_ID_VALUE,
)

pytestmark = pytest.mark.offboard


# -- Table structure ----------------------------------------------------------

def test_init_table_is_list_of_pairs():
    """Each entry in INIT_TABLE is a tuple/list of length 2."""
    for i, entry in enumerate(INIT_TABLE_1080P30_RAW10_2LANE):
        assert len(entry) == 2, (
            f"Entry {i}: expected length 2, got {len(entry)}: {entry}"
        )


def test_register_addresses_16bit():
    """All register addresses in INIT_TABLE fit in 16 bits (0-0xFFFF)."""
    for i, (addr, _) in enumerate(INIT_TABLE_1080P30_RAW10_2LANE):
        assert 0 <= addr <= 0xFFFF, (
            f"Entry {i}: address 0x{addr:X} out of 16-bit range"
        )


def test_register_values_8bit():
    """All register values in INIT_TABLE fit in 8 bits (0-0xFF)."""
    for i, (_, val) in enumerate(INIT_TABLE_1080P30_RAW10_2LANE):
        assert 0 <= val <= 0xFF, (
            f"Entry {i}: value 0x{val:X} out of 8-bit range"
        )


def test_no_duplicate_consecutive_addresses():
    """No two consecutive entries write the same register with the same value.

    Note: the COMMON_INIT has (0x30EB, 0x05) then later (0x30EB, 0x0C) etc.
    These are intentional unlock sequences with different values.
    Only check for EXACT duplicates (same addr AND same value consecutively).
    """
    for i in range(1, len(INIT_TABLE_1080P30_RAW10_2LANE)):
        prev = INIT_TABLE_1080P30_RAW10_2LANE[i - 1]
        curr = INIT_TABLE_1080P30_RAW10_2LANE[i]
        assert not (prev[0] == curr[0] and prev[1] == curr[1]), (
            f"Entries {i-1} and {i} are exact duplicates: "
            f"(0x{curr[0]:04X}, 0x{curr[1]:02X})"
        )


# -- Well-known constants -----------------------------------------------------

def test_model_id_value():
    """MODEL_ID_VALUE is 0x0219 (IMX219 sensor ID)."""
    assert MODEL_ID_VALUE == 0x0219


def test_model_id_registers():
    """Model ID registers are at addresses 0x0000 (high) and 0x0001 (low)."""
    assert REG_MODEL_ID_H == 0x0000
    assert REG_MODEL_ID_L == 0x0001


def test_mode_select_register():
    """REG_MODE_SELECT is at address 0x0100."""
    assert REG_MODE_SELECT == 0x0100


def test_software_reset_register():
    """REG_SOFTWARE_RESET is at address 0x0103."""
    assert REG_SOFTWARE_RESET == 0x0103


def test_table_not_empty():
    """INIT_TABLE should have more than 50 entries (~100+ expected)."""
    assert len(INIT_TABLE_1080P30_RAW10_2LANE) > 50
