"""On-board tests for the CSI-2 RX Subsystem.

Requires the PYNQ overlay to be loaded on a Kria board.
Uses session-scoped csi2_rx_mmio fixture from conftest.py.
"""

import pytest

from software.overlay.drivers.csi2_rx import Csi2RxDriver

pytestmark = pytest.mark.onboard


def test_core_config_readable(csi2_rx_mmio):
    """CORE_CONFIG (0x00) should be readable without exception."""
    val = csi2_rx_mmio.read(Csi2RxDriver.CORE_CONFIG)
    # Just verifying the read doesn't throw -- value can be anything.
    assert isinstance(val, int)


def test_protocol_config_max_lanes(csi2_rx_mmio):
    """PROTOCOL_CONFIG bits [4:3] = max_lanes - 1. IMX219 uses 2-lane, so expect 2."""
    proto_cfg = csi2_rx_mmio.read(Csi2RxDriver.PROTOCOL_CONFIG)
    max_lanes = ((proto_cfg >> 3) & 0x03) + 1
    assert max_lanes == 2, (
        f"Expected max_lanes=2 (IMX219 is 2-lane), "
        f"got {max_lanes} (PROTOCOL_CONFIG=0x{proto_cfg:08X})"
    )
