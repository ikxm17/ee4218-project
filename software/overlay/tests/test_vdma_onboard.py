"""On-board tests for the VDMA driver.

Requires the PYNQ overlay to be loaded on a Kria board.
Uses session-scoped fixtures from conftest.py.
"""

import time

import pytest

from software.overlay.drivers.vdma import VdmaDriver

pytestmark = pytest.mark.onboard


def test_version_register(vdma0_ip):
    """VDMA_VERSION (0x2C) should be non-zero (IP version stamp)."""
    if vdma0_ip is None:
        pytest.skip("axi_vdma_0 not in overlay")
    version = vdma0_ip.read(VdmaDriver.VDMA_VERSION)
    assert version != 0, "VDMA_VERSION register reads as 0"


def test_frmstore_readback(vdma0_ip):
    """Write S2MM_FRMSTORE=3, read back and verify."""
    if vdma0_ip is None:
        pytest.skip("axi_vdma_0 not in overlay")
    vdma0_ip.write(VdmaDriver.S2MM_FRMSTORE, 3)
    readback = vdma0_ip.read(VdmaDriver.S2MM_FRMSTORE)
    assert readback == 3, (
        f"S2MM_FRMSTORE: wrote 3, read back {readback}"
    )


def test_reset_self_clears(vdma0_ip):
    """Write DMACR_RESET to S2MM_DMACR, poll until reset bit clears."""
    if vdma0_ip is None:
        pytest.skip("axi_vdma_0 not in overlay")
    vdma0_ip.write(VdmaDriver.S2MM_DMACR, VdmaDriver.DMACR_RESET)

    deadline = time.monotonic() + 1.0  # 1 second timeout
    while time.monotonic() < deadline:
        val = vdma0_ip.read(VdmaDriver.S2MM_DMACR)
        if not (val & VdmaDriver.DMACR_RESET):
            return  # reset bit cleared -- pass
        time.sleep(0.001)

    pytest.fail(
        f"DMACR reset bit did not self-clear within 1s "
        f"(S2MM_DMACR=0x{vdma0_ip.read(VdmaDriver.S2MM_DMACR):08X})"
    )
