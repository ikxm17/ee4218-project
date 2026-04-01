"""On-board tests for the Demosaic driver.

Requires the PYNQ overlay to be loaded on a Kria board.
Uses session-scoped fixtures from conftest.py.
"""

import pytest

from software.overlay.drivers.demosaic import DemosaicDriver
from software.overlay.tests.checks import check_register_readback, check_ap_ctrl_running

pytestmark = pytest.mark.onboard


def test_width_readback(demosaic_ip):
    """Write WIDTH=1920 via register offset, read back and verify."""
    if demosaic_ip is None:
        pytest.skip("v_demosaic_0 not in overlay")
    check_register_readback(demosaic_ip, DemosaicDriver.WIDTH, 1920)


def test_height_readback(demosaic_ip):
    """Write HEIGHT=1080 via register offset, read back and verify."""
    if demosaic_ip is None:
        pytest.skip("v_demosaic_0 not in overlay")
    check_register_readback(demosaic_ip, DemosaicDriver.HEIGHT, 1080)


def test_ap_ctrl_after_start(demosaic_ip):
    """After start(), AP_CTRL should indicate running."""
    if demosaic_ip is None:
        pytest.skip("v_demosaic_0 not in overlay")
    drv = DemosaicDriver(demosaic_ip)
    drv.start()
    check_ap_ctrl_running(demosaic_ip)
