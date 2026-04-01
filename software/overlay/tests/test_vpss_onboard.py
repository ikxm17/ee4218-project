"""On-board tests for the VPSS Scaler driver.

Requires the PYNQ overlay to be loaded on a Kria board.
Uses session-scoped vpss_ip fixture from conftest.py.
"""

import pytest

from software.overlay.drivers.vpss import VpssScalerDriver
from software.overlay.tests.checks import check_register_readback

pytestmark = pytest.mark.onboard


def test_gpio_readback(vpss_ip):
    """Write GPIO_DATA=0x03 at _GPIO_BASE, read back and verify."""
    if vpss_ip is None:
        pytest.skip("v_proc_ss_0 not in overlay")
    gpio_offset = VpssScalerDriver._GPIO_BASE + VpssScalerDriver.GPIO_DATA
    check_register_readback(vpss_ip, gpio_offset, 0x03)


def test_hsc_width_in_readback(vpss_ip):
    """Write HSC_WIDTH_IN=1920 at _HSC_BASE, read back and verify."""
    if vpss_ip is None:
        pytest.skip("v_proc_ss_0 not in overlay")
    width_in_offset = VpssScalerDriver._HSC_BASE + VpssScalerDriver.HSC_WIDTH_IN
    check_register_readback(vpss_ip, width_in_offset, 1920)
