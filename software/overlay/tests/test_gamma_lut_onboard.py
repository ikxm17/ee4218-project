"""On-board tests for the Gamma LUT driver.

Requires the PYNQ overlay to be loaded on a Kria board.
Uses session-scoped fixtures from conftest.py.
"""

import pytest

from software.overlay.drivers.gamma_lut import GammaLutDriver
from software.overlay.tests.checks import check_register_readback

pytestmark = pytest.mark.onboard


def test_width_readback(gamma_lut_ip):
    """Write WIDTH=1920, read back and verify."""
    if gamma_lut_ip is None:
        pytest.skip("v_gamma_lut_0 not in overlay")
    check_register_readback(gamma_lut_ip, GammaLutDriver.WIDTH, 1920)


def test_video_format_readback(gamma_lut_ip):
    """Write VIDEO_FORMAT=0 (RGB), read back and verify."""
    if gamma_lut_ip is None:
        pytest.skip("v_gamma_lut_0 not in overlay")
    check_register_readback(gamma_lut_ip, GammaLutDriver.VIDEO_FORMAT, 0)


def test_ap_ctrl_responds(gamma_lut_ip):
    """Write AP_CTRL=0x81, read back, verify non-zero."""
    if gamma_lut_ip is None:
        pytest.skip("v_gamma_lut_0 not in overlay")
    gamma_lut_ip.write(0x00, 0x81)
    val = gamma_lut_ip.read(0x00)
    assert val != 0, f"AP_CTRL read back 0 after writing 0x81"
