"""Off-board tests for the Gamma LUT driver.

Tests GammaLutDriver register layout, LUT packing, bypass toggle,
and configure() modes using MockIP (no hardware required).
"""

import pytest

from software.overlay.drivers.gamma_lut import GammaLutDriver
from software.overlay.tests.checks import check_register_constants_sane
from software.overlay.tests.conftest import MockIP

pytestmark = pytest.mark.offboard


# -- Register sanity ----------------------------------------------------------

def test_register_offsets_sane():
    """WIDTH, HEIGHT, VIDEO_FORMAT are aligned, unique, within 0x2000."""
    check_register_constants_sane(
        offsets={
            "WIDTH": GammaLutDriver.WIDTH,
            "HEIGHT": GammaLutDriver.HEIGHT,
            "VIDEO_FORMAT": GammaLutDriver.VIDEO_FORMAT,
        },
        addr_space=0x2000,
    )


def test_lut_regions_dont_overlap_control():
    """LUT_R starts well above the last control register."""
    assert GammaLutDriver.LUT_R > GammaLutDriver.VIDEO_FORMAT


def test_lut_regions_sequential():
    """R, G, B LUT regions are 0x800 apart and in order."""
    assert GammaLutDriver.LUT_R == 0x0800
    assert GammaLutDriver.LUT_G == 0x1000
    assert GammaLutDriver.LUT_B == 0x1800
    assert GammaLutDriver.LUT_G - GammaLutDriver.LUT_R == 0x0800
    assert GammaLutDriver.LUT_B - GammaLutDriver.LUT_G == 0x0800


# -- load_lut -----------------------------------------------------------------

def test_load_lut_packing():
    """LUT entries are packed 2 per word: bits[15:0]=even, bits[31:16]=odd."""
    mock = MockIP()
    drv = GammaLutDriver(mock)

    r = list(range(GammaLutDriver.LUT_ENTRIES))
    g = list(range(GammaLutDriver.LUT_ENTRIES))
    b = list(range(GammaLutDriver.LUT_ENTRIES))
    drv.load_lut(r, g, b)

    # First word at LUT_R: entries 0 and 1 → (1 << 16) | 0
    assert mock.read(0x0800) == (1 << 16) | 0  # 0x00010000
    # Second word at LUT_R+4: entries 2 and 3 → (3 << 16) | 2
    assert mock.read(0x0804) == (3 << 16) | 2  # 0x00030002


def test_load_lut_wrong_length_raises():
    """load_lut raises ValueError when any channel has wrong length."""
    mock = MockIP()
    drv = GammaLutDriver(mock)
    with pytest.raises(ValueError):
        drv.load_lut(
            r=[0] * 100,
            g=[0] * GammaLutDriver.LUT_ENTRIES,
            b=[0] * GammaLutDriver.LUT_ENTRIES,
        )


# -- configure ----------------------------------------------------------------

def test_configure_bypass_clears_ap_ctrl():
    """configure(bypass=True) writes 0x00 to AP_CTRL."""
    mock = MockIP()
    drv = GammaLutDriver(mock)
    drv.configure(bypass=True)
    assert mock.read(0x00) == 0x00


def test_configure_active_starts_ip():
    """configure(bypass=False) writes 0x81 to AP_CTRL."""
    mock = MockIP()
    drv = GammaLutDriver(mock)
    drv.configure(bypass=False)
    assert mock.read(0x00) == 0x81


# -- set_bypass ---------------------------------------------------------------

def test_set_bypass_toggle():
    """set_bypass toggles AP_CTRL between 0x00 and 0x81."""
    mock = MockIP()
    drv = GammaLutDriver(mock)

    drv.set_bypass(True)
    assert mock.read(0x00) == 0x00

    drv.set_bypass(False)
    assert mock.read(0x00) == 0x81
