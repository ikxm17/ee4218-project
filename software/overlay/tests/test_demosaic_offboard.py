"""Off-board tests for the Demosaic driver.

Tests DemosaicDriver register writes, start/stop, and status readback
using MockIP (no hardware required).
"""

import pytest

from software.overlay.drivers.demosaic import DemosaicDriver
from software.overlay.tests.checks import check_register_constants_sane
from software.overlay.tests.conftest import MockIP

pytestmark = pytest.mark.offboard


# -- Register sanity ----------------------------------------------------------

def test_register_offsets_sane():
    """WIDTH, HEIGHT, BAYER_PHASE are aligned, unique, within 0x40."""
    check_register_constants_sane(
        offsets={
            "WIDTH": DemosaicDriver.WIDTH,
            "HEIGHT": DemosaicDriver.HEIGHT,
            "BAYER_PHASE": DemosaicDriver.BAYER_PHASE,
        },
        addr_space=0x40,
    )


# -- configure ----------------------------------------------------------------

def test_configure_writes_correct_registers():
    """configure() writes width, height, bayer_phase and starts the IP."""
    mock = MockIP()
    drv = DemosaicDriver(mock)
    drv.configure(width=1920, height=1080, bayer_phase=0)
    assert mock.read(0x10) == 1920
    assert mock.read(0x18) == 1080
    assert mock.read(0x28) == 0
    assert mock.read(0x00) == 0x81


def test_configure_custom_bayer_phase():
    """configure() writes the requested bayer_phase value."""
    mock = MockIP()
    drv = DemosaicDriver(mock)
    drv.configure(width=1920, height=1080, bayer_phase=3)
    assert mock.read(0x28) == 3


# -- start / stop ------------------------------------------------------------

def test_start_writes_ap_ctrl():
    """start() writes 0x81 to AP_CTRL."""
    mock = MockIP()
    drv = DemosaicDriver(mock)
    drv.start()
    assert mock.read(0x00) == 0x81


def test_stop_clears_ap_ctrl():
    """stop() writes 0x00 to AP_CTRL."""
    mock = MockIP()
    drv = DemosaicDriver(mock)
    drv.start()
    drv.stop()
    assert mock.read(0x00) == 0x00


# -- read_status --------------------------------------------------------------

def test_read_status_returns_dict():
    """read_status() returns dict with 'idle' key when IDLE bit set."""
    mock = MockIP({0x00: 0x04})  # AP_CTRL_IDLE
    drv = DemosaicDriver(mock)
    status = drv.read_status()
    assert isinstance(status, dict)
    assert status["idle"] is True
