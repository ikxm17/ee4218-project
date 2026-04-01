"""Off-board tests for the CSI-2 RX Subsystem driver.

Tests Csi2RxDriver register decoding, status readback, and image info
parsing using MockIP (no hardware required).

CSI-2 RX is NOT an HLS IP -- it has a custom Xilinx register map (PG232).
"""

import pytest

from software.overlay.drivers.csi2_rx import Csi2RxDriver
from software.overlay.tests.checks import check_register_constants_sane
from software.overlay.tests.conftest import MockIP

pytestmark = pytest.mark.offboard


# -- Register constants -------------------------------------------------------

# Key offsets (all within 0x2000 = 8KB address space)
CORE_CONFIG = 0x00
PROTOCOL_CONFIG = 0x04
CORE_STATUS = 0x10
GLOBAL_IRQ_EN = 0x20
ISR = 0x24
IER = 0x28
VC_SEL = 0x2C
CLK_LANE_INFO = 0x3C
LANE0_INFO = 0x40
IMG_INFO1_VC0 = 0x60
IMG_INFO2_VC0 = 0x64

# ISR bits (all W1C)
_ISR_FRAME_RECEIVED = 1 << 31
_ISR_SOT_ERROR = 1 << 13
_ISR_CRC_ERROR = 1 << 9
_ISR_UNSUPPORTED_DT = 1 << 8


# -- Register sanity ----------------------------------------------------------

def test_register_offsets_within_address_space():
    """All CSI-2 RX register offsets must be within the 8KB address space."""
    offsets = {
        "CORE_CONFIG": Csi2RxDriver.CORE_CONFIG,
        "PROTOCOL_CONFIG": Csi2RxDriver.PROTOCOL_CONFIG,
        "CORE_STATUS": Csi2RxDriver.CORE_STATUS,
        "GLOBAL_IRQ_EN": Csi2RxDriver.GLOBAL_IRQ_EN,
        "ISR": Csi2RxDriver.ISR,
        "IER": Csi2RxDriver.IER,
        "VC_SEL": Csi2RxDriver.VC_SEL,
        "CLK_LANE_INFO": Csi2RxDriver.CLK_LANE_INFO,
        "LANE0_INFO": Csi2RxDriver.LANE0_INFO,
        "IMG_INFO1_VC0": Csi2RxDriver.IMG_INFO1_VC0,
        "IMG_INFO2_VC0": Csi2RxDriver.IMG_INFO2_VC0,
    }
    for name, off in offsets.items():
        assert off < 0x2000, f"{name}: offset 0x{off:X} >= 0x2000"


def test_register_offsets_aligned():
    """All CSI-2 RX register offsets must be 4-byte aligned."""
    offsets = {
        "CORE_CONFIG": Csi2RxDriver.CORE_CONFIG,
        "PROTOCOL_CONFIG": Csi2RxDriver.PROTOCOL_CONFIG,
        "CORE_STATUS": Csi2RxDriver.CORE_STATUS,
        "GLOBAL_IRQ_EN": Csi2RxDriver.GLOBAL_IRQ_EN,
        "ISR": Csi2RxDriver.ISR,
        "IER": Csi2RxDriver.IER,
        "VC_SEL": Csi2RxDriver.VC_SEL,
        "CLK_LANE_INFO": Csi2RxDriver.CLK_LANE_INFO,
        "LANE0_INFO": Csi2RxDriver.LANE0_INFO,
        "IMG_INFO1_VC0": Csi2RxDriver.IMG_INFO1_VC0,
        "IMG_INFO2_VC0": Csi2RxDriver.IMG_INFO2_VC0,
    }
    for name, off in offsets.items():
        assert off % 4 == 0, f"{name}: offset 0x{off:X} not 4-byte aligned"


# -- read_status --------------------------------------------------------------

def test_read_status_decodes_core_config():
    """CORE_CONFIG bit 0 = core enabled."""
    mock = MockIP({CORE_CONFIG: 0x01})
    drv = Csi2RxDriver(mock)
    status = drv.read_status()
    assert status["core_enabled"] is True


def test_read_status_active_lanes():
    """PROTOCOL_CONFIG bits [1:0] = active_lanes - 1. Value 1 -> 2 lanes."""
    mock = MockIP({PROTOCOL_CONFIG: 0x01})
    drv = Csi2RxDriver(mock)
    status = drv.read_status()
    assert status["active_lanes"] == 2


def test_read_status_max_lanes():
    """PROTOCOL_CONFIG bits [4:3] = max_lanes - 1. Value 0x08 -> bits [4:3]=1 -> 2 lanes."""
    mock = MockIP({PROTOCOL_CONFIG: 0x08})
    drv = Csi2RxDriver(mock)
    status = drv.read_status()
    assert status["max_lanes"] == 2


def test_read_status_packet_count():
    """CORE_STATUS bits [31:16] = packet count."""
    mock = MockIP({CORE_STATUS: 100 << 16})
    drv = Csi2RxDriver(mock)
    status = drv.read_status()
    assert status["packet_count"] == 100


# -- read_image_info ----------------------------------------------------------

def test_read_image_info():
    """IMG_INFO1 [31:16]=line_count, [15:0]=byte_count; IMG_INFO2 [5:0]=data_type."""
    mock = MockIP({
        IMG_INFO1_VC0: (480 << 16) | 2400,
        IMG_INFO2_VC0: 0x2B,
    })
    drv = Csi2RxDriver(mock)
    info = drv.read_image_info(vc=0)
    assert info["line_count"] == 480
    assert info["byte_count"] == 2400
    assert info["data_type"] == 0x2B


def test_read_image_info_vc_range_check():
    """read_image_info(vc=16) must raise ValueError (valid range 0-15)."""
    mock = MockIP()
    drv = Csi2RxDriver(mock)
    with pytest.raises(ValueError):
        drv.read_image_info(vc=16)


def test_read_image_info_vc_offset():
    """VC1 image info is at IMG_INFO1_VC0 + 8 = 0x68."""
    mock = MockIP({
        0x68: (240 << 16) | 1200,
        0x6C: 0x2B,
    })
    drv = Csi2RxDriver(mock)
    info = drv.read_image_info(vc=1)
    assert info["line_count"] == 240
    assert info["byte_count"] == 1200
    assert info["data_type"] == 0x2B
