"""Off-board tests for the VDMA driver.

Tests VdmaDriver register constants, DMASR decoding, and stop logic
using MockIP (no hardware required).
"""

import pytest

from software.overlay.drivers.vdma import VdmaDriver
from software.overlay.tests.checks import check_register_constants_sane
from software.overlay.tests.conftest import MockIP

pytestmark = pytest.mark.offboard


# -- Register sanity ----------------------------------------------------------

def test_register_offsets_sane():
    """All S2MM register offsets are aligned and within 0x100."""
    check_register_constants_sane(
        offsets={
            "PARK_PTR_REG": VdmaDriver.PARK_PTR_REG,
            "VDMA_VERSION": VdmaDriver.VDMA_VERSION,
            "S2MM_DMACR": VdmaDriver.S2MM_DMACR,
            "S2MM_DMASR": VdmaDriver.S2MM_DMASR,
            "S2MM_FRMSTORE": VdmaDriver.S2MM_FRMSTORE,
            "S2MM_VSIZE": VdmaDriver.S2MM_VSIZE,
            "S2MM_HSIZE": VdmaDriver.S2MM_HSIZE,
            "S2MM_FRMDLY_STRIDE": VdmaDriver.S2MM_FRMDLY_STRIDE,
            "S2MM_START_ADDR_BASE": VdmaDriver.S2MM_START_ADDR_BASE,
        },
        addr_space=0x100,
    )


# -- DMASR bit fields --------------------------------------------------------

def test_dmasr_bits_no_unintended_overlap():
    """Named DMASR error/status bits are disjoint."""
    bits = [
        VdmaDriver.DMASR_HALTED,
        VdmaDriver.DMASR_VDMA_INT_ERR,
        VdmaDriver.DMASR_VDMA_SLV_ERR,
        VdmaDriver.DMASR_VDMA_DEC_ERR,
        VdmaDriver.DMASR_SOF_EARLY_ERR,
        VdmaDriver.DMASR_EOL_EARLY_ERR,
        VdmaDriver.DMASR_SOF_LATE_ERR,
        VdmaDriver.DMASR_FRM_CNT_IRQ,
        VdmaDriver.DMASR_DLY_CNT_IRQ,
        VdmaDriver.DMASR_ERR_IRQ,
        VdmaDriver.DMASR_EOL_LATE_ERR,
    ]
    for i, a in enumerate(bits):
        for j, b in enumerate(bits):
            if i >= j:
                continue
            assert a & b == 0, (
                f"DMASR bit overlap: 0x{a:04X} & 0x{b:04X} = 0x{a & b:04X}"
            )


def test_dmasr_w1c_mask_correct():
    """W1C mask covers exactly bits {4, 7, 8, 11, 12, 13, 14, 15}."""
    expected = (
        (1 << 4) | (1 << 7) | (1 << 8)
        | (1 << 11) | (1 << 12) | (1 << 13) | (1 << 14) | (1 << 15)
    )
    assert expected == 0xF990
    assert VdmaDriver.DMASR_W1C_MASK == expected


# -- read_status --------------------------------------------------------------

def test_read_status_decodes_errors():
    """read_status decodes individual error flags from DMASR."""
    sr = VdmaDriver.DMASR_VDMA_INT_ERR | VdmaDriver.DMASR_SOF_LATE_ERR
    mock = MockIP({VdmaDriver.S2MM_DMASR: sr})
    drv = VdmaDriver(mock)
    status = drv.read_status()
    assert status["err_internal"] is True
    assert status["err_sof_late"] is True
    assert status["err_slave"] is False
    assert status["err_decode"] is False
    assert status["err_sof_early"] is False
    assert status["err_eol_early"] is False
    assert status["err_eol_late"] is False


def test_read_status_frame_count():
    """read_status extracts frame_count from DMASR bits [23:16]."""
    sr = 42 << 16
    mock = MockIP({VdmaDriver.S2MM_DMASR: sr})
    drv = VdmaDriver(mock)
    status = drv.read_status()
    assert status["frame_count"] == 42


def test_read_status_delay_count():
    """read_status extracts delay_count from DMASR bits [31:24]."""
    sr = 7 << 24
    mock = MockIP({VdmaDriver.S2MM_DMASR: sr})
    drv = VdmaDriver(mock)
    status = drv.read_status()
    assert status["delay_count"] == 7


# -- current_frame ------------------------------------------------------------

def test_current_frame_reads_park_ptr():
    """current_frame extracts bits [28:24] from PARK_PTR_REG."""
    mock = MockIP({VdmaDriver.PARK_PTR_REG: 5 << 24})
    drv = VdmaDriver(mock)
    assert drv.current_frame() == 5


# -- stop ---------------------------------------------------------------------

def test_stop_clears_rs_bit():
    """stop() clears RS bit but preserves other DMACR bits."""
    mock = MockIP({VdmaDriver.S2MM_DMACR: VdmaDriver.DMACR_RS | VdmaDriver.DMACR_CIRCULAR})
    drv = VdmaDriver(mock)
    drv.stop()
    cr = mock.read(VdmaDriver.S2MM_DMACR)
    assert cr & VdmaDriver.DMACR_RS == 0, "RS bit should be cleared"
    assert cr & VdmaDriver.DMACR_CIRCULAR != 0, "CIRCULAR bit should be preserved"
