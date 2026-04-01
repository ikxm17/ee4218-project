"""Off-board tests for HLS common constants and helpers.

Tests the shared AP_CTRL constants, interrupt register offsets, and
hls_start / hls_stop / hls_read_status helpers from _hls_common.py.
"""

import pytest

from software.overlay.drivers._hls_common import (
    AP_CTRL,
    AP_CTRL_AUTO_RESTART,
    AP_CTRL_DONE,
    AP_CTRL_IDLE,
    AP_CTRL_READY,
    AP_CTRL_START,
    GIE,
    IER,
    ISR,
    hls_read_status,
    hls_start,
    hls_stop,
)
from software.overlay.tests.conftest import MockIP

pytestmark = pytest.mark.offboard


# -- Constant verification ---------------------------------------------------

def test_ap_ctrl_constants():
    """AP_CTRL bit masks have correct documented values."""
    assert AP_CTRL_START == 0x01
    assert AP_CTRL_DONE == 0x02
    assert AP_CTRL_IDLE == 0x04
    assert AP_CTRL_READY == 0x08
    assert AP_CTRL_AUTO_RESTART == 0x80


def test_ap_ctrl_bits_no_overlap():
    """All AP_CTRL bit masks are disjoint (no overlapping bits)."""
    bits = [AP_CTRL_START, AP_CTRL_DONE, AP_CTRL_IDLE, AP_CTRL_READY,
            AP_CTRL_AUTO_RESTART]
    for i, a in enumerate(bits):
        for b in bits[i + 1:]:
            assert a & b == 0, (
                f"AP_CTRL bit overlap: 0x{a:02X} & 0x{b:02X} = 0x{a & b:02X}"
            )


def test_interrupt_register_offsets():
    """Interrupt registers at documented offsets."""
    assert GIE == 0x04
    assert IER == 0x08
    assert ISR == 0x0C


# -- hls_start / hls_stop ----------------------------------------------------

def test_hls_start_writes_correct_value():
    """hls_start writes START | AUTO_RESTART (0x81) to AP_CTRL."""
    mock = MockIP()
    hls_start(mock)
    assert mock.read(AP_CTRL) == AP_CTRL_START | AP_CTRL_AUTO_RESTART
    assert mock.read(AP_CTRL) == 0x81


def test_hls_stop_clears_ap_ctrl():
    """hls_stop writes 0x00 to AP_CTRL."""
    mock = MockIP()
    hls_start(mock)  # put something in AP_CTRL first
    hls_stop(mock)
    assert mock.read(AP_CTRL) == 0x00


# -- hls_read_status ---------------------------------------------------------

def test_hls_read_status_idle():
    """Status decodes idle=True when only IDLE bit is set."""
    mock = MockIP({AP_CTRL: AP_CTRL_IDLE})  # 0x04
    status = hls_read_status(mock)
    assert status["idle"] is True
    assert status["running"] is False
    assert status["done"] is False


def test_hls_read_status_running():
    """Status decodes running=True with START|AUTO_RESTART and not idle."""
    mock = MockIP({AP_CTRL: AP_CTRL_START | AP_CTRL_AUTO_RESTART})  # 0x81
    status = hls_read_status(mock)
    assert status["running"] is True
    assert status["idle"] is False
    assert status["auto_restart"] is True
