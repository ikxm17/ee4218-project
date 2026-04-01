"""Off-board tests for the VPSS Lanczos coefficient module.

Tests the math-heavy coefficient generation, packing, and phase
computation in _vpss_coeff.py (no hardware required).
"""

import pytest

from software.overlay.drivers._vpss_coeff import (
    _lanczos_kernel,
    _generate_lanczos_table,
    LANCZOS_6TAP_64PHASE,
    pack_coefficients,
    calculate_phases,
    pack_phases,
)

pytestmark = pytest.mark.offboard


# -- Lanczos kernel -----------------------------------------------------------

def test_lanczos_kernel_at_zero():
    """Lanczos kernel at x=0 is exactly 1.0 (sinc(0) = 1)."""
    assert _lanczos_kernel(0) == 1.0


def test_lanczos_kernel_at_integer():
    """Lanczos kernel at x=1.0 and x=2.0 should be ~0.0 (sinc zeros)."""
    assert abs(_lanczos_kernel(1.0)) < 1e-10
    assert abs(_lanczos_kernel(2.0)) < 1e-10


def test_lanczos_kernel_outside_window():
    """Lanczos-3 kernel is zero for |x| >= 3."""
    assert _lanczos_kernel(3.0) == 0.0
    assert _lanczos_kernel(4.0) == 0.0


def test_lanczos_kernel_symmetry():
    """Lanczos kernel is symmetric: k(0.5) == k(-0.5)."""
    assert _lanczos_kernel(0.5) == _lanczos_kernel(-0.5)


# -- Pre-computed table -------------------------------------------------------

def test_table_shape():
    """LANCZOS_6TAP_64PHASE has 64 phases, each with 6 taps."""
    assert len(LANCZOS_6TAP_64PHASE) == 64
    for phase in LANCZOS_6TAP_64PHASE:
        assert len(phase) == 6


def test_every_phase_sums_to_2048():
    """Every phase's 6 coefficients must sum to exactly 2048 (= 1.0 in S4.11)."""
    for i, phase in enumerate(LANCZOS_6TAP_64PHASE):
        total = sum(phase)
        assert total == 2048, (
            f"Phase {i}: sum={total}, expected 2048, coeffs={phase}"
        )


def test_phase_0_center_tap():
    """Phase 0 (aligned with input pixel): center taps dominate, outer taps are 0."""
    phase0 = LANCZOS_6TAP_64PHASE[0]
    # Phase 0: output pixel aligned with input pixel.
    # Outer taps (indices 0, 1, 4, 5) should be 0.
    # Middle taps (indices 2, 3) should sum to 2048.
    assert phase0[0] == 0
    assert phase0[1] == 0
    assert phase0[4] == 0
    assert phase0[5] == 0
    assert phase0[2] + phase0[3] == 2048


# -- pack_coefficients --------------------------------------------------------

def test_pack_coefficients_word_count():
    """pack_coefficients returns 192 (offset, word) tuples: 64 phases x 3 words."""
    words = pack_coefficients(LANCZOS_6TAP_64PHASE, 6, 64)
    assert len(words) == 192


def test_pack_coefficients_format():
    """First word of phase 0: lo=coeff[0], hi=coeff[1], packed as (hi<<16)|lo."""
    words = pack_coefficients(LANCZOS_6TAP_64PHASE, 6, 64)
    offset, word = words[0]
    assert offset == 0

    coeff0 = LANCZOS_6TAP_64PHASE[0][0] & 0xFFFF
    coeff1 = LANCZOS_6TAP_64PHASE[0][1] & 0xFFFF
    expected = (coeff1 << 16) | coeff0
    assert word == expected


# -- calculate_phases ---------------------------------------------------------

def test_calculate_phases_identity():
    """1:1 scaling (256->256): all phases should be 0."""
    phases = calculate_phases(256, 256)
    assert all(p == 0 for p in phases)


def test_calculate_phases_downscale():
    """Downscale 1920->256: all 256 phase values should be in range 0-63."""
    phases = calculate_phases(1920, 256, 64)
    assert len(phases) == 256
    for p in phases:
        assert 0 <= p <= 63


def test_calculate_phases_length():
    """Output length must equal width_out."""
    phases = calculate_phases(1920, 224)
    assert len(phases) == 224


# -- pack_phases --------------------------------------------------------------

def test_pack_phases_even():
    """Two phases [0, 32] pack into one word: (32 << 16) | 0."""
    words = pack_phases([0, 32])
    assert len(words) == 1
    offset, word = words[0]
    assert word == (32 << 16) | 0


def test_pack_phases_odd_padded():
    """Odd-length [0, 32, 10] pads to [0, 32, 10, 0] -> 2 words."""
    words = pack_phases([0, 32, 10])
    assert len(words) == 2
    _, word0 = words[0]
    _, word1 = words[1]
    assert word0 == (32 << 16) | 0
    assert word1 == (0 << 16) | 10


def test_pack_phases_9bit_mask():
    """Phase values are masked to 9 bits (0x1FF max)."""
    words = pack_phases([0x1FF, 0x1FF])
    _, word = words[0]
    assert word == (0x1FF << 16) | 0x1FF
