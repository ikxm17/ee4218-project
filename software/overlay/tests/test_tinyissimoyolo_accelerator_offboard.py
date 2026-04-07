"""Off-board tests for the TinyissimoYOLO accelerator driver.

Exercises register writes, packing, post-processing, and the new
configure()/read_status() conveniences using MockIP. No PYNQ import,
no hardware required — must be runnable on a dev machine.
"""

import numpy as np
import pytest

from software.overlay.drivers.tinyissimoyolo_accelerator import (
    OUTPUT_SCALE,
    OUTPUT_ZP,
    TinyissimoYoloAcceleratorDriver,
    decode_dfl,
    post_process,
)
from software.overlay.tests.checks import check_register_constants_sane
from software.overlay.tests.conftest import MockIP

pytestmark = pytest.mark.offboard


# ---------------------------------------------------------------------------
# Driver registry integration
# ---------------------------------------------------------------------------

def test_driver_is_exported_from_package():
    """Top-level package import must surface the driver class so
    orchestration code does not need to know the submodule path."""
    from software.overlay import drivers
    assert hasattr(drivers, "TinyissimoYoloAcceleratorDriver")
    assert (
        drivers.TinyissimoYoloAcceleratorDriver
        is TinyissimoYoloAcceleratorDriver
    )


def test_driver_is_in_registry():
    """DRIVER_REGISTRY (used by audit_drivers) must include the new
    driver keyed by its IP_VLNV so .xsa audits classify it as covered."""
    from software.overlay.drivers import DRIVER_REGISTRY
    assert (
        "user.org:user:tinyissimoyolo_accelerator:1.0" in DRIVER_REGISTRY
    )
    assert (
        DRIVER_REGISTRY["user.org:user:tinyissimoyolo_accelerator:1.0"]
        is TinyissimoYoloAcceleratorDriver
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CountingMockIP(MockIP):
    """MockIP that records every write as an (offset, value) tuple.

    Needed for FIFO-style registers where the same offset is written many
    times — the parent MockIP only retains the last value.
    """

    def __init__(self, defaults: dict | None = None):
        super().__init__(defaults)
        self.writes: list[tuple[int, int]] = []

    def write(self, offset: int, value: int) -> None:
        self.writes.append((offset, value))
        super().write(offset, value)


# ---------------------------------------------------------------------------
# Identity / registry
# ---------------------------------------------------------------------------

def test_ip_vlnv_matches_component_xml():
    """IP_VLNV must match the published component.xml so audit_drivers
    finds the driver when it parses a .xsa containing this IP."""
    assert (
        TinyissimoYoloAcceleratorDriver.IP_VLNV
        == "user.org:user:tinyissimoyolo_accelerator:1.0"
    )


def test_ip_name_matches_block_design_instance():
    """IP_NAME is the convenience reference to the block design instance
    that conftest fixtures and orchestration scripts look up by name."""
    assert (
        TinyissimoYoloAcceleratorDriver.IP_NAME == "tinyissimoyolo_accel_0"
    )


# ---------------------------------------------------------------------------
# Register sanity
# ---------------------------------------------------------------------------

def test_register_offsets_sane():
    """All register constants are aligned, unique, within the 8 KB
    AXI-Lite address space (matches axil_regs.sv ADDR_W=13)."""
    check_register_constants_sane(
        offsets={
            "CTRL":        TinyissimoYoloAcceleratorDriver._CTRL,
            "STATUS":      TinyissimoYoloAcceleratorDriver._STATUS,
            "MODE":        TinyissimoYoloAcceleratorDriver._MODE,
            "CYCLE_CNT":   TinyissimoYoloAcceleratorDriver._CYCLE_CNT,
            "LAYER_IDX":   TinyissimoYoloAcceleratorDriver._LAYER_IDX,
            "PIXEL_FIFO":  TinyissimoYoloAcceleratorDriver._PIXEL_FIFO,
            "PIXEL_CNT":   TinyissimoYoloAcceleratorDriver._PIXEL_CNT,
            "RESULT_BASE": TinyissimoYoloAcceleratorDriver._RESULT_BASE,
        },
        addr_space=0x2000,
    )


def test_register_offsets_match_axil_regs_sv():
    """Spot-check the offsets against the source-of-truth in axil_regs.sv."""
    cls = TinyissimoYoloAcceleratorDriver
    assert cls._CTRL == 0x000
    assert cls._STATUS == 0x004
    assert cls._MODE == 0x008
    assert cls._CYCLE_CNT == 0x00C
    assert cls._LAYER_IDX == 0x010
    assert cls._PIXEL_FIFO == 0x020
    assert cls._PIXEL_CNT == 0x024
    assert cls._RESULT_BASE == 0x100


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_constructor_accepts_ip_keyword_arg():
    """The constructor parameter is named `ip` (matches DemosaicDriver,
    VdmaDriver, etc.) so orchestration code can use a uniform pattern."""
    mock = MockIP()
    drv = TinyissimoYoloAcceleratorDriver(ip=mock)
    assert drv is not None


# ---------------------------------------------------------------------------
# set_mode / configure / soft_reset / start
# ---------------------------------------------------------------------------

def test_set_mode_writes_mode_register():
    mock = MockIP()
    drv = TinyissimoYoloAcceleratorDriver(mock)
    drv.set_mode(0)
    assert mock.read(0x008) == 0
    drv.set_mode(1)
    assert mock.read(0x008) == 1


def test_soft_reset_writes_ctrl_bit_7():
    mock = MockIP()
    drv = TinyissimoYoloAcceleratorDriver(mock)
    drv.soft_reset()
    assert mock.read(0x000) == 0x80


def test_start_writes_start_and_fifo_reset():
    """start() asserts CTRL[0]=start and CTRL[1]=fifo_rst (= 0x03).
    Both bits are auto-clearing one-shots in axil_regs.sv:142-143."""
    mock = MockIP()
    drv = TinyissimoYoloAcceleratorDriver(mock)
    drv.start()
    assert mock.read(0x000) == 0x03


def test_configure_resets_then_sets_mode():
    """configure(mode=0) is the orchestration entry point: soft reset
    followed by mode select. Default mode is 0 (FIFO)."""
    mock = CountingMockIP()
    drv = TinyissimoYoloAcceleratorDriver(mock)
    drv.configure(mode=0)
    # Soft reset must precede the mode write so the mode latches into a
    # cleanly-reset register file.
    soft_reset_writes = [w for w in mock.writes if w == (0x000, 0x80)]
    mode_writes = [w for w in mock.writes if w[0] == 0x008]
    assert soft_reset_writes, "configure() must issue a soft reset"
    assert mode_writes == [(0x008, 0)], "configure(mode=0) must set MODE=0"
    # Order matters: soft reset must come before mode write
    soft_idx = next(i for i, w in enumerate(mock.writes) if w == (0x000, 0x80))
    mode_idx = next(i for i, w in enumerate(mock.writes) if w[0] == 0x008)
    assert soft_idx < mode_idx, "soft reset must precede mode write"


def test_configure_mode_one():
    mock = MockIP()
    drv = TinyissimoYoloAcceleratorDriver(mock)
    drv.configure(mode=1)
    assert mock.read(0x008) == 1


# ---------------------------------------------------------------------------
# read_status
# ---------------------------------------------------------------------------

def test_read_status_decodes_all_bits():
    """read_status() decodes the 4 status bits into a dict matching
    the convention of VdmaDriver.read_status()."""
    mock = MockIP({0x004: 0b1111})  # busy=1, done=1, idle=1, preload_done=1
    drv = TinyissimoYoloAcceleratorDriver(mock)
    status = drv.read_status()
    assert isinstance(status, dict)
    assert status["busy"] is True
    assert status["done"] is True
    assert status["idle"] is True
    assert status["preload_done"] is True
    assert status["raw"] == 0b1111


def test_read_status_idle_only():
    mock = MockIP({0x004: 0b0100})  # idle=1
    drv = TinyissimoYoloAcceleratorDriver(mock)
    status = drv.read_status()
    assert status["busy"] is False
    assert status["done"] is False
    assert status["idle"] is True
    assert status["preload_done"] is False


def test_read_status_done_set():
    mock = MockIP({0x004: 0b1010})  # done=1, preload_done=1
    drv = TinyissimoYoloAcceleratorDriver(mock)
    status = drv.read_status()
    assert status["busy"] is False
    assert status["done"] is True
    assert status["preload_done"] is True


# ---------------------------------------------------------------------------
# write_pixels
# ---------------------------------------------------------------------------

def _make_test_image() -> np.ndarray:
    """Deterministic 256x256 RGB image with distinct per-channel values."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[..., 0] = 200  # R = 200 → int8 72
    img[..., 1] = 100  # G = 100 → int8 -28
    img[..., 2] = 50   # B =  50 → int8 -78
    return img


def test_write_pixels_emits_exactly_65536_writes_to_pixel_fifo():
    """The accelerator's pixel preload requires exactly 65 536 32-bit
    writes to PIXEL_FIFO (verified against axil_regs.sv:193 — preload_done
    asserts when pixel_count == 65536)."""
    mock = CountingMockIP()
    drv = TinyissimoYoloAcceleratorDriver(mock)
    drv.write_pixels(_make_test_image())
    fifo_writes = [w for w in mock.writes if w[0] == 0x020]
    assert len(fifo_writes) == 65536, (
        f"Expected 65536 writes to PIXEL_FIFO, got {len(fifo_writes)}"
    )


def test_write_pixels_packs_RGB_pad_little_endian():
    """Each 32-bit FIFO word must contain {pad=0, B, G, R} in MSB→LSB
    order, matching tb_tinyissimoyolo_accel.sv:317-322 exactly:
        axil_write(ADDR_PIXEL_FIFO, {8'd0, pixel_mem[2][n],
                                     pixel_mem[1][n], pixel_mem[0][n]})

    Driver subtracts 128 from uint8 → int8 so the on-the-wire bytes match
    pixels_layer0.mem (which generate_conv3d_golden.py writes after the
    same uint8 → int8 reinterpret)."""
    mock = CountingMockIP()
    drv = TinyissimoYoloAcceleratorDriver(mock)
    drv.write_pixels(_make_test_image())

    fifo_writes = [w[1] for w in mock.writes if w[0] == 0x020]
    # First word: R=200→72 (0x48), G=100→-28 (0xE4), B=50→-78 (0xB2), pad=0
    expected = (0x00 << 24) | (0xB2 << 16) | (0xE4 << 8) | 0x48
    assert fifo_writes[0] == expected, (
        f"First FIFO word: got 0x{fifo_writes[0]:08X}, "
        f"expected 0x{expected:08X}"
    )
    # Same image throughout, so all 65 536 words must match
    assert all(w == expected for w in fifo_writes)


def test_write_pixels_rejects_wrong_shape():
    mock = CountingMockIP()
    drv = TinyissimoYoloAcceleratorDriver(mock)
    with pytest.raises(ValueError, match="must be \\(256,256,3\\)"):
        drv.write_pixels(np.zeros((128, 128, 3), dtype=np.uint8))


def test_write_pixels_rejects_wrong_dtype():
    mock = CountingMockIP()
    drv = TinyissimoYoloAcceleratorDriver(mock)
    with pytest.raises(ValueError, match="must be uint8"):
        drv.write_pixels(np.zeros((256, 256, 3), dtype=np.float32))


def test_set_mode_rejects_invalid_value():
    """set_mode is the public API surface; only modes 0 and 1 exist
    in the hardware. Pre-validate so a typo doesn't silently land in
    the MODE register and cause obscure preload failures."""
    mock = MockIP()
    drv = TinyissimoYoloAcceleratorDriver(mock)
    with pytest.raises(ValueError, match="must be 0"):
        drv.set_mode(2)


# ---------------------------------------------------------------------------
# unpack_detections
# ---------------------------------------------------------------------------

def test_unpack_detections_shape():
    """The 320 URAM words split as 256 cv2 (4 groups × 64 spatial × 16ch)
    + 64 cv3 (1 group × 64 spatial × 3ch) → (8, 8, 67) detection tensor."""
    raw = np.zeros((320, 16), dtype=np.int8)
    drv = TinyissimoYoloAcceleratorDriver(MockIP())
    out = drv.unpack_detections(raw)
    assert out.shape == (8, 8, 67)
    assert out.dtype == np.float32


def test_unpack_detections_zero_input_dequantises_to_neg_zp_scaled():
    """An int8 zero dequantises to (0 - OUTPUT_ZP) * OUTPUT_SCALE."""
    raw = np.zeros((320, 16), dtype=np.int8)
    drv = TinyissimoYoloAcceleratorDriver(MockIP())
    out = drv.unpack_detections(raw)
    expected = (0 - OUTPUT_ZP) * OUTPUT_SCALE
    assert np.allclose(out, expected)


def test_unpack_detections_known_value():
    """Set raw[0,0]=zp+1 → dequant = scale; verify it lands at (y,x)=(0,0)
    channel 0 of cv2."""
    raw = np.full((320, 16), OUTPUT_ZP, dtype=np.int8)
    raw[0, 0] = OUTPUT_ZP + 1   # one LSB above zero point
    drv = TinyissimoYoloAcceleratorDriver(MockIP())
    out = drv.unpack_detections(raw)
    # cv2 layout: 4 groups × (8, 8, 16) → transpose → (8, 8, 64)
    # raw[0] is group 0 spatial (0,0), so out[0,0,0] should be the only
    # cell holding the +1 value (rest are 0).
    assert np.isclose(out[0, 0, 0], OUTPUT_SCALE)
    # Spot check that it's not also smeared elsewhere
    assert np.isclose(out[1, 0, 0], 0.0)


# ---------------------------------------------------------------------------
# decode_dfl / post_process
# ---------------------------------------------------------------------------

def test_decode_dfl_uniform_distribution():
    """Uniform 16-bin distribution → expected value 7.5 (mean of 0..15)."""
    dfl = np.zeros((1, 4, 16), dtype=np.float32)  # softmax of zeros = uniform
    out = decode_dfl(dfl)
    assert out.shape == (1, 4)
    assert np.allclose(out, 7.5, atol=1e-5)


def test_decode_dfl_concentrated_distribution():
    """One-hot bin 5 → expected value 5.0."""
    dfl = np.full((4, 16), -1e9, dtype=np.float32)
    dfl[:, 5] = 0.0
    out = decode_dfl(dfl)
    assert np.allclose(out, 5.0, atol=1e-5)


def test_post_process_no_detections_below_threshold():
    """All-zero data → sigmoid(0) = 0.5, conf_thresh > 0.5 keeps nothing."""
    data = np.zeros((8, 8, 67), dtype=np.float32)
    boxes, scores, class_ids = post_process(data, conf_thresh=0.6)
    assert boxes == []
    assert scores == []
    assert class_ids == []


def test_post_process_strong_detection_at_center():
    """Place a strong class-0 logit and DFL pointing to known LTRB at the
    center cell; verify a single detection is produced."""
    data = np.zeros((8, 8, 67), dtype=np.float32)
    # Class 0 logit large → sigmoid ≈ 1
    data[4, 4, 64] = 10.0
    # DFL: each of 4 LTRB groups has bin 5 strongly preferred
    for grp in range(4):
        data[4, 4, grp * 16 + 5] = 10.0  # bin 5 within group
    boxes, scores, class_ids = post_process(data, conf_thresh=0.5)
    assert len(boxes) == 1
    assert class_ids == [0]
    assert scores[0] > 0.9
