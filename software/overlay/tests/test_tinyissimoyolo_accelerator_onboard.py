"""On-board tests for the TinyissimoYOLO accelerator driver.

Requires:
  - A bitstream containing user.org:user:tinyissimoyolo_accelerator:1.0
    in hardware/output/ (the conftest `accel_overlay` fixture finds it
    by inspecting sibling .hwh files).
  - PYNQ runtime + ZOCL loaded (see CLAUDE.local.md boot sequence).
  - software/inference/data/input_image.jpg (the canonical test image
    used by both the TFLite baseline and the golden generator).

These tests skip cleanly on dev machines (`pytest_collection_modifyitems`
in conftest skips anything tagged @pytest.mark.onboard when
/dev/dri/renderD128 is missing).
"""

import pathlib
import time

import numpy as np
import pytest

from software.overlay.drivers.tinyissimoyolo_accelerator import (
    TinyissimoYoloAcceleratorDriver,
)
from software.overlay.tests.checks import (
    check_register_readback,
    load_golden_uram_mem,
)

pytestmark = pytest.mark.onboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parents[3]
TEST_IMAGE = REPO_ROOT / "software" / "inference" / "data" / "input_image.jpg"
GOLDEN_LAYER16 = (
    REPO_ROOT / "hardware" / "testbench" / "inference_hdl"
    / "golden_layer16_uram.mem"
)


def _load_test_image_uint8() -> np.ndarray:
    """Mirrors software/inference/run_inference.py:119 exactly so the
    HDL preprocessing matches the TFLite baseline AND the .mem generator."""
    from PIL import Image
    img = Image.open(str(TEST_IMAGE)).convert("RGB").resize((256, 256))
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Register-level smoke
# ---------------------------------------------------------------------------

def test_mode_register_readback(tinyissimoyolo_accel_ip):
    """MODE is R/W and not auto-clearing — perfect target for a basic
    register handshake check that proves AXI-Lite reaches the IP."""
    drv = TinyissimoYoloAcceleratorDriver(tinyissimoyolo_accel_ip)
    drv.soft_reset()
    time.sleep(0.001)
    check_register_readback(tinyissimoyolo_accel_ip, drv._MODE, 0x1)
    check_register_readback(tinyissimoyolo_accel_ip, drv._MODE, 0x0)


def test_soft_reset_returns_to_idle(tinyissimoyolo_accel_ip):
    """After soft_reset() the phase FSM must report idle within a few
    cycles. The PS clock is much slower than the PL, so a 1 ms wait is
    plenty for the soft reset pulse to land and the FSM to settle."""
    drv = TinyissimoYoloAcceleratorDriver(tinyissimoyolo_accel_ip)
    drv.soft_reset()
    time.sleep(0.001)
    status = drv.read_status()
    assert status["idle"] is True, f"Expected idle, got {status}"
    assert status["busy"] is False
    assert status["done"] is False


# ---------------------------------------------------------------------------
# End-to-end inference
# ---------------------------------------------------------------------------

def test_inference_on_zeros_completes(tinyissimoyolo_accel_ip):
    """Run on a black image — proves the full pipeline (mode, preload,
    inference, done assertion) completes without timeout. Output values
    are not checked here; the bit-exact test below covers correctness."""
    drv = TinyissimoYoloAcceleratorDriver(tinyissimoyolo_accel_ip)
    drv.configure(mode=0)
    drv.start()
    drv.write_pixels(np.zeros((256, 256, 3), dtype=np.uint8))
    assert drv.wait_done(timeout_s=2.0), "Inference timed out on zero image"
    assert drv.cycle_count > 0, "CYCLE_CNT did not advance during inference"


def test_inference_on_test_image_completes(tinyissimoyolo_accel_ip):
    """Run on the canonical test image and confirm the run() helper
    returns a populated detection dict. Tolerance is loose: the
    bit-exact test below is the strict oracle."""
    if not TEST_IMAGE.exists():
        pytest.skip(f"Test image not found: {TEST_IMAGE}")
    image = _load_test_image_uint8()
    drv = TinyissimoYoloAcceleratorDriver(tinyissimoyolo_accel_ip)
    result = drv.run(image)
    assert "boxes" in result
    assert "cycle_count" in result
    assert result["cycle_count"] > 0


# ---------------------------------------------------------------------------
# Bit-exact comparison vs simulation golden
# ---------------------------------------------------------------------------

def test_run_is_deterministic_across_back_to_back_calls(tinyissimoyolo_accel_ip):
    """Two consecutive `drv.run()` calls on the same input must produce
    bit-identical results.

    Regression test for the bug where `run()` skipped the soft reset
    between calls, letting stale URAM contents from a prior inference
    leak into the next one. The fix is `run()` calling `configure()`
    (soft_reset + set_mode) before `start()`. Without it, A1 ≠ A2 with
    ~50% of the result bytes differing.
    """
    drv = TinyissimoYoloAcceleratorDriver(tinyissimoyolo_accel_ip)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    r1 = drv.run(image)["raw_table"].copy()
    r2 = drv.run(image)["raw_table"].copy()
    assert np.array_equal(r1, r2), (
        f"run() is non-deterministic across back-to-back calls: "
        f"{(r1 != r2).sum()}/{r1.size} bytes differ. "
        f"Did run() skip the soft reset?"
    )


def test_cv3_output_bit_exact_vs_golden(tinyissimoyolo_accel_ip):
    """The strictest oracle: after running on the canonical test image,
    the cv3 region of the AXI-Lite RESULT readback must byte-for-byte
    match hardware/testbench/inference_hdl/golden_layer16_uram.mem.

    The golden was generated by hardware/scripts/generate_conv3d_golden.py
    from `software/inference/data/input_image.jpg` with the same
    `(uint8 - 128).astype(int8)` preprocessing the driver applies. If
    this test fails it means either:
      (a) hardware != xsim (synthesis / placement / timing bug), or
      (b) pixels_layer0.mem in the testbench dir was regenerated from a
          different image — re-run generate_conv3d_golden.py to refresh.
    """
    if not TEST_IMAGE.exists():
        pytest.skip(f"Test image not found: {TEST_IMAGE}")
    if not GOLDEN_LAYER16.exists():
        pytest.skip(f"Golden file not found: {GOLDEN_LAYER16}")

    image = _load_test_image_uint8()
    drv = TinyissimoYoloAcceleratorDriver(tinyissimoyolo_accel_ip)
    result = drv.run(image)
    cv3_actual = result["raw_table"][256:320, :3]

    cv3_golden = load_golden_uram_mem(str(GOLDEN_LAYER16), num_words=64)[:, :3]

    if not np.array_equal(cv3_actual, cv3_golden):
        diff_idx = np.where(cv3_actual != cv3_golden)
        sample = list(zip(*diff_idx))[:5]
        pytest.fail(
            f"cv3 mismatch in {len(diff_idx[0])} positions; "
            f"first 5: {sample}\n"
            f"actual head: {cv3_actual[:2]}\ngolden head: {cv3_golden[:2]}"
        )
