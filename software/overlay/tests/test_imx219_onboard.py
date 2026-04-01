"""On-board tests for the IMX219 sensor driver.

Requires the PYNQ overlay to be loaded and the IMX219 camera connected
to the Kria board via the RPi camera connector.
"""

import pytest

pytestmark = pytest.mark.onboard


def test_sensor_detected():
    """Create Imx219Driver, call read_status(), verify detected=True and model_id=0x0219.

    Skips if smbus2 is not available or no Xilinx AXI IIC adapter is found
    (overlay not loaded or no camera connected).
    """
    smbus2 = pytest.importorskip("smbus2", reason="smbus2 not available")
    from software.overlay.drivers.imx219 import Imx219Driver

    try:
        driver = Imx219Driver()
    except RuntimeError as exc:
        if "No Xilinx AXI IIC adapter found" in str(exc):
            pytest.skip("No Xilinx AXI IIC adapter found (overlay not loaded?)")
        raise

    try:
        status = driver.read_status()
        assert status["detected"] is True, (
            f"IMX219 not detected: model_id=0x{status['model_id']:04X}"
        )
        assert status["model_id"] == 0x0219, (
            f"Expected model_id=0x0219, got 0x{status['model_id']:04X}"
        )
    finally:
        driver.close()
