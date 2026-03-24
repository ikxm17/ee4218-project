"""Frame acquisition abstraction for camera pipeline.

Provides a TestFrameSource (synthetic patterns) and a
HardwareFrameSource (PYNQ overlay + DDR buffers via CameraOverlay).
The hardware source falls back to test mode if the overlay cannot
be loaded or PYNQ is unavailable.
"""

from abc import ABC, abstractmethod
import logging
import time

import numpy as np

from . import config

logger = logging.getLogger(__name__)


class FrameSource(ABC):
    """Abstract frame source."""

    @abstractmethod
    def get_frame(self, buffer: str = "viz") -> np.ndarray:
        """Return an RGB frame as a uint8 numpy array.

        Args:
            buffer: "viz" for 720p visualization, "inference" for 256x256.
        """

    @abstractmethod
    def set_gamma_bypass(self, bypass: bool) -> None:
        """Toggle the Gamma LUT bypass register."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    @property
    @abstractmethod
    def mode(self) -> str:
        """Return "hardware" or "test"."""


class TestFrameSource(FrameSource):
    """Generates synthetic test patterns for development without hardware."""

    def __init__(self) -> None:
        self._frame_count = 0
        self._gamma_bypass = False
        logger.info("TestFrameSource initialized (no hardware)")

    @property
    def mode(self) -> str:
        return "test"

    def get_frame(self, buffer: str = "viz") -> np.ndarray:
        shape = config.BUFFER_SHAPES.get(buffer, config.BUFFER_SHAPES["viz"])
        h, w, _ = shape
        frame = self._generate_pattern(h, w)
        self._frame_count += 1
        return frame

    def set_gamma_bypass(self, bypass: bool) -> None:
        self._gamma_bypass = bypass
        logger.info("Gamma bypass set to %s (no-op in test mode)", bypass)

    def close(self) -> None:
        pass

    def _generate_pattern(self, h: int, w: int) -> np.ndarray:
        """Moving color gradient with frame counter for visual confirmation."""
        t = time.monotonic()
        # Horizontal gradient shifts over time
        x = np.linspace(0, 1, w, dtype=np.float32)
        y = np.linspace(0, 1, h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        offset = (t * 0.1) % 1.0
        r = np.uint8(255 * ((xv + offset) % 1.0))
        g = np.uint8(255 * ((yv + offset * 0.7) % 1.0))
        b = np.uint8(255 * (((xv + yv) * 0.5 + offset * 0.3) % 1.0))

        frame = np.stack([r, g, b], axis=-1)

        # Burn frame counter into top-left corner (simple 8-pixel-high bar)
        bar_width = min(self._frame_count % w, w)
        frame[:8, :bar_width] = [255, 255, 255]

        return frame


class HardwareFrameSource(FrameSource):
    """Reads frames from PYNQ-allocated DDR buffers via the camera overlay.

    Delegates to CameraOverlay (software/overlay/camera.py) which handles
    the full initialization sequence: overlay load, GPIO power enable,
    IMX219 I2C init, CMA buffer allocation, PL IP configuration, and
    streaming start.
    """

    def __init__(self, overlay_path: str) -> None:
        from software.overlay import CameraOverlay

        logger.info("Initializing camera pipeline from %s", overlay_path)
        self._camera = CameraOverlay(overlay_path)
        logger.info("Camera pipeline active")

    @property
    def mode(self) -> str:
        return "hardware"

    def get_frame(self, buffer: str = "viz") -> np.ndarray:
        return self._camera.get_frame(buffer)

    def set_gamma_bypass(self, bypass: bool) -> None:
        self._camera.set_gamma_bypass(bypass)
        logger.info("Gamma bypass set to %s (hardware)", bypass)

    def close(self) -> None:
        if hasattr(self, "_camera"):
            self._camera.close()
            logger.info("Camera pipeline closed")


def create_frame_source() -> FrameSource:
    """Create the best available frame source.

    Attempts HardwareFrameSource first; falls back to TestFrameSource
    if PYNQ is unavailable or the overlay cannot be loaded.
    """
    try:
        return HardwareFrameSource(config.OVERLAY_PATH)
    except Exception as exc:
        logger.warning("Hardware frame source unavailable: %s", exc)
        logger.info("Falling back to test pattern source")
        return TestFrameSource()
