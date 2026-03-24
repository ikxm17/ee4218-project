"""Frame acquisition abstraction for camera pipeline.

Provides a TestFrameSource (synthetic patterns) and a stubbed
HardwareFrameSource (PYNQ overlay + DDR buffers). The hardware
source falls back to test mode if the overlay cannot be loaded.
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

    Register offsets and buffer configuration are TBD until the block
    design is built in Vivado. This class will be fleshed out after
    bitstream generation.
    """

    def __init__(self, overlay_path: str) -> None:
        from pynq import Overlay  # noqa: F811

        logger.info("Loading overlay from %s", overlay_path)
        self._overlay = Overlay(overlay_path)
        logger.info("Overlay loaded. IPs: %s", list(self._overlay.ip_dict.keys()))

        # TODO: allocate CMA buffers for Multi-Scaler outputs
        # self._viz_buf = allocate(shape=(720, 1280, 3), dtype=np.uint8)
        # self._inf_buf = allocate(shape=(256, 256, 3), dtype=np.uint8)

        # TODO: configure VDMA and Multi-Scaler registers via MMIO
        # - Set VDMA frame store addresses, stride, dimensions
        # - Set Multi-Scaler source/dest addresses, output resolutions
        # See PG020 (VDMA) and PG325 (Multi-Scaler) for register maps

    @property
    def mode(self) -> str:
        return "hardware"

    def get_frame(self, buffer: str = "viz") -> np.ndarray:
        # TODO: return numpy view of the appropriate CMA buffer
        # For now, raise to trigger fallback to TestFrameSource
        raise NotImplementedError(
            "HardwareFrameSource.get_frame not yet implemented — "
            "waiting for block design and bitstream"
        )

    def set_gamma_bypass(self, bypass: bool) -> None:
        # TODO: write to Gamma LUT AXI4-Lite bypass register
        # The register offset will be determined from the .hwh file
        # after the block design is built
        logger.info("Gamma bypass set to %s (hardware)", bypass)

    def close(self) -> None:
        if hasattr(self, "_overlay"):
            self._overlay.free()
            logger.info("Overlay freed")


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
