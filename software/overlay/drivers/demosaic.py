"""Sensor Demosaic driver — Bayer-to-RGB conversion.

Reference: PG286 (docs/ips/video/pg286-v-demosaic-en-us-1.1.pdf)
"""

import logging

from ._hls_common import AP_CTRL, AP_CTRL_START, AP_CTRL_AUTO_RESTART

logger = logging.getLogger(__name__)


class DemosaicDriver:
    """Sensor Demosaic IP (PG286)."""

    # Block design instance name (for .hwh audit)
    IP_NAME = "v_demosaic_0"

    # -- Register map (PG286 Table 2-6) --
    WIDTH = 0x10
    HEIGHT = 0x18
    BAYER_PHASE = 0x28  # 0=RGGB, 1=GRBG, 2=GBRG, 3=BGGR

    def __init__(self, ip):
        """Wrap a PYNQ DefaultIP handle for the Demosaic."""
        self._ip = ip

    def configure(self, width: int = 1920, height: int = 1080) -> None:
        """Configure and start the Sensor Demosaic IP.

        IMX219 outputs RGGB Bayer pattern (bayer_phase = 0).
        """
        self._ip.write(self.WIDTH, width)
        self._ip.write(self.HEIGHT, height)
        self._ip.write(self.BAYER_PHASE, 0)  # RGGB
        self._ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)
        logger.info("Demosaic configured: %dx%d, RGGB", width, height)

    def start(self) -> None:
        """Re-assert AP_CTRL to start/restart the Demosaic."""
        self._ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)
