"""Sensor Demosaic driver — Bayer-to-RGB conversion.

Reference: PG286 (docs/ips/video/pg286-v-demosaic-en-us-1.1.pdf)

Register map (verified against .hwh and xv_demosaic_hw.h):
    0x00   AP_CTRL      (via _hls_common)
    0x04   GIE          (via _hls_common)
    0x08   IER          (via _hls_common)
    0x0C   ISR          (via _hls_common)
    0x10   width        (16-bit, active pixels per scanline)
    0x18   height       (16-bit, active lines per frame)
    0x28   bayer_phase  (2-bit, Bayer grid starting position)

ADDR_WIDTH=6 -> 64-byte register space.
"""

import logging

from ._hls_common import (
    AP_CTRL,
    AP_CTRL_START,
    AP_CTRL_AUTO_RESTART,
    hls_read_status,
    hls_stop,
)

logger = logging.getLogger(__name__)


class DemosaicDriver:
    """Sensor Demosaic IP (PG286)."""

    # IP identification (for audit — class covers ALL instances of this type)
    IP_VLNV = "xilinx.com:ip:v_demosaic:1.1"
    IP_NAME = "v_demosaic_0"  # primary instance

    # -- Register map (PG286 Table 2-6, verified against HLS header) --
    WIDTH = 0x10
    HEIGHT = 0x18
    BAYER_PHASE = 0x28  # 0=RGGB, 1=GRBG, 2=GBRG, 3=BGGR

    def __init__(self, ip):
        """Wrap a PYNQ DefaultIP handle for the Demosaic."""
        self._ip = ip

    def configure(
        self, width: int = 1920, height: int = 1080, bayer_phase: int = 0,
    ) -> None:
        """Configure and start the Sensor Demosaic IP.

        Args:
            width: Active pixels per scanline (min 64).
            height: Active lines per frame (min 64).
            bayer_phase: Bayer grid starting position.
                0=RGGB (IMX219 default), 1=GRBG, 2=GBRG, 3=BGGR.
        """
        self._ip.write(self.WIDTH, width)
        self._ip.write(self.HEIGHT, height)
        self._ip.write(self.BAYER_PHASE, bayer_phase)
        self._ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)
        logger.info(
            "Demosaic configured: %dx%d, bayer_phase=%d", width, height, bayer_phase,
        )

    def start(self) -> None:
        """Re-assert AP_CTRL to start/restart the Demosaic."""
        self._ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)

    def stop(self) -> None:
        """Clear AP_CTRL to halt the Demosaic."""
        hls_stop(self._ip)

    def read_status(self) -> dict:
        """Read and decode AP_CTRL status bits.

        Returns:
            {"idle": bool, "done": bool, "ready": bool,
             "running": bool, "auto_restart": bool, "ap_ctrl_raw": int}
        """
        return hls_read_status(self._ip)
