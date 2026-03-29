"""Gamma LUT driver — RGB lookup-table correction.

Reference: PG285 (docs/ips/video/pg285-v-gamma-lut-en-us-1.1.pdf)

Register map (from PG285 / xv_gamma_lut_hw.h):
    0x00   AP_CTRL
    0x10   width
    0x18   height
    0x20   video_format
    0x0800 gamma_lut_0[0..1023]  (R channel, 2 x U16 packed per 32-bit word)
    0x1000 gamma_lut_1[0..1023]  (G channel, 512 words = 0x800 bytes per LUT)
    0x1800 gamma_lut_2[0..1023]  (B channel)

ADDR_WIDTH=13 -> address space is 0x0000-0x1FFF (8 KB).  Each LUT
region is 0x800 bytes.  Writing at 4-byte stride (1 entry/word)
overflows LUT2 past 0x1FFF, aliasing back to 0x0000 and
corrupting the control registers.  Always use 2-entry packing.

Bypass: set ap_start=0 with auto_restart=0.  Data passes through
because the IP is wired inline on the AXI4-Stream path and uses
a pass-through mode when not started.
"""

import logging

from ._hls_common import AP_CTRL, AP_CTRL_START, AP_CTRL_AUTO_RESTART

logger = logging.getLogger(__name__)


class GammaLutDriver:
    """Gamma LUT IP (PG285)."""

    # Block design instance name (for .hwh audit)
    IP_NAME = "v_gamma_lut_0"

    # -- Register map --
    WIDTH = 0x10
    HEIGHT = 0x18
    VIDEO_FORMAT = 0x20  # 0=RGB, 1=YUV422, etc.

    def __init__(self, ip):
        """Wrap a PYNQ DefaultIP handle for the Gamma LUT."""
        self._ip = ip

    def configure(
        self, width: int = 1920, height: int = 1080, bypass: bool = True,
    ) -> None:
        """Configure the Gamma LUT IP.

        Args:
            bypass: If True, leave the IP in passthrough mode (not started).
                    If False, load a linear LUT and start the IP.
        """
        if bypass:
            self._ip.write(self.WIDTH, width)
            self._ip.write(self.HEIGHT, height)
            self._ip.write(self.VIDEO_FORMAT, 0)
            self._ip.write(AP_CTRL, 0x00)
            logger.info("Gamma LUT configured: %dx%d, bypass=True", width, height)
            return

        # Load linear 1:1 LUT (identity transform) BEFORE writing control
        # registers.  Each LUT has 1024 U16 entries packed 2-per-word in
        # 512 x 32-bit words (0x800 bytes per LUT region).
        for ch_base in (0x0800, 0x1000, 0x1800):
            for word_idx in range(512):
                lo = word_idx * 2
                hi = word_idx * 2 + 1
                self._ip.write(ch_base + word_idx * 4, (hi << 16) | lo)

        # Write control registers after LUT (safe from aliasing overwrite)
        self._ip.write(self.WIDTH, width)
        self._ip.write(self.HEIGHT, height)
        self._ip.write(self.VIDEO_FORMAT, 0)  # RGB
        self._ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)
        logger.info("Gamma LUT configured: %dx%d, linear LUT loaded", width, height)

    def start(self) -> None:
        """Re-assert AP_CTRL to start/restart the Gamma LUT."""
        self._ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)

    def set_bypass(self, bypass: bool) -> None:
        """Toggle Gamma LUT bypass at runtime."""
        if bypass:
            self._ip.write(AP_CTRL, 0x00)
        else:
            self._ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)
        logger.info("Gamma LUT bypass set to %s", bypass)
