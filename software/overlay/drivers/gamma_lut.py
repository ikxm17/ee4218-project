"""Gamma LUT driver — RGB lookup-table correction.

Reference: PG285 (docs/ips/video/pg285-v-gamma-lut.pdf)

Register map (verified against .hwh and xv_gamma_lut_hw.h):
    0x0000  AP_CTRL      (via _hls_common)
    0x0004  GIE          (via _hls_common)
    0x0008  IER          (via _hls_common)
    0x000C  ISR          (via _hls_common)
    0x0010  width        (16-bit, active pixels per scanline)
    0x0018  height       (16-bit, active lines per frame)
    0x0020  video_format (0=RGB, 1=YUV 4:4:4)
    0x0800  gamma_lut_0  (R channel, 1024 entries packed 2×U16 per 32-bit word)
    0x1000  gamma_lut_1  (G channel, 512 words = 0x800 bytes per LUT)
    0x1800  gamma_lut_2  (B channel)

ADDR_WIDTH=13 -> address space is 0x0000-0x1FFF (8 KB).  Each LUT
region is 0x800 bytes.  Writing at 4-byte stride (1 entry/word)
overflows LUT2 past 0x1FFF, aliasing back to 0x0000 and
corrupting the control registers.  Always use 2-entry packing.

Bypass: set ap_start=0 with auto_restart=0.  Data passes through
because the IP is wired inline on the AXI4-Stream path and uses
a pass-through mode when not started.
"""

import logging
from typing import Sequence

from ._hls_common import (
    AP_CTRL,
    AP_CTRL_START,
    AP_CTRL_AUTO_RESTART,
    hls_read_status,
    hls_stop,
)

logger = logging.getLogger(__name__)


class GammaLutDriver:
    """Gamma LUT IP (PG285)."""

    # IP identification (for audit — class covers ALL instances of this type)
    IP_VLNV = "xilinx.com:ip:v_gamma_lut:1.1"
    IP_NAME = "v_gamma_lut_0"  # primary instance

    # -- Register map --
    WIDTH = 0x10
    HEIGHT = 0x18
    VIDEO_FORMAT = 0x20  # 0=RGB, 1=YUV 4:4:4

    # -- LUT memory regions (MAX_DATA_WIDTH=10 → 1024 entries per channel) --
    LUT_R = 0x0800
    LUT_G = 0x1000
    LUT_B = 0x1800
    LUT_ENTRIES = 1024   # 2^MAX_DATA_WIDTH
    LUT_WORDS = 512      # LUT_ENTRIES / 2 (packed 2×U16 per 32-bit word)

    def __init__(self, ip):
        """Wrap a PYNQ DefaultIP handle for the Gamma LUT."""
        self._ip = ip

    def configure(
        self, width: int = 1920, height: int = 1080, bypass: bool = True,
    ) -> None:
        """Configure the Gamma LUT IP.

        Args:
            width: Active pixels per scanline.
            height: Active lines per frame.
            bypass: If True, leave the IP in passthrough mode (not started).
                    If False, load a linear (identity) LUT and start the IP.
        """
        if bypass:
            self._ip.write(self.WIDTH, width)
            self._ip.write(self.HEIGHT, height)
            self._ip.write(self.VIDEO_FORMAT, 0)
            self._ip.write(AP_CTRL, 0x00)
            logger.info("Gamma LUT configured: %dx%d, bypass=True", width, height)
            return

        # Load linear 1:1 LUT (identity transform) for all three channels
        identity = list(range(self.LUT_ENTRIES))
        self.load_lut(identity, identity, identity)

        # Write control registers after LUT (safe from aliasing overwrite)
        self._ip.write(self.WIDTH, width)
        self._ip.write(self.HEIGHT, height)
        self._ip.write(self.VIDEO_FORMAT, 0)  # RGB
        self._ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)
        logger.info("Gamma LUT configured: %dx%d, linear LUT loaded", width, height)

    def load_lut(
        self,
        r: Sequence[int],
        g: Sequence[int],
        b: Sequence[int],
    ) -> None:
        """Load gamma correction LUTs for all three channels.

        Each channel expects 1024 uint16 values (10-bit output range).
        Entries are packed 2 per 32-bit word: bits[15:0]=entry[2n],
        bits[31:16]=entry[2n+1].

        Args:
            r: Red channel LUT (1024 entries).
            g: Green channel LUT (1024 entries).
            b: Blue channel LUT (1024 entries).
        """
        for ch_base, lut in ((self.LUT_R, r), (self.LUT_G, g), (self.LUT_B, b)):
            for word_idx in range(self.LUT_WORDS):
                lo = lut[word_idx * 2] & 0xFFFF
                hi = lut[word_idx * 2 + 1] & 0xFFFF
                self._ip.write(ch_base + word_idx * 4, (hi << 16) | lo)

    def start(self) -> None:
        """Re-assert AP_CTRL to start/restart the Gamma LUT."""
        self._ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)

    def stop(self) -> None:
        """Clear AP_CTRL to halt the Gamma LUT (passthrough mode)."""
        hls_stop(self._ip)

    def set_bypass(self, bypass: bool) -> None:
        """Toggle Gamma LUT bypass at runtime."""
        if bypass:
            self._ip.write(AP_CTRL, 0x00)
        else:
            self._ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)
        logger.info("Gamma LUT bypass set to %s", bypass)

    def read_status(self) -> dict:
        """Read and decode AP_CTRL status bits.

        Returns:
            {"idle": bool, "done": bool, "ready": bool,
             "running": bool, "auto_restart": bool, "ap_ctrl_raw": int}
        """
        return hls_read_status(self._ip)
