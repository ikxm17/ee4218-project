"""Video Multi-Scaler driver — polyphase scaling engine.

Reference: PG325 / Xilinx embeddedsw xv_multi_scaler_hw.h

Register map from Xilinx embeddedsw driver (NOT PG325 documentation,
which uses generic offsets that don't match the HLS-synthesized IP).

The Multi-Scaler is memory-mapped: it reads source frames from DDR
and writes scaled outputs to DDR.  Per-channel registers at
0x100 + channel * 0x200.  Buffer addresses are 64-bit (lo/hi split).

Global:
    0x000  AP_CTRL
    0x010  num_outs

Per-channel (base = 0x100 + ch * 0x200):
    +0x000 WidthIn      +0x008 WidthOut
    +0x010 HeightIn     +0x018 HeightOut
    +0x020 LineRate      +0x028 PixelRate
    +0x030 InPixelFmt   +0x038 OutPixelFmt
    +0x050 InStride     +0x058 OutStride
    +0x060 SrcImgBuf0   +0x064 SrcImgBuf0_hi
    +0x090 DstImgBuf0   +0x094 DstImgBuf0_hi

Filter coefficients (polyphase mode, SCALE_MODE=2):
    V: 0x2000 + ch * 0x2000   H: 0x2800 + ch * 0x2000
    Each table: PHASES(64) x TAPS(12) I16 entries, packed 2-per-word.
"""

import logging
import math

from ._hls_common import AP_CTRL, AP_CTRL_START, AP_CTRL_AUTO_RESTART

logger = logging.getLogger(__name__)


class MultiScalerDriver:
    """Video Multi-Scaler (PG325 / embeddedsw)."""

    # -- Global registers --
    NUM_OUTS = 0x010

    # -- Per-channel layout --
    CH_STRIDE = 0x200
    CH0_BASE = 0x100
    CH_WIDTH_IN = 0x000
    CH_WIDTH_OUT = 0x008
    CH_HEIGHT_IN = 0x010
    CH_HEIGHT_OUT = 0x018
    CH_LINE_RATE = 0x020
    CH_PIXEL_RATE = 0x028
    CH_IN_PIX_FMT = 0x030
    CH_OUT_PIX_FMT = 0x038
    CH_IN_STRIDE = 0x050
    CH_OUT_STRIDE = 0x058
    CH_SRC_BUF0_LO = 0x060
    CH_SRC_BUF0_HI = 0x064
    CH_DST_BUF0_LO = 0x090
    CH_DST_BUF0_HI = 0x094

    # -- Filter coefficient memory --
    COEFF_V_BASE = 0x2000
    COEFF_H_BASE = 0x2800
    COEFF_CH_STRIDE = 0x2000

    # -- Pixel format codes (from v_multi_scaler.h) --
    FMT_RGBX10 = 15   # 4 bytes/pixel, 10-bit per channel
    FMT_RGB8 = 20      # 3 bytes/pixel, 8-bit per channel

    # -- Scaler parameters (from v_multi_scaler_config.h / .hwh) --
    TAPS = 12
    PHASES = 64              # 2^PHASE_SHIFT = 2^6
    STEP_PRECISION = 1 << 16  # 65536
    COEFF_PRECISION = 12      # Q12 fixed-point

    def __init__(self, ip):
        """Wrap a PYNQ DefaultIP handle for the Multi-Scaler."""
        self._ip = ip

    def configure(
        self,
        src_addr: int,
        src_width: int,
        src_height: int,
        src_stride: int,
        outputs: list,
    ) -> None:
        """Configure and start the Multi-Scaler.

        Args:
            src_addr: Physical address of the source frame in DDR.
            src_width: Source frame width in pixels.
            src_height: Source frame height in pixels.
            src_stride: Source frame stride in bytes.
            outputs: List of dicts with keys:
                addr: Physical address of destination buffer
                width: Output width in pixels
                height: Output height in pixels
                stride: Output stride in bytes
        """
        self._ip.write(self.NUM_OUTS, len(outputs))

        for i, out in enumerate(outputs):
            ch = self.CH0_BASE + i * self.CH_STRIDE

            # Input dimensions (same source for all channels)
            self._ip.write(ch + self.CH_WIDTH_IN, src_width)
            self._ip.write(ch + self.CH_HEIGHT_IN, src_height)
            self._ip.write(ch + self.CH_IN_PIX_FMT, self.FMT_RGBX10)
            self._ip.write(ch + self.CH_IN_STRIDE, src_stride)

            # Source buffer (64-bit address)
            self._ip.write(ch + self.CH_SRC_BUF0_LO, src_addr & 0xFFFFFFFF)
            self._ip.write(ch + self.CH_SRC_BUF0_HI, (src_addr >> 32) & 0xFFFFFFFF)

            # Output dimensions and format
            self._ip.write(ch + self.CH_WIDTH_OUT, out["width"])
            self._ip.write(ch + self.CH_HEIGHT_OUT, out["height"])
            self._ip.write(ch + self.CH_OUT_PIX_FMT, self.FMT_RGB8)
            self._ip.write(ch + self.CH_OUT_STRIDE, out["stride"])

            # Destination buffer (64-bit address)
            dst = out["addr"]
            self._ip.write(ch + self.CH_DST_BUF0_LO, dst & 0xFFFFFFFF)
            self._ip.write(ch + self.CH_DST_BUF0_HI, (dst >> 32) & 0xFFFFFFFF)

            # Scaling rates (fixed-point step size for resampling)
            pixel_rate = (src_width * self.STEP_PRECISION) // out["width"]
            line_rate = (src_height * self.STEP_PRECISION) // out["height"]
            self._ip.write(ch + self.CH_PIXEL_RATE, pixel_rate)
            self._ip.write(ch + self.CH_LINE_RATE, line_rate)

        # Load polyphase filter coefficients
        coeffs = self._generate_lanczos_coefficients()
        self._load_coefficients(len(outputs), coeffs)

        # Start the scaler
        self._ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)

        logger.info(
            "Multi-Scaler started: src %dx%d -> %d outputs",
            src_width, src_height, len(outputs),
        )

    def _generate_lanczos_coefficients(self) -> list:
        """Generate Lanczos-3 polyphase filter coefficients.

        Returns a flat list of PHASES * TAPS = 768 signed 12-bit
        fixed-point coefficients.  Each group of TAPS coefficients
        (one phase) is normalized to sum to 4096 (1.0 in Q12).
        """
        a = self.TAPS / 4  # Lanczos-a parameter (3 for 12-tap)
        center = (self.TAPS - 1) / 2.0
        flat = []
        for phase in range(self.PHASES):
            frac = phase / self.PHASES
            taps = []
            for t in range(self.TAPS):
                x = (t - center) - frac
                if abs(x) < 1e-9:
                    val = 1.0
                elif abs(x) >= a:
                    val = 0.0
                else:
                    val = (a * math.sin(math.pi * x) * math.sin(math.pi * x / a)
                           / (math.pi * math.pi * x * x))
                taps.append(val)
            # Normalize then quantize to Q12
            s = sum(taps) or 1.0
            qtaps = [int(round(t / s * (1 << self.COEFF_PRECISION)))
                     for t in taps]
            # Adjust center tap so phase sums to exactly 4096
            qtaps[self.TAPS // 2] += (1 << self.COEFF_PRECISION) - sum(qtaps)
            qtaps = [max(-2048, min(2047, q)) for q in qtaps]
            flat.extend(qtaps)
        return flat

    def _load_coefficients(self, num_channels: int, coeffs: list) -> None:
        """Write polyphase filter coefficients to all channels.

        Coefficients are I16 values packed 2-per-32-bit-word (same packing
        as Gamma LUT entries).  The same kernel is used for both V and H
        directions and for all channels.
        """
        for ch in range(num_channels):
            v_base = self.COEFF_V_BASE + ch * self.COEFF_CH_STRIDE
            h_base = self.COEFF_H_BASE + ch * self.COEFF_CH_STRIDE
            for base in (v_base, h_base):
                for i in range(0, len(coeffs), 2):
                    lo = coeffs[i] & 0xFFFF
                    hi = coeffs[i + 1] & 0xFFFF
                    self._ip.write(base + (i // 2) * 4, (hi << 16) | lo)
