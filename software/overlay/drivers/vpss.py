"""VPSS Scaler Only driver — polyphase 6-tap/64-phase downscaler.

Reference: PG231 v2.3 (docs/ips/video/pg231-v-proc-ss-en-us-2.3.pdf)

The VPSS in Scaler Only mode (C_TOPOLOGY=0) contains three sub-IPs
behind an internal AXI interconnect, addressed as offsets within the
VPSS's 256KB AXI-Lite slave space:

    H-Scaler (v_hscaler)  at +0x00000  (64KB)
    GPIO     (axi_gpio)   at +0x10000  (64KB, 2-bit reset control)
    V-Scaler (v_vscaler)  at +0x20000  (64KB)

Data flows:  s_axis → V-Scaler → H-Scaler → m_axis  (PG231 Figure 4)

The V-Scaler processes full input dimensions and produces vertically
scaled output.  The H-Scaler then horizontally scales to the final
output width.

Register offsets below are from PG231 Tables 10-11. Run scripts/probe_vpss.py
on the board to verify these match the actual synthesis before first use.
"""

import logging
import time

from ._hls_common import AP_CTRL, AP_CTRL_START, AP_CTRL_AUTO_RESTART
from . import _vpss_coeff as coeff

logger = logging.getLogger(__name__)


class VpssScalerDriver:
    """VPSS Scaler Only (PG231, C_TOPOLOGY=0)."""

    # IP identification — audit matches by VLNV type
    IP_VLNV = "xilinx.com:ip:v_proc_ss:2.3"
    IP_NAME = "v_proc_ss_0"

    # -- Sub-IP base offsets (from .hwh, verified by probe) --
    _HSC_BASE = 0x00000
    _GPIO_BASE = 0x10000
    _VSC_BASE = 0x20000

    # -- H-Scaler registers (PG231 Table 11, relative to _HSC_BASE) --
    HSC_AP_CTRL = 0x000
    HSC_HEIGHT = 0x010
    HSC_WIDTH_IN = 0x018
    HSC_WIDTH_OUT = 0x020
    HSC_COLOR_MODE = 0x028
    HSC_PIXEL_RATE = 0x030
    HSC_COLOR_MODE_OUT = 0x038
    HSC_COEFF_BASE = 0x400
    HSC_PHASE_BASE = 0x4000

    # -- V-Scaler registers (PG231 Table 10, relative to _VSC_BASE) --
    VSC_AP_CTRL = 0x000
    VSC_HEIGHT_IN = 0x010
    VSC_WIDTH = 0x018
    VSC_HEIGHT_OUT = 0x020
    VSC_LINE_RATE = 0x028
    VSC_COLOR_MODE = 0x030
    VSC_COEFF_BASE = 0x800

    # -- GPIO registers (AXI GPIO, relative to _GPIO_BASE) --
    GPIO_DATA = 0x000
    GPIO_TRI = 0x004

    # -- Constants --
    NUM_TAPS = 6
    NUM_PHASES = 64
    STEP_PRECISION = 65536  # 2^16

    # Color mode values (PG231 Table 19)
    COLOR_RGB = 0
    COLOR_YUV444 = 1
    COLOR_YUV422 = 2
    COLOR_YUV420 = 3

    def __init__(self, ip):
        """Wrap a PYNQ DefaultIP or MMIO handle for the VPSS.

        Args:
            ip: PYNQ IP handle with read(offset)/write(offset, value) interface.
        """
        self._ip = ip

    # -- Internal register access helpers --

    def _hsc_write(self, offset, value):
        self._ip.write(self._HSC_BASE + offset, value)

    def _hsc_read(self, offset):
        return self._ip.read(self._HSC_BASE + offset)

    def _vsc_write(self, offset, value):
        self._ip.write(self._VSC_BASE + offset, value)

    def _vsc_read(self, offset):
        return self._ip.read(self._VSC_BASE + offset)

    def _gpio_write(self, offset, value):
        self._ip.write(self._GPIO_BASE + offset, value)

    def _gpio_read(self, offset):
        return self._ip.read(self._GPIO_BASE + offset)

    # -- Reset --

    def _reset(self):
        """Assert and deassert the internal VPSS reset via GPIO.

        PG231 p.54: "For Full-fledged and Scaler Only modes along with
        aresetn_ctrl signal, there is an internal gpio reset signal to
        reset the sub cores of vpss."

        GPIO bit 0: aresetn_io_axis (stream-side reset, active low)
        GPIO bit 1: scaler internal reset (active low)

        Both must be asserted (driven low) then released for a clean reset.
        """
        self._gpio_write(self.GPIO_TRI, 0x00)   # both bits as outputs
        self._gpio_write(self.GPIO_DATA, 0x00)   # assert reset (active low)
        time.sleep(0.002)                         # hold for 2ms
        self._gpio_write(self.GPIO_DATA, 0x03)   # deassert reset
        time.sleep(0.002)
        logger.debug("VPSS internal reset complete")

    # -- Coefficient loading --

    def _load_v_coefficients(self):
        """Load Lanczos 6-tap/64-phase coefficients into V-Scaler."""
        words = coeff.pack_coefficients(
            coeff.LANCZOS_6TAP_64PHASE, self.NUM_TAPS, self.NUM_PHASES
        )
        for off, word in words:
            self._vsc_write(self.VSC_COEFF_BASE + off, word)
        logger.debug("V-Scaler: %d coefficient words loaded", len(words))

    def _load_h_coefficients(self):
        """Load Lanczos 6-tap/64-phase coefficients into H-Scaler."""
        words = coeff.pack_coefficients(
            coeff.LANCZOS_6TAP_64PHASE, self.NUM_TAPS, self.NUM_PHASES
        )
        for off, word in words:
            self._hsc_write(self.HSC_COEFF_BASE + off, word)
        logger.debug("H-Scaler: %d coefficient words loaded", len(words))

    def _load_h_phases(self, width_in: int, width_out: int):
        """Calculate and load phase array into H-Scaler."""
        phases = coeff.calculate_phases(width_in, width_out, self.NUM_PHASES)
        words = coeff.pack_phases(phases)
        for off, word in words:
            self._hsc_write(self.HSC_PHASE_BASE + off, word)
        logger.debug(
            "H-Scaler: %d phase words loaded for %d->%d",
            len(words), width_in, width_out,
        )

    # -- Public API --

    def configure(
        self,
        width_in: int = 1920,
        height_in: int = 1080,
        width_out: int = 224,
        height_out: int = 224,
        color_mode: int = 0,
    ) -> None:
        """Configure the VPSS scaler for the given input/output dimensions.

        Initialization sequence (PG231 Chapter 4, Appendix C):
            1. Assert internal GPIO reset
            2. Program V-Scaler dimensions, line rate, color mode
            3. Program H-Scaler dimensions, pixel rate, color mode
            4. Load V-Scaler filter coefficients
            5. Load H-Scaler filter coefficients
            6. Load H-Scaler phase data
            7. Start both scalers (AP_CTRL with auto-restart)

        Note: 6-tap filters are recommended for up to 1.5x downscale.
        Our typical use is ~7.5x (1920->224), which produces aliasing
        (PG231 Table 18 recommends 12-tap for >3.5x), but this is
        acceptable for detection network input.

        Args:
            width_in: Input width (pixels), max 1920.
            height_in: Input height (lines), max 1080.
            width_out: Output width (pixels).
            height_out: Output height (lines).
            color_mode: 0=RGB, 1=YUV444, 2=YUV422, 3=YUV420.
        """
        # 1. Reset
        self._reset()

        # 2. Compute rate parameters (PG231 p.39)
        line_rate = (height_in * self.STEP_PRECISION) // height_out
        pixel_rate = (width_in * self.STEP_PRECISION) // width_out

        # 3. Configure V-Scaler (sees full input, produces vertically scaled output)
        self._vsc_write(self.VSC_HEIGHT_IN, height_in)
        self._vsc_write(self.VSC_WIDTH, width_in)
        self._vsc_write(self.VSC_HEIGHT_OUT, height_out)
        self._vsc_write(self.VSC_LINE_RATE, line_rate)
        self._vsc_write(self.VSC_COLOR_MODE, color_mode)

        # 4. Configure H-Scaler (sees V-Scaler output: height=height_out, width=width_in)
        self._hsc_write(self.HSC_HEIGHT, height_out)
        self._hsc_write(self.HSC_WIDTH_IN, width_in)
        self._hsc_write(self.HSC_WIDTH_OUT, width_out)
        self._hsc_write(self.HSC_PIXEL_RATE, pixel_rate)
        self._hsc_write(self.HSC_COLOR_MODE, color_mode)
        self._hsc_write(self.HSC_COLOR_MODE_OUT, color_mode)

        # 5. Load filter coefficients (same Lanczos table for both)
        self._load_v_coefficients()
        self._load_h_coefficients()

        # 6. Load H-Scaler phase data
        self._load_h_phases(width_in, width_out)

        # 7. Start both scalers with auto-restart
        self._vsc_write(self.VSC_AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)
        self._hsc_write(self.HSC_AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)

        logger.info(
            "VPSS configured: %dx%d -> %dx%d (line_rate=0x%X, pixel_rate=0x%X)",
            width_in, height_in, width_out, height_out, line_rate, pixel_rate,
        )

    def start(self) -> None:
        """Re-assert AP_CTRL on both scalers (after pipeline stall recovery)."""
        self._vsc_write(self.VSC_AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)
        self._hsc_write(self.HSC_AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)

    def stop(self) -> None:
        """Stop both scalers by clearing AP_CTRL."""
        self._hsc_write(self.HSC_AP_CTRL, 0x00)
        self._vsc_write(self.VSC_AP_CTRL, 0x00)

    def read_status(self) -> dict:
        """Read diagnostic registers from both scalers and GPIO."""
        hsc_ctrl = self._hsc_read(self.HSC_AP_CTRL)
        vsc_ctrl = self._vsc_read(self.VSC_AP_CTRL)
        gpio = self._gpio_read(self.GPIO_DATA)
        return {
            "hsc_ap_ctrl": hsc_ctrl,
            "hsc_started": bool(hsc_ctrl & 0x01),
            "hsc_done": bool(hsc_ctrl & 0x02),
            "hsc_idle": bool(hsc_ctrl & 0x04),
            "hsc_ready": bool(hsc_ctrl & 0x08),
            "hsc_auto_restart": bool(hsc_ctrl & 0x80),
            "vsc_ap_ctrl": vsc_ctrl,
            "vsc_started": bool(vsc_ctrl & 0x01),
            "vsc_done": bool(vsc_ctrl & 0x02),
            "vsc_idle": bool(vsc_ctrl & 0x04),
            "vsc_ready": bool(vsc_ctrl & 0x08),
            "vsc_auto_restart": bool(vsc_ctrl & 0x80),
            "gpio_data": gpio,
            "gpio_reset_deasserted": bool(gpio & 0x03 == 0x03),
        }
