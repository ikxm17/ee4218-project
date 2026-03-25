"""PL IP runtime configuration via AXI4-Lite MMIO registers.

Each function takes a PYNQ IP object (from overlay.ip_dict) and writes
the registers needed to configure and start the IP.  Register offsets
are from the respective Xilinx Product Guides:

    - CSI-2 RX:     PG232
    - Demosaic:     PG286
    - Gamma LUT:    PG285
    - VDMA:         PG020
    - Multi-Scaler: PG325

IP objects provide ip.write(offset, value) and ip.read(offset) backed
by PYNQ's MMIO, with the base address resolved from the .hwh file.
"""

import logging
import math
import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HLS IP common registers (AP_CTRL)
# ---------------------------------------------------------------------------
AP_CTRL = 0x00
AP_CTRL_START = 0x01
AP_CTRL_AUTO_RESTART = 0x80


# ---------------------------------------------------------------------------
# MIPI CSI-2 RX Subsystem (PG232)
# ---------------------------------------------------------------------------
# Register map (from PG232 Table 2-4):
#   0x00  Core Configuration     Bit 0: Enable, Bit 1: Soft Reset
#   0x04  Protocol Configuration  Bit [4:3]: Active lanes
#   0x10  Core Status             Bit [31:16]: Packet count
#   0x24  Interrupt Status (W1C)  Stop state, SoT errors, frame events
#   0x3C  Clock Lane Info         Bit 1: Stop state, Bit 0: HS mode
#   0x40  Data Lane 0 Info        Bit 1: Stop state, Bit 0: HS mode
#   0x44  Data Lane 1 Info        Same as Lane 0

CSI2_CORE_CONFIG = 0x00
CSI2_PROTOCOL_CONFIG = 0x04
CSI2_CORE_STATUS = 0x10
CSI2_ISR = 0x24
CSI2_CLK_LANE_INFO = 0x3C
CSI2_LANE0_INFO = 0x40
CSI2_LANE1_INFO = 0x44


def reset_csi2_rx(ip) -> None:
    """Perform a soft reset of the CSI-2 RX core (PG232 Sec 2.3).

    The D-PHY receiver must see the sensor's clock lane toggling before
    it can achieve byte-level synchronization.  If the core was enabled
    at overlay load time (before the sensor started streaming), the
    D-PHY may have missed the initial LP→HS transition.  A soft reset
    clears the lane state machines and forces a fresh sync attempt.
    """
    ip.write(CSI2_CORE_CONFIG, 0x00)          # Disable core
    ip.write(CSI2_CORE_CONFIG, 0x02)          # Assert soft reset
    time.sleep(0.001)                          # Hold reset 1 ms
    ip.write(CSI2_CORE_CONFIG, 0x00)          # Deassert reset
    ip.write(CSI2_ISR, 0xFFFFFFFF)            # Clear all ISR bits
    ip.write(CSI2_CORE_CONFIG, 0x01)          # Re-enable core
    logger.info("CSI-2 RX soft reset complete, core re-enabled")


def read_csi2_status(ip) -> dict:
    """Read and decode CSI-2 RX diagnostic registers."""
    core_cfg = ip.read(CSI2_CORE_CONFIG)
    proto_cfg = ip.read(CSI2_PROTOCOL_CONFIG)
    core_status = ip.read(CSI2_CORE_STATUS)
    isr = ip.read(CSI2_ISR)
    clk_info = ip.read(CSI2_CLK_LANE_INFO)
    lane0_info = ip.read(CSI2_LANE0_INFO)
    lane1_info = ip.read(CSI2_LANE1_INFO)

    pkt_count = (core_status >> 16) & 0xFFFF

    status = {
        "core_enabled": bool(core_cfg & 0x01),
        "active_lanes": ((proto_cfg >> 3) & 0x03) + 1,
        "packet_count": pkt_count,
        "isr_raw": isr,
        "isr_stop_state": bool(isr & (1 << 17)),
        "isr_sot_error": bool(isr & (1 << 13)),
        "isr_sot_sync_error": bool(isr & (1 << 12)),
        "clk_lane_hs": bool(clk_info & 0x01),
        "clk_lane_stop": bool(clk_info & 0x02),
        "lane0_hs": bool(lane0_info & 0x01),
        "lane0_stop": bool(lane0_info & 0x02),
        "lane1_hs": bool(lane1_info & 0x01),
        "lane1_stop": bool(lane1_info & 0x02),
    }

    logger.info(
        "CSI-2 RX status: packets=%d, ISR=0x%08X, "
        "clk_hs=%s, d0_hs=%s, d1_hs=%s",
        pkt_count, isr,
        status["clk_lane_hs"], status["lane0_hs"], status["lane1_hs"],
    )
    return status


def wait_for_csi2_lock(ip, timeout_s: float = 2.0) -> bool:
    """Poll CSI-2 RX until packets are received or timeout expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        core_status = ip.read(CSI2_CORE_STATUS)
        pkt_count = (core_status >> 16) & 0xFFFF
        if pkt_count > 0:
            logger.info("CSI-2 RX locked: %d packets received", pkt_count)
            return True
        time.sleep(0.010)
    logger.warning("CSI-2 RX lock timeout: no packets after %.1fs", timeout_s)
    return False


# ---------------------------------------------------------------------------
# Sensor Demosaic (PG286)
# ---------------------------------------------------------------------------
# Register map (from PG286 Table 2-6):
#   0x00  AP_CTRL
#   0x10  width
#   0x18  height
#   0x28  bayer_phase  (0=RGGB, 1=GRBG, 2=GBRG, 3=BGGR)

DEMOSAIC_WIDTH = 0x10
DEMOSAIC_HEIGHT = 0x18
DEMOSAIC_BAYER_PHASE = 0x28


def configure_demosaic(ip, width: int = 1920, height: int = 1080) -> None:
    """Configure and start the Sensor Demosaic IP.

    IMX219 outputs RGGB Bayer pattern (bayer_phase = 0).
    """
    ip.write(DEMOSAIC_WIDTH, width)
    ip.write(DEMOSAIC_HEIGHT, height)
    ip.write(DEMOSAIC_BAYER_PHASE, 0)  # RGGB
    ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)
    logger.info("Demosaic configured: %dx%d, RGGB", width, height)


# ---------------------------------------------------------------------------
# Gamma LUT (PG285)
# ---------------------------------------------------------------------------
# Register map (from PG285 / xv_gamma_lut_hw.h):
#   0x00   AP_CTRL
#   0x10   width
#   0x18   height
#   0x20   video_format
#   0x0800 gamma_lut_0[0..1023]  (R channel, 2 x U16 packed per 32-bit word)
#   0x1000 gamma_lut_1[0..1023]  (G channel, 512 words = 0x800 bytes per LUT)
#   0x1800 gamma_lut_2[0..1023]  (B channel)
#
# ADDR_WIDTH=13 → address space is 0x0000-0x1FFF (8 KB).  Each LUT
# region is 0x800 bytes.  Writing at 4-byte stride (1 entry/word)
# overflows LUT2 past 0x1FFF, aliasing back to 0x0000 and
# corrupting the control registers.  Always use 2-entry packing.
#
# Bypass: set ap_start=0 with auto_restart=0.  Data passes through
# because the IP is wired inline on the AXI4-Stream path and uses
# a pass-through mode when not started.

GAMMA_WIDTH = 0x10
GAMMA_HEIGHT = 0x18
GAMMA_VIDEO_FORMAT = 0x20  # 0=RGB, 1=YUV422, etc.


def configure_gamma_lut(
    ip, width: int = 1920, height: int = 1080, bypass: bool = True,
) -> None:
    """Configure the Gamma LUT IP.

    Args:
        bypass: If True, leave the IP in passthrough mode (not started).
                If False, load a linear LUT and start the IP.
    """
    if bypass:
        ip.write(GAMMA_WIDTH, width)
        ip.write(GAMMA_HEIGHT, height)
        ip.write(GAMMA_VIDEO_FORMAT, 0)
        ip.write(AP_CTRL, 0x00)
        logger.info("Gamma LUT configured: %dx%d, bypass=True", width, height)
        return

    # Load linear 1:1 LUT (identity transform) BEFORE writing control
    # registers.  Each LUT has 1024 U16 entries packed 2-per-word in
    # 512 x 32-bit words (0x800 bytes per LUT region).
    for ch_base in (0x0800, 0x1000, 0x1800):
        for word_idx in range(512):
            lo = word_idx * 2
            hi = word_idx * 2 + 1
            ip.write(ch_base + word_idx * 4, (hi << 16) | lo)

    # Write control registers after LUT (safe from aliasing overwrite)
    ip.write(GAMMA_WIDTH, width)
    ip.write(GAMMA_HEIGHT, height)
    ip.write(GAMMA_VIDEO_FORMAT, 0)  # RGB
    ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)
    logger.info("Gamma LUT configured: %dx%d, linear LUT loaded", width, height)


def set_gamma_bypass(ip, bypass: bool) -> None:
    """Toggle Gamma LUT bypass at runtime."""
    if bypass:
        ip.write(AP_CTRL, 0x00)
    else:
        ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)
    logger.info("Gamma LUT bypass set to %s", bypass)


# ---------------------------------------------------------------------------
# AXI VDMA — S2MM (write) channel only (PG020)
# ---------------------------------------------------------------------------
# Register map (from PG020 Table 2-16, S2MM registers):
#   0x30  S2MM_DMACR        Control register
#   0x34  S2MM_DMASR        Status register
#   0xA0  S2MM_VSIZE         Vertical size (writing this starts the channel)
#   0xA4  S2MM_HSIZE         Horizontal size in bytes
#   0xA8  S2MM_FRMDLY_STRIDE Frame delay + stride
#   0xAC  S2MM_START_ADDRESS1  Frame store 1 base address
#   0xB0  S2MM_START_ADDRESS2  Frame store 2 base address
#   0xB4  S2MM_START_ADDRESS3  Frame store 3 base address
#
# S2MM_DMACR bits:
#   [0]    RS (Run/Stop)
#   [1]    Circular mode (1 = circular)
#   [2]    Reset
#   [4]    Frame count enable
#   [16]   IRQ on complete

S2MM_DMACR = 0x30
S2MM_DMASR = 0x34
S2MM_FRMSTORE = 0x48  # Number of frame stores (1-32)
S2MM_VSIZE = 0xA0
S2MM_HSIZE = 0xA4
S2MM_FRMDLY_STRIDE = 0xA8
S2MM_START_ADDR_BASE = 0xAC  # +0x04 per frame store


def configure_vdma_s2mm(
    ip,
    frame_addrs: list,
    width_bytes: int,
    height: int,
    stride: int,
) -> None:
    """Configure and start the VDMA S2MM (write) channel.

    The channel starts when VSIZE is written (PG020: "Writing to the
    vertical size register starts the channel").  It then waits for
    the first SOF (tuser[0]) from the CSI-2 RX before writing.

    Args:
        frame_addrs: List of physical addresses for frame stores (3 for triple-buffer).
        width_bytes: Horizontal frame size in bytes (width * bytes_per_pixel).
        height: Number of lines per frame.
        stride: Bytes per line including padding.
    """
    # Reset the channel
    ip.write(S2MM_DMACR, 0x04)
    while ip.read(S2MM_DMACR) & 0x04:
        pass  # Wait for reset to clear

    # Set number of frame stores
    ip.write(S2MM_FRMSTORE, len(frame_addrs))

    # Set frame store addresses
    for i, addr in enumerate(frame_addrs):
        ip.write(S2MM_START_ADDR_BASE + i * 4, addr)

    # Set stride (lower 16 bits) and frame delay (upper 16 bits = 0)
    ip.write(S2MM_FRMDLY_STRIDE, stride & 0xFFFF)

    # Set horizontal size in bytes
    ip.write(S2MM_HSIZE, width_bytes)

    # Clear all W1C error bits in DMASR (bits 4-8, 11-14)
    ip.write(S2MM_DMASR, 0x000079F0)

    # Run in circular mode (continuous frame capture)
    ip.write(S2MM_DMACR, 0x03)  # RS=1, Circular=1

    # Writing VSIZE starts the channel — it waits for SOF to sync
    ip.write(S2MM_VSIZE, height)

    logger.info(
        "VDMA S2MM started: %d stores, %dx%d, stride=%d",
        len(frame_addrs), width_bytes, height, stride,
    )


def read_vdma_status(ip) -> dict:
    """Read VDMA S2MM status register for debugging.

    Bit positions from xaxivdma_hw.h / PG020 Table 2-19 (S2MM_DMASR).
    """
    sr = ip.read(S2MM_DMASR)
    status = {
        "raw": sr,
        "halted": bool(sr & 0x01),
        "idle": bool(sr & 0x02),
        "err_dma_int": bool(sr & (1 << 4)),
        "err_dma_slv": bool(sr & (1 << 5)),
        "err_dma_dec": bool(sr & (1 << 6)),
        "err_fsz_less": bool(sr & (1 << 7)),
        "err_lsz_less": bool(sr & (1 << 8)),
        "err_eol_early": bool(sr & (1 << 11)),
        "err_sof_early": bool(sr & (1 << 12)),
        "err_eol_late": bool(sr & (1 << 13)),
        "err_sof_late": bool(sr & (1 << 14)),
        "frame_count": (sr >> 16) & 0xFF,
    }
    logger.info(
        "VDMA S2MM status: 0x%08X frames=%d errs=%s",
        sr, status["frame_count"],
        [k for k, v in status.items() if k.startswith("err_") and v],
    )
    return status


# ---------------------------------------------------------------------------
# Video Multi-Scaler (PG325 / xv_multi_scaler_hw.h)
# ---------------------------------------------------------------------------
# Register map from Xilinx embeddedsw driver (NOT PG325 documentation,
# which uses generic offsets that don't match the HLS-synthesized IP).
#
# The Multi-Scaler is memory-mapped: it reads source frames from DDR
# and writes scaled outputs to DDR.  Per-channel registers at
# 0x100 + channel * 0x200.  Buffer addresses are 64-bit (lo/hi split).
#
# Global:
#   0x000  AP_CTRL
#   0x010  num_outs
#
# Per-channel (base = 0x100 + ch * 0x200):
#   +0x000 WidthIn      +0x008 WidthOut
#   +0x010 HeightIn     +0x018 HeightOut
#   +0x020 LineRate      +0x028 PixelRate
#   +0x030 InPixelFmt   +0x038 OutPixelFmt
#   +0x050 InStride     +0x058 OutStride
#   +0x060 SrcImgBuf0   +0x064 SrcImgBuf0_hi
#   +0x090 DstImgBuf0   +0x094 DstImgBuf0_hi
#
# Filter coefficients (polyphase mode, SCALE_MODE=2):
#   V: 0x2000 + ch * 0x2000   H: 0x2800 + ch * 0x2000
#   Each table: PHASES(64) x TAPS(12) I16 entries, packed 2-per-word.

MSCALER_NUM_OUTS = 0x010

# Per-channel layout
MSCALER_CH_STRIDE = 0x200
MSCALER_CH0_BASE = 0x100
MSCALER_CH_WIDTH_IN = 0x000
MSCALER_CH_WIDTH_OUT = 0x008
MSCALER_CH_HEIGHT_IN = 0x010
MSCALER_CH_HEIGHT_OUT = 0x018
MSCALER_CH_LINE_RATE = 0x020
MSCALER_CH_PIXEL_RATE = 0x028
MSCALER_CH_IN_PIX_FMT = 0x030
MSCALER_CH_OUT_PIX_FMT = 0x038
MSCALER_CH_IN_STRIDE = 0x050
MSCALER_CH_OUT_STRIDE = 0x058
MSCALER_CH_SRC_BUF0_LO = 0x060
MSCALER_CH_SRC_BUF0_HI = 0x064
MSCALER_CH_DST_BUF0_LO = 0x090
MSCALER_CH_DST_BUF0_HI = 0x094

# Filter coefficient memory bases
MSCALER_COEFF_V_BASE = 0x2000
MSCALER_COEFF_H_BASE = 0x2800
MSCALER_COEFF_CH_STRIDE = 0x2000

# Pixel format codes (from v_multi_scaler.h)
MSCALER_FMT_RGBX10 = 15  # 4 bytes/pixel, 10-bit per channel
MSCALER_FMT_RGB8 = 20     # 3 bytes/pixel, 8-bit per channel

# Scaler parameters (from v_multi_scaler_config.h / .hwh)
MSCALER_TAPS = 12
MSCALER_PHASES = 64         # 2^PHASE_SHIFT = 2^6
MSCALER_STEP_PRECISION = 1 << 16  # 65536
MSCALER_COEFF_PRECISION = 12      # Q12 fixed-point


def _generate_lanczos_coefficients() -> list:
    """Generate Lanczos-3 polyphase filter coefficients.

    Returns a flat list of PHASES * TAPS = 768 signed 12-bit
    fixed-point coefficients.  Each group of TAPS coefficients
    (one phase) is normalized to sum to 4096 (1.0 in Q12).
    """
    a = MSCALER_TAPS / 4  # Lanczos-a parameter (3 for 12-tap)
    center = (MSCALER_TAPS - 1) / 2.0
    flat = []
    for phase in range(MSCALER_PHASES):
        frac = phase / MSCALER_PHASES
        taps = []
        for t in range(MSCALER_TAPS):
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
        qtaps = [int(round(t / s * (1 << MSCALER_COEFF_PRECISION)))
                 for t in taps]
        # Adjust center tap so phase sums to exactly 4096
        qtaps[MSCALER_TAPS // 2] += (1 << MSCALER_COEFF_PRECISION) - sum(qtaps)
        qtaps = [max(-2048, min(2047, q)) for q in qtaps]
        flat.extend(qtaps)
    return flat


def _load_scaler_coefficients(ip, num_channels: int, coeffs: list) -> None:
    """Write polyphase filter coefficients to all channels.

    Coefficients are I16 values packed 2-per-32-bit-word (same packing
    as Gamma LUT entries).  The same kernel is used for both V and H
    directions and for all channels.
    """
    for ch in range(num_channels):
        v_base = MSCALER_COEFF_V_BASE + ch * MSCALER_COEFF_CH_STRIDE
        h_base = MSCALER_COEFF_H_BASE + ch * MSCALER_COEFF_CH_STRIDE
        for base in (v_base, h_base):
            for i in range(0, len(coeffs), 2):
                lo = coeffs[i] & 0xFFFF
                hi = coeffs[i + 1] & 0xFFFF
                ip.write(base + (i // 2) * 4, (hi << 16) | lo)


def configure_multi_scaler(
    ip,
    src_addr: int,
    src_width: int,
    src_height: int,
    src_stride: int,
    outputs: list,
) -> None:
    """Configure and start the Multi-Scaler.

    Register offsets from Xilinx embeddedsw xv_multi_scaler_hw.h.
    Per-channel registers at 0x100 + N * 0x200.  64-bit buffer
    addresses split into lo/hi 32-bit writes.  Polyphase mode
    requires Lanczos filter coefficients.

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
    ip.write(MSCALER_NUM_OUTS, len(outputs))

    for i, out in enumerate(outputs):
        ch = MSCALER_CH0_BASE + i * MSCALER_CH_STRIDE

        # Input dimensions (same source for all channels)
        ip.write(ch + MSCALER_CH_WIDTH_IN, src_width)
        ip.write(ch + MSCALER_CH_HEIGHT_IN, src_height)
        ip.write(ch + MSCALER_CH_IN_PIX_FMT, MSCALER_FMT_RGBX10)
        ip.write(ch + MSCALER_CH_IN_STRIDE, src_stride)

        # Source buffer (64-bit address)
        ip.write(ch + MSCALER_CH_SRC_BUF0_LO, src_addr & 0xFFFFFFFF)
        ip.write(ch + MSCALER_CH_SRC_BUF0_HI, (src_addr >> 32) & 0xFFFFFFFF)

        # Output dimensions and format
        ip.write(ch + MSCALER_CH_WIDTH_OUT, out["width"])
        ip.write(ch + MSCALER_CH_HEIGHT_OUT, out["height"])
        ip.write(ch + MSCALER_CH_OUT_PIX_FMT, MSCALER_FMT_RGB8)
        ip.write(ch + MSCALER_CH_OUT_STRIDE, out["stride"])

        # Destination buffer (64-bit address)
        dst = out["addr"]
        ip.write(ch + MSCALER_CH_DST_BUF0_LO, dst & 0xFFFFFFFF)
        ip.write(ch + MSCALER_CH_DST_BUF0_HI, (dst >> 32) & 0xFFFFFFFF)

        # Scaling rates (fixed-point step size for resampling)
        pixel_rate = (src_width * MSCALER_STEP_PRECISION) // out["width"]
        line_rate = (src_height * MSCALER_STEP_PRECISION) // out["height"]
        ip.write(ch + MSCALER_CH_PIXEL_RATE, pixel_rate)
        ip.write(ch + MSCALER_CH_LINE_RATE, line_rate)

    # Load polyphase filter coefficients
    coeffs = _generate_lanczos_coefficients()
    _load_scaler_coefficients(ip, len(outputs), coeffs)

    # Start the scaler
    ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)

    logger.info(
        "Multi-Scaler started: src %dx%d -> %d outputs",
        src_width, src_height, len(outputs),
    )
