"""PL IP runtime configuration via AXI4-Lite MMIO registers.

Each function takes a PYNQ IP object (from overlay.ip_dict) and writes
the registers needed to configure and start the IP.  Register offsets
are from the respective Xilinx Product Guides:

    - Demosaic:     PG286
    - Gamma LUT:    PG285
    - VDMA:         PG020
    - Multi-Scaler: PG325

IP objects provide ip.write(offset, value) and ip.read(offset) backed
by PYNQ's MMIO, with the base address resolved from the .hwh file.
"""

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HLS IP common registers (AP_CTRL)
# ---------------------------------------------------------------------------
AP_CTRL = 0x00
AP_CTRL_START = 0x01
AP_CTRL_AUTO_RESTART = 0x80


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
# Register map (from PG285 Table 2-5):
#   0x00   AP_CTRL
#   0x10   width
#   0x18   height
#   0x20   video_format (ignored for passthrough)
#   0x0800 gamma_lut_0[0..1023]  (R channel LUT)
#   0x1000 gamma_lut_1[0..1023]  (G channel LUT)
#   0x1800 gamma_lut_2[0..1023]  (B channel LUT)
#
# Bypass: set ap_start=0 with auto_restart=0.  Data passes through
# because the IP is wired inline on the AXI4-Stream path and uses
# a pass-through mode when not started.

GAMMA_WIDTH = 0x10
GAMMA_HEIGHT = 0x18


def configure_gamma_lut(
    ip, width: int = 1920, height: int = 1080, bypass: bool = True,
) -> None:
    """Configure the Gamma LUT IP.

    Args:
        bypass: If True, leave the IP in passthrough mode (not started).
                If False, load a linear LUT and start the IP.
    """
    ip.write(GAMMA_WIDTH, width)
    ip.write(GAMMA_HEIGHT, height)

    if bypass:
        # Don't start the IP — data passes through unmodified
        ip.write(AP_CTRL, 0x00)
        logger.info("Gamma LUT configured: %dx%d, bypass=True", width, height)
    else:
        # Load linear 1:1 LUT (identity transform) for all 3 channels
        # LUT format: 1024 entries of 10-bit values (matching max data width)
        for ch_offset in (0x0800, 0x1000, 0x1800):
            for i in range(1024):
                ip.write(ch_offset + i * 4, i)
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

    # Set frame store addresses
    for i, addr in enumerate(frame_addrs):
        ip.write(S2MM_START_ADDR_BASE + i * 4, addr)

    # Set stride (lower 16 bits) and frame delay (upper 16 bits = 0)
    ip.write(S2MM_FRMDLY_STRIDE, stride & 0xFFFF)

    # Set horizontal size in bytes
    ip.write(S2MM_HSIZE, width_bytes)

    # Run in circular mode (continuous frame capture)
    ip.write(S2MM_DMACR, 0x03)  # RS=1, Circular=1

    # Writing VSIZE starts the channel — it waits for SOF to sync
    ip.write(S2MM_VSIZE, height)

    logger.info(
        "VDMA S2MM started: %d stores, %dx%d, stride=%d",
        len(frame_addrs), width_bytes, height, stride,
    )


def read_vdma_status(ip) -> dict:
    """Read VDMA S2MM status register for debugging."""
    sr = ip.read(S2MM_DMASR)
    return {
        "halted": bool(sr & 0x01),
        "idle": bool(sr & 0x02),
        "err_internal": bool(sr & (1 << 4)),
        "err_slave": bool(sr & (1 << 5)),
        "err_decode": bool(sr & (1 << 6)),
        "err_sof_early": bool(sr & (1 << 7)),
        "err_sof_late": bool(sr & (1 << 8)),
        "frame_count": (sr >> 16) & 0xFF,
    }


# ---------------------------------------------------------------------------
# Video Multi-Scaler (PG325)
# ---------------------------------------------------------------------------
# Register map (from PG325 Table 2-7).
# The Multi-Scaler is memory-mapped: it reads source frames from DDR
# and writes scaled outputs to DDR.  Up to 8 outputs, each with its
# own register block.
#
# The register layout is complex and partially depends on the number
# of outputs configured at build time.  The offsets below are for
# a 2-output configuration.
#
# AP_CTRL:
#   0x000  AP_CTRL
#   0x010  num_outs (number of active outputs, RW)
#
# Per-output registers (base offset varies by output index):
#   The exact register map depends on the .hwh.  The following offsets
#   are typical for a 2-output build and will be verified against the
#   actual .hwh after block design synthesis.

MSCALER_NUM_OUTS = 0x010
MSCALER_WIDTHIN = 0x020
MSCALER_HEIGHTIN = 0x028
MSCALER_INPIXELFMT = 0x030  # Input pixel format
MSCALER_INSTRIDE = 0x038     # Input stride in bytes
MSCALER_SRCIMGBUF = 0x040    # Source buffer address (low 32 bits)

# Output 0 registers (offsets may shift depending on .hwh)
MSCALER_OUT0_WIDTH = 0x048
MSCALER_OUT0_HEIGHT = 0x050
MSCALER_OUT0_PIXFMT = 0x058
MSCALER_OUT0_STRIDE = 0x060
MSCALER_OUT0_DSTBUF = 0x068

# Output 1 registers
MSCALER_OUT1_WIDTH = 0x070
MSCALER_OUT1_HEIGHT = 0x078
MSCALER_OUT1_PIXFMT = 0x080
MSCALER_OUT1_STRIDE = 0x088
MSCALER_OUT1_DSTBUF = 0x090

# Pixel format codes (PG325 Table 2-9)
MSCALER_FMT_RGBX10 = 15  # 4 bytes/pixel, 10-bit per channel
MSCALER_FMT_RGB8 = 14     # 3 bytes/pixel, 8-bit per channel


def configure_multi_scaler(
    ip,
    src_addr: int,
    src_width: int,
    src_height: int,
    src_stride: int,
    outputs: list,
) -> None:
    """Configure and start the Multi-Scaler.

    Note: Register offsets are preliminary and must be verified against
    the .hwh after the block design is synthesized.  The Multi-Scaler's
    register map varies based on build-time configuration.

    Args:
        src_addr: Physical address of the source frame in DDR.
        src_width: Source frame width in pixels.
        src_height: Source frame height in pixels.
        src_stride: Source frame stride in bytes.
        outputs: List of dicts with keys:
            addr: Physical address of destination buffer
            width: Output width in pixels
            height: Output height in pixels
            stride: Output stride in bytes (width * 3 for RGB8)
    """
    ip.write(MSCALER_NUM_OUTS, len(outputs))
    ip.write(MSCALER_WIDTHIN, src_width)
    ip.write(MSCALER_HEIGHTIN, src_height)
    ip.write(MSCALER_INPIXELFMT, MSCALER_FMT_RGBX10)
    ip.write(MSCALER_INSTRIDE, src_stride)
    ip.write(MSCALER_SRCIMGBUF, src_addr)

    # Output register bases (for 2-output config)
    out_regs = [
        (MSCALER_OUT0_WIDTH, MSCALER_OUT0_HEIGHT,
         MSCALER_OUT0_PIXFMT, MSCALER_OUT0_STRIDE, MSCALER_OUT0_DSTBUF),
        (MSCALER_OUT1_WIDTH, MSCALER_OUT1_HEIGHT,
         MSCALER_OUT1_PIXFMT, MSCALER_OUT1_STRIDE, MSCALER_OUT1_DSTBUF),
    ]

    for i, out in enumerate(outputs):
        w_reg, h_reg, fmt_reg, s_reg, dst_reg = out_regs[i]
        ip.write(w_reg, out["width"])
        ip.write(h_reg, out["height"])
        ip.write(fmt_reg, MSCALER_FMT_RGB8)
        ip.write(s_reg, out["stride"])
        ip.write(dst_reg, out["addr"])

    # Start the scaler
    ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)

    logger.info(
        "Multi-Scaler started: src %dx%d -> %d outputs",
        src_width, src_height, len(outputs),
    )
