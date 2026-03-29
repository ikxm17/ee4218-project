"""IMX219 register configuration for 1920x1080 RAW10 @ 30fps, 2-lane MIPI CSI-2.

Register format: list of (16-bit_address, 8-bit_value) tuples.
IMX219 uses CCI protocol: 16-bit big-endian register addresses, 8-bit data.
Multi-byte sensor values are split across consecutive addresses (MSB first).

Sources:
    - Linux kernel driver: drivers/media/i2c/imx219.c (v6.x, torvalds/linux)
    - IMX219PQH5-C datasheet (docs/peripherals/IMX219PQ.pdf)

PLL configuration assumes 24 MHz external clock (standard RPi Camera Module v2
on-board oscillator). KV260 RPi connector does not provide a clock — the module
has its own 24 MHz crystal.
"""

# ---------------------------------------------------------------------------
# Manufacturer-specific access unlock + vendor tuning (imx219_common_regs)
# ---------------------------------------------------------------------------
# The 0x30EB sequence unlocks access to registers in the 0x3000-0x5FFF range.
# Vendor tuning registers (0x4xxx) are undocumented but required for correct
# analog front-end behavior.  Values from Linux kernel driver.

COMMON_INIT = [
    # Standby before configuration (datasheet Sec 3-2-2, Table 37 step 4)
    (0x0100, 0x00),

    # Manufacturer-specific access sequence (datasheet Sec 3-4)
    (0x30EB, 0x05),
    (0x30EB, 0x0C),
    (0x300A, 0xFF),
    (0x300B, 0xFF),
    (0x30EB, 0x05),
    (0x30EB, 0x09),

    # Vendor analog tuning registers
    (0x455E, 0x00),
    (0x471E, 0x4B),
    (0x4767, 0x0F),
    (0x4750, 0x14),
    (0x4540, 0x00),
    (0x47B4, 0x14),
    (0x4713, 0x30),
    (0x478B, 0x10),
    (0x478F, 0x10),
    (0x4793, 0x10),
    (0x4797, 0x0E),
    (0x479B, 0x0E),

    # Sub-sampling increment (datasheet Sec 3-2-3, page 31)
    # 1 = read every pixel (no skip).  Set to 3 for 2x2 binning modes.
    (0x0170, 0x01),  # X_ODD_INC_A
    (0x0171, 0x01),  # Y_ODD_INC_A

    # D-PHY timing control (datasheet Sec 3-2-2, page 29)
    # 0 = auto mode (sensor computes timing from PLL settings)
    (0x0128, 0x00),  # DPHY_CTRL

    # External clock frequency declaration: 24.000 MHz
    # Register is in MHz with 8-bit fractional: 24 * 256 = 0x1800
    # (datasheet Sec 3-2-2, 0x012A-0x012B)
    (0x012A, 0x18),  # EXCK_FREQ[15:8] = 24
    (0x012B, 0x00),  # EXCK_FREQ[7:0]  = .00
]


# ---------------------------------------------------------------------------
# PLL + 2-lane CSI-2 configuration (imx219_2lane_regs)
# ---------------------------------------------------------------------------
# VT pixel clock = EXCK / PREPLLCK_VT_DIV * PLL_VT_MPY / VTSYCK_DIV / VTPXCK_DIV
#                = 24 / 3 * 57 / 1 / 5 = 91.2 MHz
# Pixel rate     = VT_pix_clk * 2 (2 pixels/clock in readout) = 182.4 Mpix/s
# OP bit rate    = EXCK / PREPLLCK_OP_DIV * PLL_OP_MPY / OPSYCK_DIV
#                = 24 / 3 * 114 / 1 = 912 Mbps per lane
# MIPI link freq = 912 / 2 = 456 MHz (DDR)

PLL_CONFIG = [
    (0x0301, 0x05),  # VTPXCK_DIV = 5
    (0x0303, 0x01),  # VTSYCK_DIV = 1
    (0x0304, 0x03),  # PREPLLCK_VT_DIV = 3
    (0x0305, 0x03),  # PREPLLCK_OP_DIV = 3
    (0x0306, 0x00),  # PLL_VT_MPY[10:8] = 0
    (0x0307, 0x39),  # PLL_VT_MPY[7:0]  = 57
    (0x0309, 0x0A),  # OPPXCK_DIV = 10 (= bits per pixel for RAW10)
    (0x030B, 0x01),  # OPSYCK_DIV = 1
    (0x030C, 0x00),  # PLL_OP_MPY[10:8] = 0
    (0x030D, 0x72),  # PLL_OP_MPY[7:0]  = 114
]


# ---------------------------------------------------------------------------
# CSI-2 lane configuration
# ---------------------------------------------------------------------------
# Must be set before streaming starts (datasheet Sec 4-1-1, Table 12:
# "Setting before standby cancel").  Default is 0x03 (4-lane).

CSI_LANE_CONFIG = [
    (0x0114, 0x01),  # CSI_LANE_MODE = 2-lane
]


# ---------------------------------------------------------------------------
# Frame format: 1920x1080 full-pixel readout (no binning)
# ---------------------------------------------------------------------------
# Crop window calculated from the active pixel array:
#   Active array: 3280 x 2464 (starting at offset 8,8 from native 3296x2480)
#   Crop center:  (3296 - 1920) / 2 = 688 native → 688 - 8 = 680 in array coords
#                 (2480 - 1080) / 2 = 700 native → 700 - 8 = 692 in array coords
#
# All Frame Bank A registers (datasheet Sec 3-2-3, page 30-31)

FRAME_CONFIG = [
    # Crop window (datasheet Sec 6-5, 6-6)
    (0x0164, 0x02),  # X_ADD_STA_A[11:8] = 680 >> 8 = 2
    (0x0165, 0xA8),  # X_ADD_STA_A[7:0]  = 680 & 0xFF = 0xA8
    (0x0166, 0x0A),  # X_ADD_END_A[11:8]  = 2599 >> 8 = 10
    (0x0167, 0x27),  # X_ADD_END_A[7:0]   = 2599 & 0xFF = 0x27
    (0x0168, 0x02),  # Y_ADD_STA_A[11:8]  = 692 >> 8 = 2
    (0x0169, 0xB4),  # Y_ADD_STA_A[7:0]   = 692 & 0xFF = 0xB4
    (0x016A, 0x06),  # Y_ADD_END_A[11:8]  = 1771 >> 8 = 6
    (0x016B, 0xEB),  # Y_ADD_END_A[7:0]   = 1771 & 0xFF = 0xEB

    # No binning (datasheet Sec 6-5, page 67)
    (0x0174, 0x00),  # BINNING_MODE_H_A = 0 (no binning)
    (0x0175, 0x00),  # BINNING_MODE_V_A = 0 (no binning)

    # Output size
    (0x016C, 0x07),  # x_output_size[11:8] = 1920 >> 8 = 7
    (0x016D, 0x80),  # x_output_size[7:0]  = 1920 & 0xFF = 0x80
    (0x016E, 0x04),  # y_output_size[11:8] = 1080 >> 8 = 4
    (0x016F, 0x38),  # y_output_size[7:0]  = 1080 & 0xFF = 0x38

    # Test pattern window (matches output size)
    (0x0624, 0x07),  # TP_WINDOW_WIDTH[11:8]
    (0x0625, 0x80),  # TP_WINDOW_WIDTH[7:0]
    (0x0626, 0x04),  # TP_WINDOW_HEIGHT[11:8]
    (0x0627, 0x38),  # TP_WINDOW_HEIGHT[7:0]

    # CSI-2 data format: RAW10 (datasheet Sec 4-1-5, Table 14)
    (0x018C, 0x0A),  # CSI_DATA_FORMAT_A[15:8] = 0x0A
    (0x018D, 0x0A),  # CSI_DATA_FORMAT_A[7:0]  = 0x0A
]


# ---------------------------------------------------------------------------
# Timing: line length, frame length, exposure, gain
# ---------------------------------------------------------------------------
# LINE_LENGTH_A = 3448 (min for full-width no-binning, datasheet default)
# FRM_LENGTH_A  = 1763 (for ~30 fps: 182.4M / (3448 * 1763) ≈ 30.0 fps)

TIMING_CONFIG = [
    (0x0162, 0x0D),  # LINE_LENGTH_A[15:8] = 3448 >> 8
    (0x0163, 0x78),  # LINE_LENGTH_A[7:0]  = 3448 & 0xFF
    (0x0160, 0x06),  # FRM_LENGTH_A[15:8]  = 1763 >> 8
    (0x0161, 0xE3),  # FRM_LENGTH_A[7:0]   = 1763 & 0xFF

    # Default exposure: 1600 lines (safe value, max = FRM_LENGTH - 4 = 1759)
    (0x015A, 0x06),  # COARSE_INTEGRATION_TIME_A[15:8] = 1600 >> 8
    (0x015B, 0x40),  # COARSE_INTEGRATION_TIME_A[7:0]  = 1600 & 0xFF

    # Analog gain = 0 (1x, datasheet Sec 5-8, Table 23)
    (0x0157, 0x00),  # ANA_GAIN_GLOBAL_A

    # Digital gain = 1x (256 in 8.8 fixed-point)
    (0x0158, 0x01),  # DIG_GAIN_GLOBAL_A[11:8]
    (0x0159, 0x00),  # DIG_GAIN_GLOBAL_A[7:0]

    # Image orientation: no flip, no mirror
    (0x0172, 0x00),  # IMG_ORIENTATION_A

    # Test pattern: disabled
    (0x0600, 0x00),  # test_pattern_mode[8]
    (0x0601, 0x00),  # test_pattern_mode[7:0]
]


# ---------------------------------------------------------------------------
# Assembled init table
# ---------------------------------------------------------------------------
# Write this entire sequence while mode_select = 0x00 (standby).
# After writing, set mode_select = 0x01 to start streaming.

INIT_TABLE_1080P30_RAW10_2LANE = (
    COMMON_INIT
    + PLL_CONFIG
    + CSI_LANE_CONFIG
    + FRAME_CONFIG
    + TIMING_CONFIG
)

# Stream control registers
REG_MODE_SELECT = 0x0100
REG_SOFTWARE_RESET = 0x0103
REG_MODEL_ID_H = 0x0000
REG_MODEL_ID_L = 0x0001

MODEL_ID_VALUE = 0x0219
