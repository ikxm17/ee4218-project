"""MIPI CSI-2 RX Subsystem driver — D-PHY receiver and packet decoder.

Reference: PG232 v6.0 (docs/ips/mipi/pg232-mipi-csi2-rx-en-us-6.0.pdf)
"""

import logging
import time

logger = logging.getLogger(__name__)


class Csi2RxDriver:
    """MIPI CSI-2 RX Subsystem (PG232 v6.0).

    The CSI-2 RX is NOT an HLS IP — it uses a Xilinx-specific register
    map, not the AP_CTRL handshake.

    The 8 KB AXI-Lite address space is split into two 4 KB regions:
      0x0000–0x0FFF  CSI-2 RX Controller registers (this driver)
      0x1000–0x1FFF  MIPI D-PHY registers (PG202, not used here)
    """

    # IP identification (for audit — class covers ALL instances of this type)
    IP_VLNV = "xilinx.com:ip:mipi_csi2_rx_subsystem:6.0"
    IP_NAME = "mipi_csi2_rx_subsyst_0"  # primary instance

    # ── Register map (PG232 v6.0 Tables 17–45) ──────────────────────

    # Core control & status
    CORE_CONFIG     = 0x00
    PROTOCOL_CONFIG = 0x04
    CORE_STATUS     = 0x10

    # Interrupts
    GLOBAL_IRQ_EN   = 0x20
    ISR             = 0x24
    IER             = 0x28

    # Protocol
    VC_SEL          = 0x2C
    VCX_FRAME_ERR   = 0x34

    # Lane info
    CLK_LANE_INFO   = 0x3C
    LANE0_INFO      = 0x40
    LANE1_INFO      = 0x44
    LANE2_INFO      = 0x48
    LANE3_INFO      = 0x4C

    # Per-VC image info (VC0 base; stride 8 bytes per VC, 0–15)
    IMG_INFO1_VC0   = 0x60   # [31:16] line count, [15:0] byte count
    IMG_INFO2_VC0   = 0x64   # [5:0] data type

    # D-PHY sub-core offset (PG202, for future use)
    DPHY_BASE_OFFSET = 0x1000

    # ── Bit-field constants ──────────────────────────────────────────

    # Core Configuration Register (0x00)
    _CCR_CORE_ENABLE = 1 << 0
    _CCR_SOFT_RESET  = 1 << 1

    # Protocol Configuration Register (0x04)
    _PCR_ACTIVE_LANES_MASK = 0x03        # bits [1:0] R/W
    _PCR_MAX_LANES_MASK    = 0x03 << 3   # bits [4:3] R-only

    # Core Status Register (0x10)
    _CSR_PKT_COUNT_SHIFT    = 16
    _CSR_PKT_COUNT_MASK     = 0xFFFF << 16
    _CSR_SP_FIFO_FULL       = 1 << 3
    _CSR_SP_FIFO_NOT_EMPTY  = 1 << 2
    _CSR_LINE_BUF_FULL      = 1 << 1
    _CSR_SOFT_RESET_IN_PROG = 1 << 0

    # Clock Lane Info (0x3C) — bit 0 is Reserved
    _CLI_STOP_STATE = 1 << 1

    # Data Lane Info (0x40–0x4C)
    _DLI_STOP_STATE     = 1 << 5
    _DLI_SKEWCALHS      = 1 << 2
    _DLI_SOT_ERROR      = 1 << 1
    _DLI_SOT_SYNC_ERROR = 1 << 0

    # Interrupt Status Register (0x24) — all bits W1C
    _ISR_FRAME_RECEIVED       = 1 << 31
    _ISR_VCX_FRAME_ERR        = 1 << 30
    _ISR_RX_SKEWCALHS         = 1 << 29
    _ISR_YUV420_WC_ERR        = 1 << 28
    _ISR_PENDING_WRITE_FIFO   = 1 << 27
    _ISR_WC_CORRUPTION        = 1 << 22
    _ISR_INCORRECT_LANE_CFG   = 1 << 21
    _ISR_SP_FIFO_FULL         = 1 << 20
    _ISR_SP_FIFO_NOT_EMPTY    = 1 << 19
    _ISR_STREAM_LINE_BUF_FULL = 1 << 18
    _ISR_STOP_STATE           = 1 << 17
    _ISR_SOT_ERROR            = 1 << 13
    _ISR_SOT_SYNC_ERROR       = 1 << 12
    _ISR_ECC_2BIT_ERR         = 1 << 11
    _ISR_ECC_1BIT_ERR         = 1 << 10
    _ISR_CRC_ERROR            = 1 << 9
    _ISR_UNSUPPORTED_DT       = 1 << 8

    # ── Methods ──────────────────────────────────────────────────────

    def __init__(self, ip):
        """Wrap a PYNQ IP or MMIO handle for the CSI-2 RX subsystem."""
        self._ip = ip

    def _poll_reset_done(self, timeout_s: float = 0.1) -> None:
        """Poll CORE_STATUS bit 0 until soft reset / disable completes.

        PG232 Sec 4: after asserting soft reset or clearing Core Enable,
        bit 0 of CORE_STATUS reads 1 while the operation is in progress.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not (self._ip.read(self.CORE_STATUS) & self._CSR_SOFT_RESET_IN_PROG):
                return
            time.sleep(0.0001)  # 100 µs poll interval
        logger.warning(
            "CSI-2 RX reset/disable did not complete within %.3fs "
            "(CORE_STATUS=0x%08X)",
            timeout_s, self._ip.read(self.CORE_STATUS),
        )

    def disable(self) -> None:
        """Disable the CSI-2 RX core and wait for completion.

        PG232 Sec 4: after clearing Core Enable, poll CORE_STATUS[0]
        until the disable completes.
        """
        self._ip.write(self.CORE_CONFIG, 0x00)
        self._poll_reset_done()
        logger.info("CSI-2 RX core disabled")

    def reset(self) -> None:
        """Soft-reset the CSI-2 RX core with proper completion polling.

        PG232 Sec 4 programming sequence:
          1. Assert soft reset (set bit 1)
          2. Poll CORE_STATUS[0] until reset completes
          3. Deassert soft reset (clear bit 1)
          4. Clear all ISR flags (W1C)
          5. Re-enable core (set bit 0)

        The D-PHY receiver must see the sensor's clock lane toggling before
        it can achieve byte-level synchronization.  A soft reset clears the
        lane state machines and forces a fresh sync attempt.
        """
        self._ip.write(self.CORE_CONFIG, self._CCR_SOFT_RESET)
        self._poll_reset_done()
        self._ip.write(self.CORE_CONFIG, 0x00)            # deassert reset
        self._ip.write(self.ISR, 0xFFFFFFFF)               # clear all ISR bits
        self._ip.write(self.CORE_CONFIG, self._CCR_CORE_ENABLE)
        logger.info("CSI-2 RX soft reset complete, core re-enabled")

    def start(self) -> None:
        """Start the CSI-2 RX core (alias for reset / D-PHY re-sync)."""
        self.reset()

    def read_status(self) -> dict:
        """Read and decode CSI-2 RX diagnostic registers.

        Returns a dict with bit-field positions per PG232 v6.0.
        """
        core_cfg    = self._ip.read(self.CORE_CONFIG)
        proto_cfg   = self._ip.read(self.PROTOCOL_CONFIG)
        core_status = self._ip.read(self.CORE_STATUS)
        isr         = self._ip.read(self.ISR)
        clk_info    = self._ip.read(self.CLK_LANE_INFO)
        lane0_info  = self._ip.read(self.LANE0_INFO)
        lane1_info  = self._ip.read(self.LANE1_INFO)

        pkt_count = (core_status >> self._CSR_PKT_COUNT_SHIFT) & 0xFFFF

        status = {
            # Core state
            "core_enabled":      bool(core_cfg & self._CCR_CORE_ENABLE),
            "soft_reset_active": bool(core_cfg & self._CCR_SOFT_RESET),

            # Protocol — active_lanes is bits [1:0], max_lanes is bits [4:3]
            "active_lanes": (proto_cfg & self._PCR_ACTIVE_LANES_MASK) + 1,
            "max_lanes":    ((proto_cfg >> 3) & 0x03) + 1,

            # Core status
            "packet_count":      pkt_count,
            "reset_in_progress": bool(core_status & self._CSR_SOFT_RESET_IN_PROG),
            "line_buf_full":     bool(core_status & self._CSR_LINE_BUF_FULL),
            "sp_fifo_not_empty": bool(core_status & self._CSR_SP_FIFO_NOT_EMPTY),
            "sp_fifo_full":      bool(core_status & self._CSR_SP_FIFO_FULL),

            # ISR
            "isr_raw":             isr,
            "isr_frame_received":  bool(isr & self._ISR_FRAME_RECEIVED),
            "isr_stop_state":      bool(isr & self._ISR_STOP_STATE),
            "isr_line_buf_full":   bool(isr & self._ISR_STREAM_LINE_BUF_FULL),
            "isr_sot_error":       bool(isr & self._ISR_SOT_ERROR),
            "isr_sot_sync_error":  bool(isr & self._ISR_SOT_SYNC_ERROR),
            "isr_crc_error":       bool(isr & self._ISR_CRC_ERROR),
            "isr_ecc_1bit":        bool(isr & self._ISR_ECC_1BIT_ERR),
            "isr_ecc_2bit":        bool(isr & self._ISR_ECC_2BIT_ERR),
            "isr_unsupported_dt":  bool(isr & self._ISR_UNSUPPORTED_DT),
            "isr_wc_corruption":   bool(isr & self._ISR_WC_CORRUPTION),
            "isr_incorrect_lanes": bool(isr & self._ISR_INCORRECT_LANE_CFG),

            # Clock lane — bit 1 only (bit 0 is reserved)
            "clk_lane_stop": bool(clk_info & self._CLI_STOP_STATE),

            # Data lanes — bit 5 = stop, bit 2 = skew, bit 1 = SoT, bit 0 = SoT sync
            "lane0_stop":           bool(lane0_info & self._DLI_STOP_STATE),
            "lane0_skewcalhs":      bool(lane0_info & self._DLI_SKEWCALHS),
            "lane0_sot_error":      bool(lane0_info & self._DLI_SOT_ERROR),
            "lane0_sot_sync_error": bool(lane0_info & self._DLI_SOT_SYNC_ERROR),
            "lane1_stop":           bool(lane1_info & self._DLI_STOP_STATE),
            "lane1_skewcalhs":      bool(lane1_info & self._DLI_SKEWCALHS),
            "lane1_sot_error":      bool(lane1_info & self._DLI_SOT_ERROR),
            "lane1_sot_sync_error": bool(lane1_info & self._DLI_SOT_SYNC_ERROR),
        }

        logger.info(
            "CSI-2 RX status: packets=%d, ISR=0x%08X, "
            "clk_stop=%s, d0_stop=%s, d1_stop=%s, "
            "active_lanes=%d, max_lanes=%d",
            pkt_count, isr,
            status["clk_lane_stop"], status["lane0_stop"], status["lane1_stop"],
            status["active_lanes"], status["max_lanes"],
        )
        return status

    def read_image_info(self, vc: int = 0) -> dict:
        """Read per-VC image information registers (PG232 Tables 44–45).

        Args:
            vc: Virtual channel number (0–15).

        Returns:
            {"line_count": int, "byte_count": int, "data_type": int}
        """
        if not 0 <= vc <= 15:
            raise ValueError(f"VC must be 0–15, got {vc}")
        info1 = self._ip.read(self.IMG_INFO1_VC0 + vc * 8)
        info2 = self._ip.read(self.IMG_INFO2_VC0 + vc * 8)
        result = {
            "line_count": (info1 >> 16) & 0xFFFF,
            "byte_count": info1 & 0xFFFF,
            "data_type":  info2 & 0x3F,
        }
        logger.debug(
            "CSI-2 RX VC%d image info: %d lines, %d bytes, DT=0x%02X",
            vc, result["line_count"], result["byte_count"], result["data_type"],
        )
        return result

    def wait_for_lock(self, timeout_s: float = 2.0) -> bool:
        """Poll CSI-2 RX until packets are received or timeout expires."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            core_status = self._ip.read(self.CORE_STATUS)
            pkt_count = (core_status >> self._CSR_PKT_COUNT_SHIFT) & 0xFFFF
            if pkt_count > 0:
                logger.info("CSI-2 RX locked: %d packets received", pkt_count)
                return True
            time.sleep(0.010)
        logger.warning("CSI-2 RX lock timeout: no packets after %.1fs", timeout_s)
        return False
