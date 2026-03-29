"""MIPI CSI-2 RX Subsystem driver — D-PHY receiver and packet decoder.

Reference: PG232 (docs/ips/video/pg232-mipi-csi2-rx-en-us-4.1.pdf)
"""

import logging
import time

logger = logging.getLogger(__name__)


class Csi2RxDriver:
    """MIPI CSI-2 RX Subsystem (PG232).

    The CSI-2 RX is NOT an HLS IP — it uses a Xilinx-specific register
    map, not the AP_CTRL handshake.
    """

    # IP identification (for audit — class covers ALL instances of this type)
    IP_VLNV = "xilinx.com:ip:mipi_csi2_rx_subsystem:6.0"
    IP_NAME = "mipi_csi2_rx_subsyst_0"  # primary instance

    # -- Register map (PG232 Table 2-4) --
    CORE_CONFIG = 0x00       # Bit 0: Enable, Bit 1: Soft Reset
    PROTOCOL_CONFIG = 0x04   # Bit [4:3]: Active lanes
    CORE_STATUS = 0x10       # Bit [31:16]: Packet count
    ISR = 0x24               # Interrupt Status (W1C)
    CLK_LANE_INFO = 0x3C     # Bit 1: Stop state, Bit 0: HS mode
    LANE0_INFO = 0x40        # Bit 1: Stop state, Bit 0: HS mode
    LANE1_INFO = 0x44        # Same as Lane 0

    def __init__(self, ip):
        """Wrap a PYNQ IP or MMIO handle for the CSI-2 RX subsystem."""
        self._ip = ip

    def disable(self) -> None:
        """Disable the CSI-2 RX core (write 0 to CORE_CONFIG)."""
        self._ip.write(self.CORE_CONFIG, 0x00)
        logger.info("CSI-2 RX core disabled")

    def reset(self) -> None:
        """Perform a soft reset of the CSI-2 RX core (PG232 Sec 2.3).

        The D-PHY receiver must see the sensor's clock lane toggling before
        it can achieve byte-level synchronization.  If the core was enabled
        at overlay load time (before the sensor started streaming), the
        D-PHY may have missed the initial LP->HS transition.  A soft reset
        clears the lane state machines and forces a fresh sync attempt.
        """
        self._ip.write(self.CORE_CONFIG, 0x00)       # Disable core
        self._ip.write(self.CORE_CONFIG, 0x02)       # Assert soft reset
        time.sleep(0.001)                             # Hold reset 1 ms
        self._ip.write(self.CORE_CONFIG, 0x00)       # Deassert reset
        self._ip.write(self.ISR, 0xFFFFFFFF)         # Clear all ISR bits
        self._ip.write(self.CORE_CONFIG, 0x01)       # Re-enable core
        logger.info("CSI-2 RX soft reset complete, core re-enabled")

    def start(self) -> None:
        """Start the CSI-2 RX core (alias for reset / D-PHY re-sync)."""
        self.reset()

    def read_status(self) -> dict:
        """Read and decode CSI-2 RX diagnostic registers."""
        core_cfg = self._ip.read(self.CORE_CONFIG)
        proto_cfg = self._ip.read(self.PROTOCOL_CONFIG)
        core_status = self._ip.read(self.CORE_STATUS)
        isr = self._ip.read(self.ISR)
        clk_info = self._ip.read(self.CLK_LANE_INFO)
        lane0_info = self._ip.read(self.LANE0_INFO)
        lane1_info = self._ip.read(self.LANE1_INFO)

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

    def wait_for_lock(self, timeout_s: float = 2.0) -> bool:
        """Poll CSI-2 RX until packets are received or timeout expires."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            core_status = self._ip.read(self.CORE_STATUS)
            pkt_count = (core_status >> 16) & 0xFFFF
            if pkt_count > 0:
                logger.info("CSI-2 RX locked: %d packets received", pkt_count)
                return True
            time.sleep(0.010)
        logger.warning("CSI-2 RX lock timeout: no packets after %.1fs", timeout_s)
        return False
