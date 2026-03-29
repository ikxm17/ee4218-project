"""AXI VDMA driver — S2MM (write) channel for frame capture.

Reference: PG020 (docs/ips/other/pg020_axi_vdma-en-us-6.3.pdf)

Register map (PG020 Table 2-16, S2MM registers):
    0x30  S2MM_DMACR        Control register
    0x34  S2MM_DMASR        Status register
    0x48  S2MM_FRMSTORE     Number of frame stores (1-32)
    0xA0  S2MM_VSIZE         Vertical size (writing this starts the channel)
    0xA4  S2MM_HSIZE         Horizontal size in bytes
    0xA8  S2MM_FRMDLY_STRIDE Frame delay + stride
    0xAC  S2MM_START_ADDRESS1  Frame store 1 base address
    0xB0  S2MM_START_ADDRESS2  Frame store 2 base address
    0xB4  S2MM_START_ADDRESS3  Frame store 3 base address

S2MM_DMACR bits:
    [0]    RS (Run/Stop)
    [1]    Circular mode (1 = circular)
    [2]    Reset
    [4]    Frame count enable
    [16]   IRQ on complete
"""

import logging

logger = logging.getLogger(__name__)


class VdmaDriver:
    """AXI VDMA S2MM channel (PG020)."""

    # IP identification (for audit — class covers ALL instances of this type)
    IP_VLNV = "xilinx.com:ip:axi_vdma:6.3"
    IP_NAME = "axi_vdma_0"  # primary instance

    # -- Register map --
    S2MM_DMACR = 0x30
    S2MM_DMASR = 0x34
    S2MM_FRMSTORE = 0x48
    S2MM_VSIZE = 0xA0
    S2MM_HSIZE = 0xA4
    S2MM_FRMDLY_STRIDE = 0xA8
    S2MM_START_ADDR_BASE = 0xAC  # +0x04 per frame store

    def __init__(self, ip):
        """Wrap a PYNQ DefaultIP handle for the VDMA."""
        self._ip = ip

    def configure(
        self,
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
        self._ip.write(self.S2MM_DMACR, 0x04)
        while self._ip.read(self.S2MM_DMACR) & 0x04:
            pass  # Wait for reset to clear

        # Set number of frame stores
        self._ip.write(self.S2MM_FRMSTORE, len(frame_addrs))

        # Set frame store addresses
        for i, addr in enumerate(frame_addrs):
            self._ip.write(self.S2MM_START_ADDR_BASE + i * 4, addr)

        # Set stride (lower 16 bits) and frame delay (upper 16 bits = 0)
        self._ip.write(self.S2MM_FRMDLY_STRIDE, stride & 0xFFFF)

        # Set horizontal size in bytes
        self._ip.write(self.S2MM_HSIZE, width_bytes)

        # Clear all W1C error bits in DMASR (bits 4-8, 11-14)
        self._ip.write(self.S2MM_DMASR, 0x000079F0)

        # Run in circular mode (continuous frame capture)
        self._ip.write(self.S2MM_DMACR, 0x03)  # RS=1, Circular=1

        # Writing VSIZE starts the channel — it waits for SOF to sync
        self._ip.write(self.S2MM_VSIZE, height)

        logger.info(
            "VDMA S2MM started: %d stores, %dx%d, stride=%d",
            len(frame_addrs), width_bytes, height, stride,
        )

    def read_dmasr(self) -> int:
        """Read the raw S2MM_DMASR register value."""
        return self._ip.read(self.S2MM_DMASR)

    def read_status(self) -> dict:
        """Read VDMA S2MM status register for debugging.

        Bit positions from xaxivdma_hw.h / PG020 Table 2-19 (S2MM_DMASR).
        """
        sr = self._ip.read(self.S2MM_DMASR)
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
