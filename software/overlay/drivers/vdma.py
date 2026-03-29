"""AXI VDMA driver — S2MM (write) channel for frame capture.

Reference: PG020 v6.3 (docs/ips/video/pg020-axi-vdma.pdf)

Register map (S2MM channel, PG020 Tables 2-11/2-12):
    0x28  PARK_PTR_REG       Park pointer & current frame store index (shared)
    0x2C  VDMA_VERSION       IP version (RO)
    0x30  S2MM_DMACR         Control register
    0x34  S2MM_DMASR         Status register
    0x3C  S2MM_IRQ_MASK      Error interrupt mask
    0x48  S2MM_FRMSTORE      Number of frame stores (1-32)
    0xA0  S2MM_VSIZE         Vertical size (writing this starts the channel)
    0xA4  S2MM_HSIZE         Horizontal size in bytes
    0xA8  S2MM_FRMDLY_STRIDE Frame delay (bits 28:24) + stride (bits 15:0)
    0xAC+ S2MM_START_ADDRESS Frame store base addresses (+0x04 per store)

S2MM_DMACR bits (PG020 Table 2-11):
    [0]      RS — Run/Stop
    [1]      Circular_Park — 0=park, 1=circular
    [2]      Reset — soft reset (self-clearing)
    [3]      GenLockEn
    [4]      FrameCntEn
    [23:16]  IRQFrameCount — interrupt after N frames
"""

import logging

logger = logging.getLogger(__name__)


class VdmaDriver:
    """AXI VDMA S2MM channel (PG020 v6.3)."""

    # IP identification (for audit — class covers ALL instances of this type)
    IP_VLNV = "xilinx.com:ip:axi_vdma:6.3"
    IP_NAME = "axi_vdma_0"  # primary instance

    # -- Register offsets --
    PARK_PTR_REG        = 0x28
    VDMA_VERSION        = 0x2C
    S2MM_DMACR          = 0x30
    S2MM_DMASR          = 0x34
    S2MM_IRQ_MASK       = 0x3C
    S2MM_FRMSTORE       = 0x48
    S2MM_VSIZE          = 0xA0
    S2MM_HSIZE          = 0xA4
    S2MM_FRMDLY_STRIDE  = 0xA8
    S2MM_START_ADDR_BASE = 0xAC  # +0x04 per frame store

    # -- S2MM_DMACR bit fields (PG020 Table 2-11) --
    DMACR_RS            = 1 << 0
    DMACR_CIRCULAR      = 1 << 1
    DMACR_RESET         = 1 << 2
    DMACR_GENLOCK_EN    = 1 << 3
    DMACR_FRMCNT_EN     = 1 << 4

    # -- S2MM_DMASR bit fields (PG020 Table 2-12) --
    DMASR_HALTED        = 1 << 0
    DMASR_VDMA_INT_ERR  = 1 << 4   # R/WC — internal error (HSIZE/VSIZE=0, mismatch)
    DMASR_VDMA_SLV_ERR  = 1 << 5   # RO   — AXI slave error
    DMASR_VDMA_DEC_ERR  = 1 << 6   # RO   — AXI decode error (bad address)
    DMASR_SOF_EARLY_ERR = 1 << 7   # R/WC — frame shorter than VSIZE
    DMASR_EOL_EARLY_ERR = 1 << 8   # R/WC — line shorter than HSIZE
    DMASR_SOF_LATE_ERR  = 1 << 11  # R/WC — frame longer than VSIZE
    DMASR_FRM_CNT_IRQ   = 1 << 12  # R/WC — frame count interrupt
    DMASR_DLY_CNT_IRQ   = 1 << 13  # R/WC — delay count interrupt
    DMASR_ERR_IRQ       = 1 << 14  # R/WC — error interrupt (aggregate)
    DMASR_EOL_LATE_ERR  = 1 << 15  # R/WC — line longer than HSIZE

    # Mask of all R/WC bits in S2MM_DMASR (for clearing).
    # Excludes bits 5-6 (RO — writes ignored).
    DMASR_W1C_MASK      = 0xF990   # bits 4, 7-8, 11-15

    def __init__(self, ip):
        """Wrap a PYNQ DefaultIP handle for the VDMA."""
        self._ip = ip

    def reset(self):
        """Soft-reset the S2MM channel and wait for completion."""
        self._ip.write(self.S2MM_DMACR, self.DMACR_RESET)
        while self._ip.read(self.S2MM_DMACR) & self.DMACR_RESET:
            pass

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
        the first SOF (tuser[0]) from the upstream IP before writing.

        Args:
            frame_addrs: Physical addresses for frame stores (e.g. 3 for triple-buffer).
            width_bytes: Horizontal frame size in bytes (width * bytes_per_pixel).
            height: Number of lines per frame.
            stride: Bytes per line including any padding.
        """
        self.reset()

        # Set number of frame stores
        self._ip.write(self.S2MM_FRMSTORE, len(frame_addrs))

        # Set frame store addresses
        for i, addr in enumerate(frame_addrs):
            self._ip.write(self.S2MM_START_ADDR_BASE + i * 4, addr)

        # Set stride (lower 16 bits); frame delay = 0 (upper bits)
        self._ip.write(self.S2MM_FRMDLY_STRIDE, stride & 0xFFFF)

        # Set horizontal size in bytes
        self._ip.write(self.S2MM_HSIZE, width_bytes)

        # Clear all W1C error/IRQ bits in DMASR
        self._ip.write(self.S2MM_DMASR, self.DMASR_W1C_MASK)

        # Run in circular mode (continuous frame capture)
        self._ip.write(self.S2MM_DMACR, self.DMACR_RS | self.DMACR_CIRCULAR)

        # Writing VSIZE starts the channel — waits for SOF to sync
        self._ip.write(self.S2MM_VSIZE, height)

        logger.info(
            "VDMA S2MM started: %d stores, %dx%d, stride=%d",
            len(frame_addrs), width_bytes, height, stride,
        )

    def stop(self):
        """Stop the S2MM channel (clear RS bit)."""
        cr = self._ip.read(self.S2MM_DMACR)
        self._ip.write(self.S2MM_DMACR, cr & ~self.DMACR_RS)

    def current_frame(self) -> int:
        """Return the S2MM frame store index currently being written.

        Reads PARK_PTR_REG bits [28:24] (WrFrmStore).
        """
        return (self._ip.read(self.PARK_PTR_REG) >> 24) & 0x1F

    def read_dmasr(self) -> int:
        """Read the raw S2MM_DMASR register value."""
        return self._ip.read(self.S2MM_DMASR)

    def read_status(self) -> dict:
        """Read and decode S2MM status register.

        Bit positions from PG020 Table 2-12 (S2MM_DMASR).
        Error fields use ``err_`` prefix; IRQ flags use ``irq_`` prefix.
        """
        sr = self._ip.read(self.S2MM_DMASR)
        status = {
            "raw": sr,
            "halted":        bool(sr & self.DMASR_HALTED),
            "err_internal":  bool(sr & self.DMASR_VDMA_INT_ERR),
            "err_slave":     bool(sr & self.DMASR_VDMA_SLV_ERR),
            "err_decode":    bool(sr & self.DMASR_VDMA_DEC_ERR),
            "err_sof_early": bool(sr & self.DMASR_SOF_EARLY_ERR),
            "err_eol_early": bool(sr & self.DMASR_EOL_EARLY_ERR),
            "err_sof_late":  bool(sr & self.DMASR_SOF_LATE_ERR),
            "err_eol_late":  bool(sr & self.DMASR_EOL_LATE_ERR),
            "irq_frm_cnt":  bool(sr & self.DMASR_FRM_CNT_IRQ),
            "irq_dly_cnt":  bool(sr & self.DMASR_DLY_CNT_IRQ),
            "irq_err":      bool(sr & self.DMASR_ERR_IRQ),
            "frame_count":  (sr >> 16) & 0xFF,
            "delay_count":  (sr >> 24) & 0xFF,
        }
        logger.info(
            "VDMA S2MM status: 0x%08X frames=%d errs=%s",
            sr, status["frame_count"],
            [k for k, v in status.items() if k.startswith("err_") and v],
        )
        return status
