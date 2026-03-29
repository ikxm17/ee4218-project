"""CMA buffer allocation and frame readout for the camera pipeline.

Manages physically contiguous DDR buffers required by the VDMA and
Multi-Scaler IPs.  Uses PYNQ's allocate() which returns PynqBuffer
objects backed by Linux CMA (contiguous memory allocator).

Cache coherency: The HP ports (S_AXI_HP0/HP1) are non-coherent.
When PL writes to DDR and PS reads it, the CPU cache may hold stale
data.  Every read must be preceded by .invalidate() to flush the
CPU cache for that buffer region.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# VDMA writes 10-bit RGB at 1 sample/clock.  Demosaic outputs 3 channels
# of 10-bit data.  VDMA packs this as 32-bit words (30 bits used, 2 padded).
# Stride = width * 4 bytes/pixel.
VDMA_BYTES_PER_PIXEL = 4
VDMA_FRAME_COUNT = 3  # Triple-buffering (matches block design config)


class FrameBuffers:
    """Manages CMA-allocated frame buffers for the camera pipeline.

    Buffers:
        vdma_bufs:      3 x 1080p frames for VDMA 0 triple-buffering
                        (32-bit/pixel, 10-bit RGB padded)
        vdma1_bufs:     3 x inference-sized frames for VDMA 1 (VPSS output)
                        (32-bit/pixel, 10-bit RGB padded)
    """

    def __init__(
        self,
        raw_width: int = 1920,
        raw_height: int = 1080,
        inf_width: int = 224,
        inf_height: int = 224,
    ):
        from pynq import allocate

        logger.info("Allocating CMA buffers...")

        # VDMA 0 frame stores: 1080p, 32-bit words (10-bit RGB padded)
        self.vdma_bufs = [
            allocate(shape=(raw_height, raw_width), dtype=np.uint32)
            for _ in range(VDMA_FRAME_COUNT)
        ]
        self._raw_width = raw_width
        self._raw_height = raw_height

        # VDMA 1 frame stores: inference-sized, 32-bit words (VPSS output)
        self.vdma1_bufs = [
            allocate(shape=(inf_height, inf_width), dtype=np.uint32)
            for _ in range(VDMA_FRAME_COUNT)
        ]
        self._inf_width = inf_width
        self._inf_height = inf_height

        logger.info(
            "CMA buffers allocated: %d x %dx%d VDMA0, %d x %dx%d VDMA1",
            VDMA_FRAME_COUNT, raw_height, raw_width,
            VDMA_FRAME_COUNT, inf_height, inf_width,
        )

    @property
    def vdma_phys_addrs(self) -> list:
        """Physical addresses for VDMA frame stores."""
        return [buf.physical_address for buf in self.vdma_bufs]

    @property
    def vdma_stride(self) -> int:
        """VDMA 0 stride in bytes (width * bytes_per_pixel)."""
        return self._raw_width * VDMA_BYTES_PER_PIXEL

    @property
    def vdma1_phys_addrs(self) -> list:
        """Physical addresses for VDMA 1 (inference) frame stores."""
        return [buf.physical_address for buf in self.vdma1_bufs]

    @property
    def vdma1_stride(self) -> int:
        """VDMA 1 stride in bytes (inf_width * bytes_per_pixel)."""
        return self._inf_width * VDMA_BYTES_PER_PIXEL

    @staticmethod
    def _unpack_rgbx10(raw: np.ndarray) -> np.ndarray:
        """Unpack RGBX10 32-bit pixels to RGB uint8.

        RGBX10 format: [31:30]=pad, [29:20]=B, [19:10]=G, [9:0]=R
        Each channel is 10-bit, right-shifted by 2 to get 8-bit.
        """
        h, w = raw.shape
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = (raw & 0x3FF) >> 2
        rgb[:, :, 1] = (raw >> 10 & 0x3FF) >> 2
        rgb[:, :, 2] = (raw >> 20 & 0x3FF) >> 2
        return rgb

    def get_frame(self, buffer: str = "viz") -> np.ndarray:
        """Read a frame, unpack 10-bit to 8-bit.

        For "inference": reads VPSS-scaled frame from VDMA 1 (hardware scaled).
        For "viz": reads from VDMA 0 and CPU-scales to 720p.

        Returns:
            RGB uint8 numpy array.
        """
        if buffer == "inference":
            self.vdma1_bufs[0].invalidate()
            raw = np.array(self.vdma1_bufs[0], copy=False)
            return self._unpack_rgbx10(raw)

        import cv2

        # Visualization: CPU-scale from VDMA 0 (1080p)
        self.vdma_bufs[0].invalidate()
        raw = np.array(self.vdma_bufs[0], copy=False)
        raw = raw[::2, ::2]  # subsample to 540×960 for speed
        rgb = self._unpack_rgbx10(raw)
        return cv2.resize(rgb, (1280, 720), interpolation=cv2.INTER_LINEAR)

    def free(self) -> None:
        """Release all CMA buffers."""
        for buf in self.vdma_bufs:
            buf.freebuffer()
        for buf in self.vdma1_bufs:
            buf.freebuffer()
        logger.info("CMA buffers freed")
