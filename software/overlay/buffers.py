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
        vdma_bufs:  3 x 1080p frames for VDMA triple-buffering
                    (32-bit/pixel, 10-bit RGB padded)
        viz_buf:    720p RGB8 output from Multi-Scaler (visualization)
        inf_buf:    256x256 RGB8 output from Multi-Scaler (inference)
    """

    def __init__(
        self,
        raw_width: int = 1920,
        raw_height: int = 1080,
        viz_shape: tuple = (720, 1280, 3),
        inf_shape: tuple = (256, 256, 3),
    ):
        from pynq import allocate

        logger.info("Allocating CMA buffers...")

        # VDMA frame stores: 32-bit words (10-bit RGB padded)
        self.vdma_bufs = [
            allocate(shape=(raw_height, raw_width), dtype=np.uint32)
            for _ in range(VDMA_FRAME_COUNT)
        ]
        self._raw_width = raw_width
        self._raw_height = raw_height

        # Multi-Scaler output buffers: RGB8 (3 bytes/pixel)
        self.viz_buf = allocate(shape=viz_shape, dtype=np.uint8)
        self.inf_buf = allocate(shape=inf_shape, dtype=np.uint8)

        logger.info(
            "CMA buffers allocated: %d x %dx%d VDMA, %s viz, %s inf",
            VDMA_FRAME_COUNT, raw_height, raw_width,
            viz_shape, inf_shape,
        )

    @property
    def vdma_phys_addrs(self) -> list:
        """Physical addresses for VDMA frame stores."""
        return [buf.physical_address for buf in self.vdma_bufs]

    @property
    def vdma_stride(self) -> int:
        """VDMA stride in bytes (width * bytes_per_pixel)."""
        return self._raw_width * VDMA_BYTES_PER_PIXEL

    @property
    def viz_phys_addr(self) -> int:
        return self.viz_buf.physical_address

    @property
    def inf_phys_addr(self) -> int:
        return self.inf_buf.physical_address

    def get_frame(self, buffer: str = "viz") -> np.ndarray:
        """Return a numpy view of the requested output buffer.

        Invalidates the CPU cache before reading to ensure coherency
        with PL-written data on the non-coherent HP ports.

        Args:
            buffer: "viz" for 720p visualization, "inference" for 256x256.

        Returns:
            RGB uint8 numpy array (height, width, 3).
        """
        if buffer == "viz":
            self.viz_buf.invalidate()
            return np.array(self.viz_buf, copy=False)
        elif buffer == "inference":
            self.inf_buf.invalidate()
            return np.array(self.inf_buf, copy=False)
        else:
            raise ValueError(f"Unknown buffer: {buffer!r} (expected 'viz' or 'inference')")

    def free(self) -> None:
        """Release all CMA buffers."""
        for buf in self.vdma_bufs:
            buf.freebuffer()
        self.viz_buf.freebuffer()
        self.inf_buf.freebuffer()
        logger.info("CMA buffers freed")
