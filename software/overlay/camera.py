"""Camera pipeline orchestrator: overlay load, sensor init, IP config.

Initialization sequence derived from:
    - IMX219 datasheet Sec 8-1, Table 36-37 (power-on timing)
    - PG232 Sec 2.3 (CSI-2 RX soft reset for D-PHY synchronization)
    - Camera bringup guide (notes/guides/camera-bringup.md)

The sequence is:
    1. Load overlay (programs FPGA, enables GPIO EMIO)
    2. Disable CSI-2 RX core (clean state before sensor streams)
    3. Drive cam_pwren high (GPIO EMIO[0] -> F11)
    4. Initialize IMX219 via I2C (mux select, ID check, register table)
    5. Allocate CMA buffers
    6. Configure PL IPs (Demosaic, Gamma, VDMA, Multi-Scaler)
    7. Start IMX219 streaming
    8. Soft-reset CSI-2 RX (D-PHY re-sync with active sensor clock)
    9. Wait for D-PHY lock, log diagnostic status
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class CameraOverlay:
    """Full camera pipeline: overlay + sensor + IP config + frame capture."""

    # IP names as they appear in the block design ip_dict.
    # Verified against tinyissimoyolo.hwh (flat hierarchy, Vivado auto-names).
    IP_CSI2_RX = "mipi_csi2_rx_subsyst_0"
    IP_DEMOSAIC = "v_demosaic_0"
    IP_GAMMA = "v_gamma_lut_0"
    IP_VDMA = "axi_vdma_0"
    IP_MULTI_SCALER = "v_multi_scaler_0"
    IP_AXI_IIC = "axi_iic_0"

    # Fallback base address for CSI-2 RX if PYNQ filters it from ip_dict.
    # PYNQ sometimes excludes subsystem IPs with DRIVERMODE=MIXED.
    _CSI2_RX_BASE_ADDR = 0xA0010000
    _CSI2_RX_ADDR_RANGE = 0x1000

    def __init__(self, bitstream_path: str):
        from pynq import DefaultIP, GPIO, MMIO, Overlay

        from . import ip_config
        from .buffers import FrameBuffers
        from .i2c import IMX219I2C

        # --- Step 1: Load overlay ---
        logger.info("Loading overlay from %s", bitstream_path)
        self._overlay = Overlay(bitstream_path, ignore_version=True)
        logger.info("Overlay loaded. IPs: %s", list(self._overlay.ip_dict.keys()))

        # PYNQ's AxiVDMA driver requires interrupt attributes (s2mm_introut)
        # that are only populated when the interrupt controller is in the
        # device tree.  Our overlay doesn't expose the VDMA interrupt via
        # the device tree (only the AXI IIC has a dtbo node), so the
        # AxiVDMA.__init__ fails with "'AxiVDMA' object has no attribute
        # 's2mm_introut'".  Since we configure the VDMA via raw MMIO
        # registers (ip_config.py), DefaultIP is sufficient.
        self._overlay._ip_map._description["ip"][self.IP_VDMA][
            "driver"
        ] = DefaultIP

        # Resolve IP handles
        self._ip_demosaic = self._resolve_ip(self.IP_DEMOSAIC)
        self._ip_gamma = self._resolve_ip(self.IP_GAMMA)
        self._ip_vdma = self._resolve_ip(self.IP_VDMA)
        self._ip_scaler = self._resolve_ip(self.IP_MULTI_SCALER)
        # AXI IIC is kernel-managed (xiic-i2c driver) — PYNQ filters it
        # from ip_dict. Its presence is validated by I2C bus auto-detection.

        # CSI-2 RX: PYNQ may filter subsystem IPs (DRIVERMODE=MIXED) from
        # ip_dict, so fall back to raw MMIO if the IP handle is missing.
        if self.IP_CSI2_RX in self._overlay.ip_dict:
            self._ip_csi2 = self._resolve_ip(self.IP_CSI2_RX)
        else:
            logger.info(
                "CSI-2 RX not in ip_dict, using MMIO at 0x%08X",
                self._CSI2_RX_BASE_ADDR,
            )
            self._ip_csi2 = MMIO(
                self._CSI2_RX_BASE_ADDR, self._CSI2_RX_ADDR_RANGE
            )

        # --- Step 2: Disable CSI-2 RX before sensor starts ---
        # The core is enabled at overlay load.  Disable it now so the
        # D-PHY doesn't try to sync against an idle bus.
        self._ip_csi2.write(ip_config.CSI2_CORE_CONFIG, 0x00)
        logger.info("CSI-2 RX core disabled (pre-streaming)")

        # --- Step 3: Camera power enable ---
        # GPIO EMIO[0] -> cam_pwren (F11), mapped to PS GPIO base + 78
        # PYNQ GPIO pin number = EMIO offset (0) via GPIO.get_gpio_pin()
        self._cam_pwren = GPIO(GPIO.get_gpio_pin(0), "out")
        self._cam_pwren.write(1)
        logger.info("Camera power enabled (GPIO EMIO[0] = high)")

        # Wait for power rail + XCLR stabilization
        # Datasheet Table 36: t3 >= 0.5 us, t5 >= 6 ms
        time.sleep(0.010)

        # --- Step 4: I2C sensor initialization ---
        # AXI IIC bus is auto-detected via sysfs (xiic-i2c adapter)
        self._i2c = IMX219I2C()
        if not self._i2c.verify_sensor_id():
            raise RuntimeError(
                "IMX219 sensor not detected. Check: (1) camera ribbon cable, "
                "(2) cam_pwren on F11, (3) AXI IIC in block design."
            )
        self._i2c.init_sensor()

        # --- Step 5: Allocate CMA buffers ---
        self._buffers = FrameBuffers()

        # --- Step 6: Configure stream-processing IPs ---
        # Demosaic and Gamma LUT must be running before data arrives so
        # they assert tready on the AXI4-Stream bus.
        ip_config.configure_demosaic(self._ip_demosaic)
        # Gamma LUT must be started for data to flow through the inline
        # AXI4-Stream path (HLS IPs hold tready low when idle).  Use a
        # linear 1:1 LUT for transparent passthrough.
        ip_config.configure_gamma_lut(self._ip_gamma, bypass=False)

        # --- Step 7: Start streaming + D-PHY lock ---
        # Start the sensor and lock the D-PHY BEFORE starting the VDMA.
        # This ensures the VDMA sees a clean, aligned stream from the
        # first SOF — starting the VDMA before the stream is established
        # causes EOL/SOF framing errors from transient startup artifacts.
        self._i2c.start_streaming()
        time.sleep(0.010)
        ip_config.reset_csi2_rx(self._ip_csi2)

        locked = ip_config.wait_for_csi2_lock(self._ip_csi2, timeout_s=2.0)
        status = ip_config.read_csi2_status(self._ip_csi2)

        # --- Step 8: Start VDMA (after stream is established) ---
        ip_config.configure_vdma_s2mm(
            self._ip_vdma,
            frame_addrs=self._buffers.vdma_phys_addrs,
            width_bytes=1920 * 4,  # 10-bit RGB in 32-bit words
            height=1080,
            stride=self._buffers.vdma_stride,
        )

        # --- Step 9: Start Multi-Scaler ---
        ip_config.configure_multi_scaler(
            self._ip_scaler,
            src_addr=self._buffers.vdma_phys_addrs[0],
            src_width=1920,
            src_height=1080,
            src_stride=self._buffers.vdma_stride,
            outputs=[
                {
                    "addr": self._buffers.inf_phys_addr,
                    "width": 256,
                    "height": 256,
                    "stride": 256 * 3,
                },
                {
                    "addr": self._buffers.viz_phys_addr,
                    "width": 1280,
                    "height": 720,
                    "stride": 1280 * 3,
                },
            ],
        )

        if not locked:
            logger.error(
                "CSI-2 RX D-PHY failed to lock. ISR=0x%08X. "
                "Check: (1) ribbon cable orientation, (2) camera module, "
                "(3) C_HS_LINE_RATE matches sensor PLL output.",
                status["isr_raw"],
            )

        logger.info("Camera pipeline initialized and streaming")

    def _resolve_ip(self, name: str):
        """Get an IP handle from the overlay by exact name."""
        if name not in self._overlay.ip_dict:
            available = list(self._overlay.ip_dict.keys())
            raise RuntimeError(
                f"IP '{name}' not found in overlay. Available: {available}"
            )
        return getattr(self._overlay, name)

    def get_frame(self, buffer: str = "viz") -> np.ndarray:
        """Read the latest frame from a Multi-Scaler output buffer.

        Args:
            buffer: "viz" for 720p visualization, "inference" for 256x256.

        Returns:
            RGB uint8 numpy array.
        """
        return self._buffers.get_frame(buffer)

    def set_gamma_bypass(self, bypass: bool) -> None:
        """Toggle the Gamma LUT bypass at runtime."""
        from . import ip_config
        ip_config.set_gamma_bypass(self._ip_gamma, bypass)

    def close(self) -> None:
        """Shutdown the camera pipeline in reverse order."""
        # Stop sensor streaming
        if hasattr(self, "_i2c"):
            try:
                self._i2c.stop_streaming()
            except Exception:
                logger.warning("Failed to stop IMX219 streaming", exc_info=True)
            self._i2c.close()

        # Free CMA buffers
        if hasattr(self, "_buffers"):
            self._buffers.free()

        # Power off camera
        if hasattr(self, "_cam_pwren"):
            try:
                self._cam_pwren.write(0)
            except Exception:
                pass

        # Free overlay
        if hasattr(self, "_overlay"):
            self._overlay.free()

        logger.info("Camera pipeline shut down")
