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
    6. Configure PL IPs (Demosaic, Gamma, VPSS, VDMAs)
    7. Start IMX219 streaming
    8. Soft-reset CSI-2 RX (D-PHY re-sync with active sensor clock)
    9. Start stream IPs bottom-up (VDMA1 → VPSS → VDMA0 → Gamma → Demosaic)
   10. Wait for D-PHY lock, log diagnostic status
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
    IP_VDMA1 = "axi_vdma_1"
    IP_VPSS = "v_proc_ss_0"
    IP_AXI_IIC = "axi_iic_0"

    # Fallback base address for CSI-2 RX if PYNQ filters it from ip_dict.
    # PYNQ sometimes excludes subsystem IPs with DRIVERMODE=MIXED.
    # Verified via ip_dict + register read-back: 0xA0010000, 4K range.
    _CSI2_RX_BASE_ADDR = 0xA0010000
    _CSI2_RX_ADDR_RANGE = 0x1000

    # Fallback for VPSS (also DRIVERMODE=MIXED in .hwh)
    _VPSS_BASE_ADDR = 0xA0080000
    _VPSS_ADDR_RANGE = 0x40000  # 256KB

    def __init__(
        self,
        bitstream_path: str,
        inference_size: tuple = (256, 256),
    ):
        from pynq import DefaultIP, GPIO, MMIO, Overlay

        from .buffers import FrameBuffers
        from .drivers import (
            Csi2RxDriver,
            DemosaicDriver,
            GammaLutDriver,
            Imx219Driver,
            VdmaDriver,
            VpssScalerDriver,
        )

        self._inf_w, self._inf_h = inference_size

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
        # registers, DefaultIP is sufficient.
        for vdma_name in (self.IP_VDMA, self.IP_VDMA1):
            if vdma_name in self._overlay._ip_map._description["ip"]:
                self._overlay._ip_map._description["ip"][vdma_name][
                    "driver"
                ] = DefaultIP

        # Resolve IP handles and wrap in driver classes
        self._demosaic = DemosaicDriver(self._resolve_ip(self.IP_DEMOSAIC))
        self._gamma = GammaLutDriver(self._resolve_ip(self.IP_GAMMA))
        self._vdma = VdmaDriver(self._resolve_ip(self.IP_VDMA))
        self._vdma1 = VdmaDriver(self._resolve_ip(self.IP_VDMA1))
        # AXI IIC is kernel-managed (xiic-i2c driver) — PYNQ filters it
        # from ip_dict. Its presence is validated by I2C bus auto-detection.

        # VPSS: PYNQ may filter subsystem IPs (DRIVERMODE=MIXED) from
        # ip_dict, so fall back to raw MMIO if the IP handle is missing.
        if self.IP_VPSS in self._overlay.ip_dict:
            vpss_handle = self._resolve_ip(self.IP_VPSS)
        else:
            logger.info(
                "VPSS not in ip_dict, using MMIO at 0x%08X",
                self._VPSS_BASE_ADDR,
            )
            vpss_handle = MMIO(self._VPSS_BASE_ADDR, self._VPSS_ADDR_RANGE)
        self._vpss = VpssScalerDriver(vpss_handle)

        # CSI-2 RX: same DRIVERMODE=MIXED fallback
        if self.IP_CSI2_RX in self._overlay.ip_dict:
            csi2_handle = self._resolve_ip(self.IP_CSI2_RX)
        else:
            logger.info(
                "CSI-2 RX not in ip_dict, using MMIO at 0x%08X",
                self._CSI2_RX_BASE_ADDR,
            )
            csi2_handle = MMIO(
                self._CSI2_RX_BASE_ADDR, self._CSI2_RX_ADDR_RANGE
            )
        self._csi2 = Csi2RxDriver(csi2_handle)

        # --- Step 2: Disable CSI-2 RX before sensor starts ---
        # The core is enabled at overlay load.  Disable it now so the
        # D-PHY doesn't try to sync against an idle bus.
        self._csi2.disable()
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
        self._sensor = Imx219Driver()
        sensor_status = self._sensor.read_status()
        if not sensor_status["detected"]:
            raise RuntimeError(
                "IMX219 sensor not detected. Check: (1) camera ribbon cable, "
                "(2) cam_pwren on F11, (3) AXI IIC in block design."
            )
        self._sensor.configure()

        # --- Step 5: Allocate CMA buffers ---
        self._buffers = FrameBuffers(
            inf_width=self._inf_w, inf_height=self._inf_h,
        )

        # --- Step 6: Configure stream-processing IPs (load params only) ---
        # Load parameters but do NOT rely on AP_CTRL staying set — HLS IPs
        # self-stop when started without upstream data or downstream backpressure.
        # We re-assert AP_CTRL after the full downstream path is ready.
        self._demosaic.configure()
        self._gamma.configure(bypass=False)
        self._vpss.configure(
            width_in=1920, height_in=1080,
            width_out=self._inf_w, height_out=self._inf_h,
        )

        # --- Step 7: Start streaming + D-PHY lock ---
        # Start the sensor and lock the D-PHY BEFORE starting the VDMA.
        # This ensures the VDMA sees a clean, aligned stream from the
        # first SOF — starting the VDMA before the stream is established
        # causes EOL/SOF framing errors from transient startup artifacts.
        self._sensor.start()
        time.sleep(0.010)
        self._csi2.reset()

        locked = self._csi2.wait_for_lock(timeout_s=2.0)
        status = self._csi2.read_status()

        # --- Step 8: Second CSI-2 reset ---
        # The CSI-2 RX accumulated stale framing state during steps
        # 6-7 (stream IPs self-stopped -> backpressure -> line buffer
        # overflow).  Reset it before starting VDMA so the DMA engine
        # sees only clean, properly framed data from the first SOF.
        self._csi2.reset()

        # --- Step 9: Start stream IPs (bottom-up) ---
        # The Broadcaster requires ALL outputs to assert tready before
        # data flows.  Start from the furthest downstream (VDMA sinks)
        # and work upstream so each IP has a ready consumer.

        # VDMA 1 (inference): configure first — this is downstream of VPSS,
        # which is downstream of the Broadcaster's M01 output.
        vdma1_kwargs = dict(
            frame_addrs=self._buffers.vdma1_phys_addrs,
            width_bytes=self._inf_w * 4,
            height=self._inf_h,
            stride=self._buffers.vdma1_stride,
        )
        self._vdma1.configure(**vdma1_kwargs)

        # VPSS: re-assert AP_CTRL so it asserts tready on s_axis
        self._vpss.start()

        # VDMA 0 (1080p capture): configure to assert tready on Broadcaster M00
        vdma_kwargs = dict(
            frame_addrs=self._buffers.vdma_phys_addrs,
            width_bytes=1920 * 4,
            height=1080,
            stride=self._buffers.vdma_stride,
        )
        self._vdma.configure(**vdma_kwargs)

        # Re-assert ap_ctrl on Gamma then Demosaic
        self._gamma.start()
        self._demosaic.start()

        # Wait for clean D-PHY lock with the pipeline fully running
        locked = self._csi2.wait_for_lock(timeout_s=2.0)
        status = self._csi2.read_status()

        # --- Step 10: VDMA first-frame DMAIntErr retry ---
        time.sleep(0.100)  # let first frame attempt complete

        for vdma_name, vdma_drv, kwargs in [
            ("VDMA0", self._vdma, vdma_kwargs),
            ("VDMA1", self._vdma1, vdma1_kwargs),
        ]:
            dmasr = vdma_drv.read_dmasr()
            if dmasr & 0x10:  # DMAIntErr
                logger.warning(
                    "%s DMAIntErr on first capture (DMASR=0x%08X), "
                    "resetting and retrying", vdma_name, dmasr,
                )
                vdma_drv.configure(**kwargs)

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
        """Read the latest frame from the pipeline.

        Args:
            buffer: "viz" for 720p visualization (CPU-scaled from VDMA 0),
                    "inference" for VPSS-scaled frame from VDMA 1.

        Returns:
            RGB uint8 numpy array.
        """
        return self._buffers.get_frame(buffer)

    def set_gamma_bypass(self, bypass: bool) -> None:
        """Toggle the Gamma LUT bypass at runtime."""
        self._gamma.set_bypass(bypass)

    def close(self) -> None:
        """Shutdown the camera pipeline in reverse order."""
        # Stop sensor streaming
        if hasattr(self, "_sensor"):
            try:
                self._sensor.stop()
            except Exception:
                logger.warning("Failed to stop IMX219 streaming", exc_info=True)
            self._sensor.close()

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
