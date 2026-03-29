"""IMX219 CMOS image sensor driver — I2C access and streaming control.

Reference: IMX219PQH5-C datasheet (docs/peripherals/IMX219PQ.pdf)

I2C path (from hardware/constraints/camera.xdc):
    AXI IIC (PL) -> F10/G11 -> TCA8546A mux (0x74, ch2) -> IMX219 (0x10)

The AXI IIC controller is instantiated in the PL block design and exposed
as a Linux I2C adapter via the xiic-i2c kernel driver. The bus number is
assigned dynamically when the PYNQ overlay loads the device tree fragment.

Uses smbus2 for raw I2C messages supporting the IMX219's CCI protocol
(16-bit register addresses with 8-bit data).
"""

import logging
import pathlib
import time

from smbus2 import SMBus, i2c_msg

from . import _imx219_regs as regs

logger = logging.getLogger(__name__)


class Imx219Driver:
    """IMX219 sensor driver via AXI IIC through the KV260's I2C mux."""

    # Block design instance name — the AXI IIC controller that provides
    # the I2C bus to the sensor (for .hwh audit)
    IP_NAME = "axi_iic_0"

    # -- I2C bus addresses --
    I2C_MUX_ADDR = 0x74
    I2C_MUX_CHANNEL = 0x04  # Channel 2 bitmask
    IMX219_ADDR = 0x10

    def __init__(self, bus: int | None = None):
        """Open the I2C bus and select the mux channel to the camera.

        Args:
            bus: Linux I2C bus number (/dev/i2c-N). If None, auto-detects
                 the Xilinx AXI IIC adapter (requires overlay to be loaded).
        """
        if bus is None:
            bus = self._find_axi_iic_bus()
        self._bus_num = bus
        self._bus = SMBus(bus)
        self._select_mux_channel()
        logger.info("I2C bus %d opened, mux channel selected", bus)

    @staticmethod
    def _find_axi_iic_bus() -> int:
        """Find the I2C bus number for the Xilinx AXI IIC adapter.

        Scans /sys/class/i2c-adapter/ for adapters whose name contains
        'xiic', which is the Linux driver for Xilinx AXI IIC (PG090).
        The overlay must be loaded first so the device tree fragment
        instantiates the adapter.

        Returns:
            The bus number (N in /dev/i2c-N).

        Raises:
            RuntimeError: If no AXI IIC adapter is found.
        """
        adapter_dir = pathlib.Path("/sys/class/i2c-adapter")
        for adapter in sorted(adapter_dir.iterdir()):
            name_file = adapter / "name"
            if name_file.exists() and "xiic" in name_file.read_text():
                bus_num = int(adapter.name.split("-")[1])
                logger.debug("Found AXI IIC adapter: i2c-%d", bus_num)
                return bus_num
        raise RuntimeError(
            "No Xilinx AXI IIC adapter found in sysfs. "
            "Ensure the overlay is loaded before initializing I2C."
        )

    def _select_mux_channel(self) -> None:
        """Write channel bitmask to TCA8546A to route I2C to the IMX219."""
        msg = i2c_msg.write(self.I2C_MUX_ADDR, [self.I2C_MUX_CHANNEL])
        self._bus.i2c_rdwr(msg)
        logger.debug("I2C mux 0x%02X channel set to 0x%02X",
                      self.I2C_MUX_ADDR, self.I2C_MUX_CHANNEL)

    # -- Low-level register access --

    def write_reg(self, reg: int, value: int) -> None:
        """Write an 8-bit value to a 16-bit register address.

        CCI single write (datasheet Fig. 16, page 22):
            [S][slave_W][A] [reg_hi][A] [reg_lo][A] [data][A/NA][P]
        """
        msg = i2c_msg.write(
            self.IMX219_ADDR,
            [(reg >> 8) & 0xFF, reg & 0xFF, value & 0xFF],
        )
        self._bus.i2c_rdwr(msg)

    def read_reg(self, reg: int) -> int:
        """Read an 8-bit value from a 16-bit register address.

        CCI single read from random location (datasheet Fig. 12, page 20):
            [S][slave_W][A] [reg_hi][A] [reg_lo][A]
            [Sr][slave_R][A] [data][NA][P]
        """
        write_msg = i2c_msg.write(
            self.IMX219_ADDR,
            [(reg >> 8) & 0xFF, reg & 0xFF],
        )
        read_msg = i2c_msg.read(self.IMX219_ADDR, 1)
        self._bus.i2c_rdwr(write_msg, read_msg)
        return list(read_msg)[0]

    def write_table(self, table: list) -> None:
        """Write a sequence of (register, value) pairs.

        Args:
            table: List of (16-bit_addr, 8-bit_value) tuples.
        """
        for reg, value in table:
            self.write_reg(reg, value)

    # -- Driver interface --

    def configure(self) -> None:
        """Full sensor initialization for 1080p30 RAW10 2-lane mode.

        Sequence (datasheet Sec 8-1, Table 37):
            1. Software reset
            2. Wait for reset completion
            3. Write full register table (while in standby)
        """
        self.write_reg(regs.REG_SOFTWARE_RESET, 0x01)
        time.sleep(0.010)  # 10 ms for reset completion
        self.write_table(regs.INIT_TABLE_1080P30_RAW10_2LANE)
        logger.info("IMX219 register table written (1080p30 RAW10 2-lane)")

    def start(self) -> None:
        """Set mode_select = 1 to begin streaming."""
        self.write_reg(regs.REG_MODE_SELECT, 0x01)
        logger.info("IMX219 streaming started")

    def stop(self) -> None:
        """Set mode_select = 0 to enter software standby."""
        self.write_reg(regs.REG_MODE_SELECT, 0x00)
        logger.info("IMX219 streaming stopped")

    def read_status(self) -> dict:
        """Read MODEL_ID and return sensor identification info.

        Returns:
            Dict with 'model_id' (int) and 'detected' (bool).
        """
        id_h = self.read_reg(regs.REG_MODEL_ID_H)
        id_l = self.read_reg(regs.REG_MODEL_ID_L)
        sensor_id = (id_h << 8) | id_l
        detected = sensor_id == regs.MODEL_ID_VALUE
        if detected:
            logger.info("IMX219 sensor detected (ID: 0x%04X)", sensor_id)
        else:
            logger.error(
                "Unexpected sensor ID: 0x%04X (expected 0x%04X)",
                sensor_id, regs.MODEL_ID_VALUE,
            )
        return {"model_id": sensor_id, "detected": detected}

    def close(self) -> None:
        """Close the I2C bus."""
        if hasattr(self, "_bus"):
            self._bus.close()
            logger.info("I2C bus %d closed", self._bus_num)
