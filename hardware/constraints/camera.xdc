# ===========================================================================
# camera.xdc — Pin constraints for IMX219 camera interface on KV260
# ===========================================================================
#
# Constraint file management:
#   Vivado processes all .xdc files in the constrs_1 fileset together.
#   Split constraints by subsystem (camera.xdc, timing.xdc, system.xdc, etc.)
#   and add each to the project. No #include needed — Vivado concatenates them.
#
# Source: KV260 v2 carrier board schematic (038-05058-03-A1), page 13
#         "ISP, IAS, and RPI" section — RPi camera connector J9 (15-pin FFC)
#
# MIPI pin LOCs are configured inside the MIPI CSI-2 RX IP via its
# CLK_LANE_IO_LOC, DATA_LANE0_IO_LOC, DATA_LANE1_IO_LOC properties.
# The IP auto-generates internal LOC constraints from these settings.
# The commented pin assignments below serve as documentation and backup
# in case manual override is needed.
# ===========================================================================


# ---------------------------------------------------------------------------
# Camera power enable
# ---------------------------------------------------------------------------
# Schematic: page 13, signal RPI_ENABLE (active high)
#   J9 pin 11 → SOM240_1 pin A15 → FPGA pin F11
#   Bank 45 (HDIO), VCCO = 1.8V
#   Directly controls camera module power — drive high to enable, low to disable.
#   Connected to PS GPIO EMIO bit 0 in the block design for runtime control.
set_property PACKAGE_PIN F11      [get_ports {cam_pwren}]
set_property IOSTANDARD  LVCMOS18 [get_ports {cam_pwren}]


# ---------------------------------------------------------------------------
# MIPI CSI-2 D-PHY lanes — Bank 66 (HP I/O), VCCO = 1.2V
# ---------------------------------------------------------------------------
# These are auto-constrained by the CSI-2 RX IP when configured with:
#   HP_IO_BANK_SELECTION = 66
#   CLK_LANE_IO_LOC      = J5
#   DATA_LANE0_IO_LOC    = K8
#   DATA_LANE1_IO_LOC    = L7
#
# Pin mapping (schematic page 13, RPi camera connector J9):
#
#   J9 Pin | Schematic Signal | SOM240_1 | FPGA Pin | Function
#   -------+------------------+----------+----------+-----------------
#      8   | HPA10_CC_P       | C12      | J5       | MIPI CLK lane +
#      9   | HPA10_CC_N       | C13      | H5       | MIPI CLK lane -
#      5   | HPA11_P          | B10      | K8       | MIPI Data 0 +
#      4   | HPA11_N          | B11      | J8       | MIPI Data 0 -
#      7   | HPA12_P          | A9       | L7       | MIPI Data 1 +
#      6   | HPA12_N          | A10      | L8       | MIPI Data 1 -
#
# HPA10_CC is a clock-capable (CC) pair — required for the MIPI clock lane.
# The D-PHY uses sub-LVDS signaling at 1.2V, handled internally by the IP.
# No IOSTANDARD constraint is needed — the D-PHY primitives set it.
#
# Uncomment ONLY if the IP's auto-generated constraints are insufficient:
# set_property PACKAGE_PIN J5 [get_ports {mipi_phy_if_clk_p}]
# set_property PACKAGE_PIN H5 [get_ports {mipi_phy_if_clk_n}]
# set_property PACKAGE_PIN K8 [get_ports {mipi_phy_if_data_p[0]}]
# set_property PACKAGE_PIN J8 [get_ports {mipi_phy_if_data_n[0]}]
# set_property PACKAGE_PIN L7 [get_ports {mipi_phy_if_data_p[1]}]
# set_property PACKAGE_PIN L8 [get_ports {mipi_phy_if_data_n[1]}]


# ---------------------------------------------------------------------------
# I2C — NOT constrained here
# ---------------------------------------------------------------------------
# The IMX219 sensor I2C (SCL/SDA) is NOT routed through PL fabric.
# It goes: PS I2C1 (MIO 24/25) → TCA8546A I2C switch (addr 0x74, ch 2) → J9
# All handled by the PS I2C controller — no XDC constraints needed.
