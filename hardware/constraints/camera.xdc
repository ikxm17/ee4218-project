# ===========================================================================
# camera.xdc — Pin constraints for IMX219 camera interface on KV260
# ===========================================================================
#
# Constraint file management:
#   Vivado processes all .xdc files in the constrs_1 fileset together.
#   Split constraints by subsystem (camera.xdc, timing.xdc, system.xdc, etc.)
#   and add each to the project. No #include needed — Vivado concatenates them.
#
# Sources:
#   - KV260 v2 carrier board schematic (038-05058-03-A1), page 13
#     "ISP, IAS, and RPI" section — RPi camera connector J9 (15-pin FFC)
#   - Kria K26 SOM Data Sheet (DS987 v1.5), Table 11 — SOM240_1 connector pinout
#   - K26 SOM XDC (XTP685, Kria_K26_SOM_Rev1.xdc) — SOM240 pin → FPGA package pin
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
#   CLK_LANE_IO_LOC      = D7   (IO_L13P_T2L_N0_GC_QBC_66)
#   DATA_LANE0_IO_LOC    = E5   (IO_L14P_T2L_N2_GC_66)
#   DATA_LANE1_IO_LOC    = G6   (IO_L15P_T2L_N4_AD11P_66)
#
# NOTE: The IAS camera connector (U34A / AP1302 image co-processor) uses
# different pins on the same bank: G1/E1/F2 (byte lane 0, T0L). Do NOT
# confuse with the RPi connector which uses byte lane 2 (T2L).
#
# Pin mapping (schematic page 13 → DS987 Table 11 → K26 XDC):
#   Schematic: J9 pin → SOM240_1 connector signal name (HPA*)
#   DS987 Table 11: SOM240_1 pin position → signal name
#   K26 XDC (XTP685): SOM240_1 pin → FPGA package pin (Bank 66)
#
#   J9 Pin | Schematic Signal | SOM240_1 | FPGA Pin | Function
#   -------+------------------+----------+----------+-----------------
#      8   | HPA10_CC_P       | C12      | D7       | MIPI CLK lane +
#      9   | HPA10_CC_N       | C13      | D6       | MIPI CLK lane -
#      5   | HPA11_P          | B10      | E5       | MIPI Data 0 +
#      4   | HPA11_N          | B11      | D5       | MIPI Data 0 -
#      7   | HPA12_P          | A9       | G6       | MIPI Data 1 +
#      6   | HPA12_N          | A10      | F6       | MIPI Data 1 -
#
# HPA10_CC is a clock-capable (GC_QBC) pair — required for the MIPI clock
# lane. The D-PHY uses sub-LVDS signaling at 1.2V, handled internally by
# the IP. No IOSTANDARD constraint is needed — the D-PHY primitives set it.
#
# Uncomment ONLY if the IP's auto-generated constraints are insufficient:
# set_property PACKAGE_PIN D7 [get_ports {mipi_phy_if_clk_p}]
# set_property PACKAGE_PIN D6 [get_ports {mipi_phy_if_clk_n}]
# set_property PACKAGE_PIN E5 [get_ports {mipi_phy_if_data_p[0]}]
# set_property PACKAGE_PIN D5 [get_ports {mipi_phy_if_data_n[0]}]
# set_property PACKAGE_PIN G6 [get_ports {mipi_phy_if_data_p[1]}]
# set_property PACKAGE_PIN F6 [get_ports {mipi_phy_if_data_n[1]}]


# ---------------------------------------------------------------------------
# I2C — AXI IIC controller to camera connector J9
# ---------------------------------------------------------------------------
# Camera I2C is routed through PL, NOT through PS I2C1 (MIO 24/25):
#   AXI IIC (PG090) → F10 (SDA) / G11 (SCL) → level shifter → TCA8546A
#   mux (0x74, ch 2) → J9 RPi connector → IMX219 (0x10)
#
# Bank 45 (HDIO), VCCO = 1.8V — same bank as cam_pwren (F11).
# PULLUP enables weak internal pull-ups (backup for board external pull-ups).
#
# Port names must match the block design external interface. When you make
# the AXI IIC "IIC" interface external in Vivado, rename the ports to
# iic_cam_scl_io and iic_cam_sda_io (right-click port → Make External).
#
# Schematic page 13: J9 pin 14 (SCL) → SOM240_1 A16 → FPGA G11
#                    J9 pin 15 (SDA) → SOM240_1 A17 → FPGA F10
set_property PACKAGE_PIN G11      [get_ports {iic_cam_scl_io}]
set_property IOSTANDARD  LVCMOS18 [get_ports {iic_cam_scl_io}]
set_property PULLUP      TRUE     [get_ports {iic_cam_scl_io}]

set_property PACKAGE_PIN F10      [get_ports {iic_cam_sda_io}]
set_property IOSTANDARD  LVCMOS18 [get_ports {iic_cam_sda_io}]
set_property PULLUP      TRUE     [get_ports {iic_cam_sda_io}]
