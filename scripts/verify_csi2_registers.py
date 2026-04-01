#!/usr/bin/env python3
"""Non-destructive register read-back for CSI-2 RX (PG232 v6.0 verification).

Loads the overlay first (programs FPGA), then reads CSI-2 RX registers
to verify bit-field positions match PG232 v6.0.

Usage: echo <pw> | sudo -S XILINX_XRT=/usr /opt/ee4218/ee4218-venv/bin/python3 scripts/verify_csi2_registers.py
"""

import sys
from pathlib import Path

# Find the bitstream
bit_candidates = list(Path("hardware/output").glob("*.bit"))
if not bit_candidates:
    print("ERROR: No .bit file in hardware/output/. Build the hardware first.")
    sys.exit(1)
bitpath = str(bit_candidates[0])
print(f"Loading overlay: {bitpath}")

from pynq import Overlay

ol = Overlay(bitpath)
print("Overlay loaded.\n")

# Get CSI-2 RX handle — try ip_dict first, fall back to MMIO
CSI2_INST = "mipi_csi2_rx_subsyst_0"
if CSI2_INST in ol.ip_dict:
    from pynq import MMIO
    info = ol.ip_dict[CSI2_INST]
    base = info["phys_addr"]
    size = info["addr_range"]
    print(f"CSI-2 RX found in ip_dict: base=0x{base:08X}, range=0x{size:X}")
    csi2 = MMIO(base, size)
else:
    from pynq import MMIO
    base = 0xA0030000
    size = 0x2000
    print(f"CSI-2 RX not in ip_dict, using MMIO fallback: base=0x{base:08X}")
    csi2 = MMIO(base, size)

print(f"\n=== CSI-2 RX Register Read-Back (PG232 v6.0) ===\n")

# Core Config (0x00)
cc = csi2.read(0x00)
print(f"CORE_CONFIG (0x00) = 0x{cc:08X}")
print(f"  Core Enable (bit 0):  {bool(cc & 0x01)}")
print(f"  Soft Reset  (bit 1):  {bool(cc & 0x02)}\n")

# Protocol Config (0x04) — THE KEY TEST
pc = csi2.read(0x04)
print(f"PROTOCOL_CONFIG (0x04) = 0x{pc:08X}")
print(f"  Active Lanes bits[1:0]: raw={pc & 0x03} -> {(pc & 0x03) + 1} lane(s)  [NEW correct]")
print(f"  Max Lanes    bits[4:3]: raw={(pc >> 3) & 0x03} -> {((pc >> 3) & 0x03) + 1} lane(s)")
print(f"  OLD decode   bits[4:3] as active: {((pc >> 3) & 0x03) + 1} <- this was the bug\n")

# Core Status (0x10)
cs = csi2.read(0x10)
pkt = (cs >> 16) & 0xFFFF
print(f"CORE_STATUS (0x10) = 0x{cs:08X}")
print(f"  Packet Count [31:16]: {pkt}")
print(f"  SP FIFO Full   (bit 3): {bool(cs & 0x08)}")
print(f"  SP FIFO NEmpty (bit 2): {bool(cs & 0x04)}")
print(f"  Line Buf Full  (bit 1): {bool(cs & 0x02)}")
print(f"  Reset InProg   (bit 0): {bool(cs & 0x01)}\n")

# Clock Lane Info (0x3C)
cli = csi2.read(0x3C)
print(f"CLK_LANE_INFO (0x3C) = 0x{cli:08X}")
print(f"  Stop State (bit 1): {bool(cli & 0x02)}")
print(f"  Reserved   (bit 0): {bool(cli & 0x01)} <- must be 0 (confirms v6.0)\n")

# Data Lane 0 Info (0x40)
dl0 = csi2.read(0x40)
print(f"LANE0_INFO (0x40) = 0x{dl0:08X}")
print(f"  Stop State     (bit 5): {bool(dl0 & 0x20)}")
print(f"  Skewcalhs      (bit 2): {bool(dl0 & 0x04)}")
print(f"  SoT Error      (bit 1): {bool(dl0 & 0x02)}")
print(f"  SoT Sync Error (bit 0): {bool(dl0 & 0x01)}")
print(f"  OLD stop (bit 1):       {bool(dl0 & 0x02)} <- was reading SoT Error!\n")

# Data Lane 1 Info (0x44)
dl1 = csi2.read(0x44)
print(f"LANE1_INFO (0x44) = 0x{dl1:08X}")
print(f"  Stop State     (bit 5): {bool(dl1 & 0x20)}")
print(f"  SoT Error      (bit 1): {bool(dl1 & 0x02)}")
print(f"  SoT Sync Error (bit 0): {bool(dl1 & 0x01)}\n")

# ISR (0x24)
isr = csi2.read(0x24)
print(f"ISR (0x24) = 0x{isr:08X}\n")

# Image Info VC0 (0x60, 0x64)
img1 = csi2.read(0x60)
img2 = csi2.read(0x64)
print(f"IMG_INFO1_VC0 (0x60) = 0x{img1:08X}")
print(f"  Line Count [31:16]: {(img1 >> 16) & 0xFFFF}")
print(f"  Byte Count [15:0]:  {img1 & 0xFFFF}")
print(f"IMG_INFO2_VC0 (0x64) = 0x{img2:08X}")
dt = img2 & 0x3F
print(f"  Data Type [5:0]: 0x{dt:02X}", end="")
if dt == 0x2B:
    print("  -> RAW10 (expected for IMX219)")
else:
    print(f"  (sensor not streaming)" if dt == 0 else "")

print("\n=== Read-back complete ===")
