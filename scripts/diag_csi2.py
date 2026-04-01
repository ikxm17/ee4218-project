#!/usr/bin/env python3
"""CSI-2 RX D-PHY diagnostic tool.

Reads MIPI CSI-2 RX registers to diagnose D-PHY lock issues without
running the full camera pipeline.  Optionally initializes the sensor
and performs a soft reset to test the full lock sequence.

Usage:
    python scripts/diag_csi2.py <bitstream_path> [--init-sensor]

Examples:
    # Read-only status dump (overlay must already be loaded):
    python scripts/diag_csi2.py hardware/output/camera_pipeline.bit

    # Full sequence: load overlay, init sensor, reset CSI-2, check lock:
    python scripts/diag_csi2.py hardware/output/camera_pipeline.bit --init-sensor
"""

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# CSI-2 RX register offsets — sourced from the driver to avoid duplication.
from software.overlay.drivers.csi2_rx import Csi2RxDriver as _Csi2

CSI2_CORE_CONFIG = _Csi2.CORE_CONFIG
CSI2_PROTOCOL_CONFIG = _Csi2.PROTOCOL_CONFIG
CSI2_CORE_STATUS = _Csi2.CORE_STATUS
CSI2_ISR = _Csi2.ISR
CSI2_CLK_LANE_INFO = _Csi2.CLK_LANE_INFO
CSI2_LANE0_INFO = _Csi2.LANE0_INFO
CSI2_LANE1_INFO = _Csi2.LANE1_INFO

CSI2_BASE_ADDR = 0xA0010000
CSI2_ADDR_RANGE = 0x1000


def dump_csi2_status(csi2, label: str = "") -> dict:
    """Read and print all CSI-2 RX diagnostic registers."""
    core_cfg = csi2.read(CSI2_CORE_CONFIG)
    proto_cfg = csi2.read(CSI2_PROTOCOL_CONFIG)
    core_status = csi2.read(CSI2_CORE_STATUS)
    isr = csi2.read(CSI2_ISR)
    clk_info = csi2.read(CSI2_CLK_LANE_INFO)
    lane0_info = csi2.read(CSI2_LANE0_INFO)
    lane1_info = csi2.read(CSI2_LANE1_INFO)

    pkt_count = (core_status >> 16) & 0xFFFF

    header = f"=== CSI-2 RX Status{f' ({label})' if label else ''} ==="
    print(header)
    print(f"  Core Config (0x00):     0x{core_cfg:08X}"
          f"  [enabled={bool(core_cfg & 0x01)}]")
    print(f"  Protocol Cfg (0x04):    0x{proto_cfg:08X}"
          f"  [lanes={((proto_cfg >> 3) & 0x03) + 1}]")
    print(f"  Core Status (0x10):     0x{core_status:08X}"
          f"  [packets={pkt_count}]")
    print(f"  ISR (0x24):             0x{isr:08X}")
    print(f"    Stop state (bit 17):  {bool(isr & (1 << 17))}")
    print(f"    SoT error (bit 13):   {bool(isr & (1 << 13))}")
    print(f"    SoT sync (bit 12):    {bool(isr & (1 << 12))}")
    print(f"  Clock Lane (0x3C):      0x{clk_info:08X}"
          f"  [HS={bool(clk_info & 0x01)}, stop={bool(clk_info & 0x02)}]")
    print(f"  Lane 0 (0x40):          0x{lane0_info:08X}"
          f"  [HS={bool(lane0_info & 0x01)}, stop={bool(lane0_info & 0x02)}]")
    print(f"  Lane 1 (0x44):          0x{lane1_info:08X}"
          f"  [HS={bool(lane1_info & 0x01)}, stop={bool(lane1_info & 0x02)}]")
    print()

    return {"packet_count": pkt_count, "isr": isr}


def soft_reset(csi2) -> None:
    """Perform CSI-2 RX soft reset cycle."""
    print("Performing CSI-2 RX soft reset...")
    csi2.write(CSI2_CORE_CONFIG, 0x00)        # Disable
    csi2.write(CSI2_CORE_CONFIG, 0x02)        # Assert reset
    time.sleep(0.001)
    csi2.write(CSI2_CORE_CONFIG, 0x00)        # Deassert reset
    csi2.write(CSI2_ISR, 0xFFFFFFFF)          # Clear ISR
    csi2.write(CSI2_CORE_CONFIG, 0x01)        # Re-enable
    print("Soft reset complete, core re-enabled.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bitstream", help="Path to .bit/.xsa bitstream")
    parser.add_argument("--init-sensor", action="store_true",
                        help="Initialize IMX219 and test full lock sequence")
    args = parser.parse_args()

    from pynq import MMIO, Overlay

    print(f"Loading overlay: {args.bitstream}")
    ol = Overlay(args.bitstream, ignore_version=True)
    csi2 = MMIO(CSI2_BASE_ADDR, CSI2_ADDR_RANGE)

    # Baseline status
    dump_csi2_status(csi2, "baseline")

    if args.init_sensor:
        from pynq import GPIO

        # Disable core before sensor init
        csi2.write(CSI2_CORE_CONFIG, 0x00)
        print("CSI-2 RX core disabled.\n")

        # Power on camera
        pwren = GPIO(GPIO.get_gpio_pin(0), "out")
        pwren.write(1)
        print("Camera power enabled.")
        time.sleep(0.010)

        # Init and start sensor
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
        from software.overlay.drivers import Imx219Driver

        sensor = Imx219Driver()
        if not sensor.read_status()["detected"]:
            print("ERROR: IMX219 not detected!")
            sys.exit(1)
        sensor.configure()
        sensor.start()
        print("IMX219 streaming started.")
        time.sleep(0.010)

        # Soft reset and check lock
        soft_reset(csi2)

        print("Waiting for D-PHY lock (2s timeout)...")
        deadline = time.monotonic() + 2.0
        locked = False
        while time.monotonic() < deadline:
            core_status = csi2.read(CSI2_CORE_STATUS)
            pkt_count = (core_status >> 16) & 0xFFFF
            if pkt_count > 0:
                locked = True
                break
            time.sleep(0.010)

        result = dump_csi2_status(csi2, "after reset + streaming")

        if locked:
            print("SUCCESS: D-PHY locked, packets flowing!")
        else:
            print("FAIL: No packets received.")
            print("Next steps:")
            print("  1. Flip the ribbon cable and retry")
            print("  2. Try a different camera module")
            print("  3. Verify C_HS_LINE_RATE matches sensor PLL output")

        # Cleanup
        sensor.stop()
        sensor.close()
        pwren.write(0)
    else:
        # Just do a soft reset and re-check
        soft_reset(csi2)
        time.sleep(0.200)
        dump_csi2_status(csi2, "after reset")

    ol.free()


if __name__ == "__main__":
    main()
