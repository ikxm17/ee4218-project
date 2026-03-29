#!/usr/bin/env python3
"""Probe VPSS (v_proc_ss_0) register space to verify PG231 offsets.

The VPSS in Scaler Only mode contains three sub-IPs behind an internal
AXI interconnect, each at a fixed offset within the 256KB address space:

    H-Scaler (v_hscaler)  at +0x00000  (64KB)
    GPIO     (axi_gpio)   at +0x10000  (64KB)
    V-Scaler (v_vscaler)  at +0x20000  (64KB)

This script probes each region using write-readback to discover writable
registers, then compares against PG231 Tables 10-11.  Run this BEFORE
writing the production driver — HLS synthesis offsets may differ from docs.

Usage (on board):
    echo asdfzxcv | sudo -S XILINX_XRT=/usr \\
        /opt/ee4218/ee4218-venv/bin/python3 scripts/probe_vpss.py \\
        hardware/output/camera_pipeline.bit

Reference: PG231 v2.3, Chapter 3 Register Space (Scaler Only Mode)
"""

import argparse
import sys


# Expected register maps from PG231 (offset, name, expected_bits)
PG231_HSC = [
    (0x000, "AP_CTRL", 8),
    (0x004, "GIE", 1),
    (0x008, "IER", 2),
    (0x00C, "ISR", 2),
    (0x010, "HEIGHT", 16),
    (0x018, "WIDTH_IN", 16),
    (0x020, "WIDTH_OUT", 16),
    (0x028, "COLOR_MODE", 8),
    (0x030, "PIXEL_RATE", 32),
    (0x038, "COLOR_MODE_OUT", 8),
]

PG231_VSC = [
    (0x000, "AP_CTRL", 8),
    (0x004, "GIE", 1),
    (0x008, "IER", 2),
    (0x00C, "ISR", 2),
    (0x010, "HEIGHT_IN", 16),
    (0x018, "WIDTH", 16),
    (0x020, "HEIGHT_OUT", 16),
    (0x028, "LINE_RATE", 32),
    (0x030, "COLOR_MODE", 8),
]

PG231_GPIO = [
    (0x000, "GPIO_DATA", 2),
    (0x004, "GPIO_TRI", 2),
]


def probe_region(mmio, region_offset, start, end, step=4):
    """Probe a range of offsets for writable registers.

    For each offset: read original, write 0xFFFFFFFF, read mask, restore.
    Returns list of (offset, mask, bit_width) for non-zero masks.
    """
    hits = []
    for off in range(start, end, step):
        addr = region_offset + off
        try:
            orig = mmio.read(addr)
            mmio.write(addr, 0xFFFFFFFF)
            mask = mmio.read(addr)
            mmio.write(addr, orig)  # restore
            if mask != 0:
                bits = bin(mask).count("1")
                hits.append((off, mask, bits))
        except Exception:
            pass  # unmapped address, skip
    return hits


def match_pg231(hits, pg231_table):
    """Match probe hits against a PG231 register table."""
    pg_offsets = {entry[0]: entry for entry in pg231_table}
    results = []
    for off, mask, bits in hits:
        if off in pg_offsets:
            _, name, expected_bits = pg_offsets[off]
            match = "MATCH" if bits == expected_bits else f"MISMATCH (PG231={expected_bits}-bit)"
            results.append((off, mask, bits, name, match))
        else:
            results.append((off, mask, bits, "???", "NOT IN PG231"))
    # Check for PG231 entries not found in probe
    hit_offsets = {h[0] for h in hits}
    for pg_off, name, expected_bits in pg231_table:
        if pg_off not in hit_offsets:
            results.append((pg_off, 0, 0, name, f"MISSING (PG231 expects {expected_bits}-bit)"))
    results.sort(key=lambda x: x[0])
    return results


def print_region(title, results):
    """Pretty-print probe results for a region."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for off, mask, bits, name, status in results:
        if mask == 0:
            print(f"  0x{off:04X}: (not writable)       -> {name:20s} [{status}]")
        else:
            print(f"  0x{off:04X}: mask=0x{mask:08X} ({bits:2d}-bit) -> {name:20s} [{status}]")
    print()


def probe_coefficients(mmio, region_offset, coeff_base, expected_count, label):
    """Probe coefficient RAM region — just check first/last few words."""
    print(f"  {label} coefficient RAM at +0x{coeff_base:04X}:")
    writable = 0
    for i in range(0, expected_count * 4, 4):
        addr = region_offset + coeff_base + i
        try:
            orig = mmio.read(addr)
            mmio.write(addr, 0xFFFFFFFF)
            mask = mmio.read(addr)
            mmio.write(addr, orig)
            if mask != 0:
                writable += 1
        except Exception:
            pass
    print(f"    {writable}/{expected_count} words writable (expected: {expected_count})")
    return writable


def main():
    parser = argparse.ArgumentParser(description="Probe VPSS register space")
    parser.add_argument("bitstream", help="Path to .bit file")
    parser.add_argument(
        "--base", type=lambda x: int(x, 0), default=0xA0080000,
        help="VPSS base address (default: 0xA0080000)",
    )
    parser.add_argument(
        "--range", type=lambda x: int(x, 0), default=0x40000,
        help="Address range in bytes (default: 256KB = 0x40000)",
    )
    args = parser.parse_args()

    from pynq import MMIO, Overlay

    print(f"Loading overlay: {args.bitstream}")
    ol = Overlay(args.bitstream, ignore_version=True)
    print(f"Overlay loaded. Creating MMIO at 0x{args.base:08X} (range=0x{args.range:X})")

    mmio = MMIO(args.base, args.range)

    # ─── H-Scaler (expected at +0x00000) ───
    HSC_OFF = 0x00000
    print("\nProbing H-Scaler control registers (+0x00000, range 0x000-0x100)...")
    hsc_hits = probe_region(mmio, HSC_OFF, 0x000, 0x100)
    hsc_results = match_pg231(hsc_hits, PG231_HSC)
    print_region("H-Scaler (base+0x00000)", hsc_results)

    # H-Scaler coefficient RAM
    # 6 taps * 64 phases = 384 coefficients, packed 2 per word = 192 words
    probe_coefficients(mmio, HSC_OFF, 0x400, 192, "H-Scaler")

    # H-Scaler phase RAM
    # max 1920 output pixels, packed 2 per word = 960 words
    # But for 224 output, only 112 words needed.  Probe first 512.
    print(f"  H-Scaler phase RAM at +0x4000:")
    phase_writable = 0
    for i in range(0, 512 * 4, 4):
        addr = HSC_OFF + 0x4000 + i
        try:
            orig = mmio.read(addr)
            mmio.write(addr, 0xFFFFFFFF)
            mask = mmio.read(addr)
            mmio.write(addr, orig)
            if mask != 0:
                phase_writable += 1
        except Exception:
            pass
    print(f"    {phase_writable}/512 words writable in first 2KB")

    # ─── GPIO (expected at +0x10000) ───
    GPIO_OFF = 0x10000
    print("\nProbing GPIO (+0x10000, range 0x000-0x200)...")
    gpio_hits = probe_region(mmio, GPIO_OFF, 0x000, 0x200)
    gpio_results = match_pg231(gpio_hits, PG231_GPIO)
    print_region("GPIO (base+0x10000)", gpio_results)

    # ─── V-Scaler (expected at +0x20000) ───
    VSC_OFF = 0x20000
    print("Probing V-Scaler control registers (+0x20000, range 0x000-0x100)...")
    vsc_hits = probe_region(mmio, VSC_OFF, 0x000, 0x100)
    vsc_results = match_pg231(vsc_hits, PG231_VSC)
    print_region("V-Scaler (base+0x20000)", vsc_results)

    # V-Scaler coefficient RAM
    # 6 taps * 64 phases = 384 coefficients, packed 2 per word = 192 words
    probe_coefficients(mmio, VSC_OFF, 0x800, 192, "V-Scaler")

    # ─── Validation: write-readback test ───
    print("=" * 60)
    print("  Write-Readback Validation")
    print("=" * 60)

    # Test H-Scaler WIDTH_IN (PG231: 0x018)
    test_val = 1920  # 0x780
    for name, base, off in [
        ("HSC WIDTH_IN", HSC_OFF, 0x018),
        ("VSC HEIGHT_IN", VSC_OFF, 0x010),
    ]:
        addr = base + off
        orig = mmio.read(addr)
        mmio.write(addr, test_val)
        readback = mmio.read(addr)
        mmio.write(addr, orig)
        status = "OK" if readback == test_val else f"FAIL (got 0x{readback:X})"
        print(f"  {name} @ +0x{addr:05X}: write 0x{test_val:04X}, read 0x{readback:04X} [{status}]")

    # ─── GPIO reset test ───
    print(f"\n{'=' * 60}")
    print("  GPIO Reset Test")
    print(f"{'=' * 60}")

    gpio_data_addr = GPIO_OFF + 0x000
    gpio_tri_addr = GPIO_OFF + 0x004

    # Set as outputs
    mmio.write(gpio_tri_addr, 0x00)
    tri_val = mmio.read(gpio_tri_addr)
    print(f"  GPIO_TRI = 0x{tri_val:02X} (set to outputs)")

    # Read current state
    cur = mmio.read(gpio_data_addr)
    print(f"  GPIO_DATA current = 0x{cur:02X}")

    # Assert reset (write 0x00)
    mmio.write(gpio_data_addr, 0x00)
    val_low = mmio.read(gpio_data_addr)
    print(f"  GPIO_DATA after write 0x00 = 0x{val_low:02X} (reset asserted)")

    # Deassert reset (write 0x03)
    mmio.write(gpio_data_addr, 0x03)
    val_high = mmio.read(gpio_data_addr)
    print(f"  GPIO_DATA after write 0x03 = 0x{val_high:02X} (reset deasserted)")

    # Check scaler status after reset release
    import time
    time.sleep(0.01)
    hsc_ctrl = mmio.read(HSC_OFF + 0x000)
    vsc_ctrl = mmio.read(VSC_OFF + 0x000)
    print(f"  HSC AP_CTRL = 0x{hsc_ctrl:02X} (expect idle=0x04 after reset)")
    print(f"  VSC AP_CTRL = 0x{vsc_ctrl:02X} (expect idle=0x04 after reset)")

    print(f"\n{'=' * 60}")
    print("  Probe complete. Update driver register constants if needed.")
    print(f"{'=' * 60}")

    ol.free()


if __name__ == "__main__":
    main()
