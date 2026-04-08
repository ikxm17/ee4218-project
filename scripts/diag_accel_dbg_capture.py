"""Read the layer-0 debug capture registers added in the latest bitstream.

After running layer 0 (via set_max_layers=1), this dumps the captured
conv_res write addresses and out_buf_wr addresses. Expected: 0, 1, 2, 3
for both capture sets (strictly monotonic from zero).

If silicon shows:
  - Both sets at 0, 1, 2, 3  ->  conv3d + rmw are both correct.
                                  The bug is in the URAM primitive or
                                  its address decode (below rmw).
  - conv_res at 0, 1, 2, 3 but out_wr shifted  ->  bug is in the
                                  activation/max_pool/rmw chain.
  - conv_res shifted  ->  bug is in conv3d itself.

Usage:
    ssh kria-01 'cd ~/workspace/ee4218-project && \\
        echo asdfzxcv | sudo -S XILINX_XRT=/usr PYTHONPATH=. \\
        /opt/ee4218/ee4218-venv/bin/python3 scripts/diag_accel_dbg_capture.py'
"""
import hashlib
import pathlib
import numpy as np
from pynq import Overlay
from software.overlay.drivers.tinyissimoyolo_accelerator import TinyissimoYoloAcceleratorDriver

BIT_PATH = pathlib.Path("hardware/output/playground.bit")
MEM_PATH = pathlib.Path("hardware/testbench/inference_hdl/pixels_layer0.mem")

# Debug register offsets (axil_regs.sv)
ADDR_DBG_CONV_RES  = 0x030  # packs conv_res[1:0]: {2'd0, a1, 2'd0, a0}
ADDR_DBG_CONV_RES1 = 0x034  # packs conv_res[3:2]
ADDR_DBG_OUT_WR    = 0x038  # packs out_wr[1:0]
ADDR_DBG_OUT_WR1   = 0x03C  # packs out_wr[3:2]
ADDR_DBG_CAP_CNT   = 0x040  # {out_idx, conv_res_idx}

print(f"=== bitstream md5: {hashlib.md5(BIT_PATH.read_bytes()).hexdigest()} ===")

# Load image (matches diag_accel_layer0_max1.py)
with open(MEM_PATH) as f:
    mem_bytes = np.array([int(l.strip(), 16) for l in f if l.strip()], dtype=np.uint8)
mem_int8 = mem_bytes.view(np.int8).reshape(3, 256, 256)
image = (mem_int8.astype(np.int16) + 128).astype(np.uint8).transpose(1, 2, 0)

ol = Overlay(str(BIT_PATH), ignore_version=True)
drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)

# Stop inference after layer 0
print("\n=== set_max_layers(1) — stop after layer 0 ===")
drv.configure(mode=0)
drv.set_max_layers(1)
drv.start()

print("=== write 65536 pixels via FIFO ===")
drv.write_pixels(image)
print(f"  pixel_count={drv.pixel_count}")

print("\n=== wait for inference done ===")
if not drv.wait_done(timeout_s=3.0):
    raise TimeoutError("wait_done timed out")
print(f"  cycle_count: {drv.cycle_count}")

# Read debug capture registers
print("\n=== debug capture registers ===")
def unpack_pair(word, label0, label1):
    """Unpack two 14-bit addresses from a 32-bit word.
    Layout: {2'd0, addr1[13:0], 2'd0, addr0[13:0]} -> bits [31:18]=addr1, [13:0]=addr0.
    """
    addr0 = word & 0x3FFF
    addr1 = (word >> 16) & 0x3FFF
    print(f"  {label0} = {addr0:>5} (0x{addr0:04x})")
    print(f"  {label1} = {addr1:>5} (0x{addr1:04x})")
    return addr0, addr1

w_conv0 = drv._ip.read(ADDR_DBG_CONV_RES)
w_conv1 = drv._ip.read(ADDR_DBG_CONV_RES1)
w_out0  = drv._ip.read(ADDR_DBG_OUT_WR)
w_out1  = drv._ip.read(ADDR_DBG_OUT_WR1)
w_cnt   = drv._ip.read(ADDR_DBG_CAP_CNT)

print(f"\nRaw words: conv_res={w_conv0:08x}/{w_conv1:08x}  out_wr={w_out0:08x}/{w_out1:08x}  cnt={w_cnt:08x}")

print("\nconv3d/conv1d RES write addresses (first 4, layer 0 only):")
cr0, cr1 = unpack_pair(w_conv0, "conv_res[0]", "conv_res[1]")
cr2, cr3 = unpack_pair(w_conv1, "conv_res[2]", "conv_res[3]")

print("\nout_buf_wr addresses (first 4, post-rmw, layer 0 only):")
ow0, ow1 = unpack_pair(w_out0, "out_wr[0]", "out_wr[1]")
ow2, ow3 = unpack_pair(w_out1, "out_wr[2]", "out_wr[3]")

print(f"\ncapture count: {w_cnt & 0xF} (low = conv_res_idx, high = out_wr_idx)")

# Analyze
conv_res_seq = [cr0, cr1, cr2, cr3]
out_wr_seq   = [ow0, ow1, ow2, ow3]

print("\n=== analysis ===")
print(f"conv_res sequence: {conv_res_seq}")
print(f"  Expected (correct):  [0, 1, 2, 3]")
print(f"  If +1 shift:         [16383, 0, 1, 2]  (wraparound of the first conv write)")
print(f"out_wr sequence:   {out_wr_seq}")
print(f"  Expected (correct):  [0, 1, 2, 3]")
print(f"  If +1 shift:         [16383, 0, 1, 2]")

if conv_res_seq == [0, 1, 2, 3]:
    print("\n  conv_res: CORRECT — conv3d/conv1d writes to expected addresses")
elif conv_res_seq == [16383, 0, 1, 2]:
    print("\n  conv_res: SHIFTED — conv3d is writing to addr-1 (with wraparound)")
else:
    print(f"\n  conv_res: UNEXPECTED — {conv_res_seq}")

if out_wr_seq == [0, 1, 2, 3]:
    print("  out_wr:   CORRECT — rmw writer passes through correct addresses")
elif out_wr_seq == [16383, 0, 1, 2]:
    print("  out_wr:   SHIFTED — rmw writer (or something downstream) shifts by -1")
else:
    print(f"  out_wr:   UNEXPECTED — {out_wr_seq}")

ol.free()
