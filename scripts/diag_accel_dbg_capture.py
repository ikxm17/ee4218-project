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
ADDR_DBG_CONV_RES  = 0x030
ADDR_DBG_CONV_RES1 = 0x034
ADDR_DBG_OUT_WR    = 0x038
ADDR_DBG_OUT_WR1   = 0x03C
ADDR_DBG_CAP_CNT   = 0x040
ADDR_DBG_POOL      = 0x044
ADDR_DBG_POOL1     = 0x048
ADDR_DBG_RMW_S0    = 0x04C
ADDR_DBG_RMW_S01   = 0x050
ADDR_DBG_RMW_BASE  = 0x054
ADDR_DBG_RMW_BASE1 = 0x058
ADDR_DBG_INPUTS    = 0x05C

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
w_pool0 = drv._ip.read(ADDR_DBG_POOL)
w_pool1 = drv._ip.read(ADDR_DBG_POOL1)
w_rmw0  = drv._ip.read(ADDR_DBG_RMW_S0)
w_rmw1  = drv._ip.read(ADDR_DBG_RMW_S01)
w_out0  = drv._ip.read(ADDR_DBG_OUT_WR)
w_out1  = drv._ip.read(ADDR_DBG_OUT_WR1)
w_cnt   = drv._ip.read(ADDR_DBG_CAP_CNT)

print(f"\nRaw words:")
print(f"  conv_res = {w_conv0:08x}/{w_conv1:08x}")
print(f"  pool_out = {w_pool0:08x}/{w_pool1:08x}")
print(f"  rmw_s0   = {w_rmw0:08x}/{w_rmw1:08x}")
print(f"  out_wr   = {w_out0:08x}/{w_out1:08x}")
print(f"  cnt      = {w_cnt:08x}")

print("\n[1] conv_res write addresses (captured at conv_res_en high):")
cr0, cr1 = unpack_pair(w_conv0, "conv_res[0]", "conv_res[1]")
cr2, cr3 = unpack_pair(w_conv1, "conv_res[2]", "conv_res[3]")

print("\n[2] pool_out_addr (captured at pool_out_valid high):")
po0, po1 = unpack_pair(w_pool0, "pool_out[0]", "pool_out[1]")
po2, po3 = unpack_pair(w_pool1, "pool_out[2]", "pool_out[3]")

print("\n[3] rmw_s0_addr (captured at rmw_s0_valid high):")
rs0, rs1 = unpack_pair(w_rmw0, "rmw_s0[0]", "rmw_s0[1]")
rs2, rs3 = unpack_pair(w_rmw1, "rmw_s0[2]", "rmw_s0[3]")

print("\n[4] out_buf_wr_addr (captured at out_buf_wr_en high):")
ow0, ow1 = unpack_pair(w_out0, "out_wr[0]", "out_wr[1]")
ow2, ow3 = unpack_pair(w_out1, "out_wr[2]", "out_wr[3]")

# Read rmw_base_addr and input snapshot
w_base0  = drv._ip.read(ADDR_DBG_RMW_BASE)
w_base1  = drv._ip.read(ADDR_DBG_RMW_BASE1)
w_inputs = drv._ip.read(ADDR_DBG_INPUTS)

print("\n[3a] rmw_base_addr (comb, snapshotted at each rmw_s0_valid capture):")
rb0, rb1 = unpack_pair(w_base0, "rmw_base[0]", "rmw_base[1]")
rb2, rb3 = unpack_pair(w_base1, "rmw_base[2]", "rmw_base[3]")

# DBG_INPUTS: [31]=0, [30:22]=h_out[8:0], [21:14]=ch_out[7:0], [13:0]=pp_wr_offset[13:0]
pp_wr_offset_snap = w_inputs & 0x3FFF
ch_out_snap       = (w_inputs >> 14) & 0xFF
h_out_snap        = (w_inputs >> 22) & 0x1FF
print(f"\n[5] rmw_base_addr inputs (snapshotted at first rmw_s0_valid):")
print(f"    curr_pp_wr_offset = {pp_wr_offset_snap} (0x{pp_wr_offset_snap:04x})")
print(f"    curr_ch_out       = {ch_out_snap}")
print(f"    h_out             = {h_out_snap}")
print(f"    computed base     = pp_wr_offset + (ch_out >> 4) * h_out * h_out")
print(f"                      = {pp_wr_offset_snap} + ({ch_out_snap >> 4}) * {h_out_snap} * {h_out_snap}")
print(f"                      = {pp_wr_offset_snap + (ch_out_snap >> 4) * h_out_snap * h_out_snap}")

print(f"\ncapture count: {w_cnt & 0xF}")

# Analyze
conv_res_seq = [cr0, cr1, cr2, cr3]
pool_out_seq = [po0, po1, po2, po3]
rmw_s0_seq   = [rs0, rs1, rs2, rs3]
out_wr_seq   = [ow0, ow1, ow2, ow3]

print("\n=== PIPELINE CAPTURE SUMMARY ===")
print(f"  conv_res  -> activation -> pool -> rmw_s0 -> out_buf_wr")
print(f"  conv_res  : {conv_res_seq}")
print(f"  pool_out  : {pool_out_seq}")
print(f"  rmw_s0    : {rmw_s0_seq}")
print(f"  out_wr    : {out_wr_seq}")
print(f"  Expected  : [0, 1, 2, 3] at each stage")

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
