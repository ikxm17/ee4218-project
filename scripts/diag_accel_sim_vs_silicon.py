"""Diagnostic: bit-exact compare HDL accelerator output against simulation goldens.

Run this on the Kria board to verify whether the silicon's TinyissimoYOLO
accelerator produces the same cv2/cv3 detection bytes that
`tb_tinyissimoyolo_accel.sv` produces in xsim. Reads `pixels_layer0.mem`
(the int8 channel-major dump that the testbench feeds in) and reconstructs
the equivalent uint8 HWC image so the driver gets bit-equivalent input.

Usage:
    ssh kria-01 'cd ~/workspace/ee4218-project && \\
        echo asdfzxcv | sudo -S XILINX_XRT=/usr PYTHONPATH=. \\
        /opt/ee4218/ee4218-venv/bin/python3 scripts/diag_accel_sim_vs_silicon.py'

History: developed during the Apr 7-8 2026 sim-vs-silicon debug session
that uncovered the layer 0 +1 URAM word shift bug. Originally /tmp/diag_accel2.py.
"""
import hashlib
import pathlib
import numpy as np
from pynq import Overlay
from software.overlay.drivers.tinyissimoyolo_accelerator import TinyissimoYoloAcceleratorDriver
from software.overlay.tests.checks import load_golden_uram_mem

BIT_PATH    = pathlib.Path("hardware/output/playground.bit")
MEM_PATH    = pathlib.Path("hardware/testbench/inference_hdl/pixels_layer0.mem")
GOLDEN_CV2  = pathlib.Path("hardware/testbench/inference_hdl/golden_layer13_uram.mem")
GOLDEN_CV3  = pathlib.Path("hardware/testbench/inference_hdl/golden_layer16_uram.mem")

print(f"=== bitstream md5: {hashlib.md5(BIT_PATH.read_bytes()).hexdigest()} ===")

with open(MEM_PATH) as f:
    mem_bytes = np.array([int(l.strip(), 16) for l in f if l.strip()], dtype=np.uint8)
mem_int8 = mem_bytes.view(np.int8).reshape(3, 256, 256)
mem_uint8 = (mem_int8.astype(np.int16) + 128).astype(np.uint8)
image = mem_uint8.transpose(1, 2, 0)
print(f"reconstructed image: shape={image.shape}, dtype={image.dtype}, "
      f"range=[{image.min()}, {image.max()}]")

ol = Overlay(str(BIT_PATH), ignore_version=True)
drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)
result = drv.run(image)
raw = result["raw_table"]

cv2_g = load_golden_uram_mem(str(GOLDEN_CV2), num_words=256)
cv2_match = np.array_equal(raw[0:256], cv2_g)
cv2_diffs = int((raw[0:256] != cv2_g).sum())
print(f"cv2 (layer 13) matches golden: {cv2_match}  diffs: {cv2_diffs}/{256*16}")

cv3_g = load_golden_uram_mem(str(GOLDEN_CV3), num_words=64)
cv3_match = np.array_equal(raw[256:320, :3], cv3_g[:, :3])
cv3_diffs = int((raw[256:320, :3] != cv3_g[:, :3]).sum())
print(f"cv3 (layer 16) matches golden: {cv3_match}  diffs: {cv3_diffs}/{64*3}")
print(f"cycle_count: {result['cycle_count']}")

# Side-by-side head-of-table for the first 8 cv3 rows
print("\n--- cv3 head: actual vs golden (lanes 0..2) ---")
print(f"  {'word':>4} | {'actual':<20} | {'golden':<20} | match")
print(f"  {'-'*4} | {'-'*20} | {'-'*20} | {'-'*5}")
for w in range(8):
    a = raw[256 + w, :3].tolist()
    g = cv3_g[w, :3].tolist()
    print(f"  {w:>4} | {str(a):<20} | {str(g):<20} | {'YES' if a == g else 'NO'}")

print()
if cv2_match and cv3_match:
    print("PASS: hardware == simulation (bit-exact against goldens).")
else:
    print("FAIL: HDL output diverges from sim.")

ol.free()
