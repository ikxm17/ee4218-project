"""Diagnostic: probe whether back-to-back accelerator runs produce identical output.

Three test pairs:
  - Pair A: drv.run(image) twice on the same overlay, no reset between
  - Pair B: drv.run(image) twice on the same overlay, soft_reset() between
  - Pair C: drv.run(image) twice on a fresh Overlay() each time

If A is non-deterministic, B is deterministic, and the first run of A/B/C
all match, then the bug is "stale URAM state leaks between runs because
run() doesn't restore a clean FSM state". This was Bug 1 in the Apr 8
2026 debug session — fixed by having run() call configure(mode=0) before
start(), which issues a soft_reset before set_mode. The regression test
`test_run_is_deterministic_across_back_to_back_calls` in
test_tinyissimoyolo_accelerator_onboard.py guards this fix.

Usage:
    ssh kria-01 'cd ~/workspace/ee4218-project && \\
        echo asdfzxcv | sudo -S XILINX_XRT=/usr PYTHONPATH=. \\
        /opt/ee4218/ee4218-venv/bin/python3 scripts/diag_accel_determinism.py'

History: developed during the Apr 7-8 2026 sim-vs-silicon debug session.
Originally /tmp/diag_accel6.py.
"""
import hashlib
import pathlib
import time
import numpy as np
from pynq import Overlay
from software.overlay.drivers.tinyissimoyolo_accelerator import TinyissimoYoloAcceleratorDriver

BIT_PATH = pathlib.Path("hardware/output/playground.bit")
MEM_PATH = pathlib.Path("hardware/testbench/inference_hdl/pixels_layer0.mem")

print(f"=== bitstream md5: {hashlib.md5(BIT_PATH.read_bytes()).hexdigest()} ===")

with open(MEM_PATH) as f:
    mem_bytes = np.array([int(l.strip(), 16) for l in f if l.strip()], dtype=np.uint8)
mem_int8 = mem_bytes.view(np.int8).reshape(3, 256, 256)
image = (mem_int8.astype(np.int16) + 128).astype(np.uint8).transpose(1, 2, 0)


def md5(arr):
    return hashlib.md5(arr.tobytes()).hexdigest()


# Pair A: no soft_reset between runs
print("\n=== Pair A: run x2 with NO soft_reset between runs ===")
ol = Overlay(str(BIT_PATH), ignore_version=True)
drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)
a1 = drv.run(image)["raw_table"].copy()
print(f"  A1 md5: {md5(a1)}  cv2[0,:4]={a1[0,:4].tolist()}")
a2 = drv.run(image)["raw_table"].copy()
print(f"  A2 md5: {md5(a2)}  cv2[0,:4]={a2[0,:4].tolist()}")
print(f"  identical: {np.array_equal(a1, a2)}")
ol.free()

# Pair B: soft_reset between runs
print("\n=== Pair B: run x2 WITH soft_reset between runs ===")
ol = Overlay(str(BIT_PATH), ignore_version=True)
drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)
b1 = drv.run(image)["raw_table"].copy()
print(f"  B1 md5: {md5(b1)}  cv2[0,:4]={b1[0,:4].tolist()}")
drv.soft_reset()
time.sleep(0.001)
b2 = drv.run(image)["raw_table"].copy()
print(f"  B2 md5: {md5(b2)}  cv2[0,:4]={b2[0,:4].tolist()}")
print(f"  identical: {np.array_equal(b1, b2)}")
ol.free()

# Pair C: fresh overlay each time
print("\n=== Pair C: fresh Overlay() between runs ===")
ol = Overlay(str(BIT_PATH), ignore_version=True)
drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)
c1 = drv.run(image)["raw_table"].copy()
print(f"  C1 md5: {md5(c1)}  cv2[0,:4]={c1[0,:4].tolist()}")
ol.free()
ol = Overlay(str(BIT_PATH), ignore_version=True)
drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)
c2 = drv.run(image)["raw_table"].copy()
print(f"  C2 md5: {md5(c2)}  cv2[0,:4]={c2[0,:4].tolist()}")
print(f"  identical: {np.array_equal(c1, c2)}")
ol.free()

print("\n=== summary ===")
print(f"  A (no reset between runs):    {'DET' if np.array_equal(a1, a2) else 'NONDET'}")
print(f"  B (soft_reset between runs):  {'DET' if np.array_equal(b1, b2) else 'NONDET'}")
print(f"  C (fresh overlay between):    {'DET' if np.array_equal(c1, c2) else 'NONDET'}")
print(f"  A1 == B1 == C1 (first runs):  "
      f"{np.array_equal(a1, b1) and np.array_equal(b1, c1)}")
