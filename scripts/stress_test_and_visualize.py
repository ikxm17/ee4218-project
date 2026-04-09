"""Stress test + visualize the TinyissimoYOLO HDL accelerator.

Runs inference N times on the same image, verifies every run produces
bit-identical detection output (determinism under load), then draws
bounding boxes on the input image using the FIRST run's detections and
saves it as `stress_test_output.jpg`.

Used to confirm the fixed bitstream (md5 5e86ce6c...) produces stable,
repeatable detection output across many back-to-back runs.

Usage:
    ssh kria-01 'cd ~/workspace/ee4218-project && \\
        echo asdfzxcv | sudo -S XILINX_XRT=/usr PYTHONPATH=. \\
        /opt/ee4218/ee4218-venv/bin/python3 scripts/stress_test_and_visualize.py'
"""
import hashlib
import pathlib
import sys
import time

import numpy as np
from PIL import Image
from pynq import Overlay

from software.overlay.drivers.tinyissimoyolo_accelerator import (
    CLASS_COLORS, CLASS_NAMES, TinyissimoYoloAcceleratorDriver,
)

BIT_PATH    = pathlib.Path("hardware/output/playground.bit")
IMAGE_PATH  = pathlib.Path("software/inference/data/input_image.jpg")
OUTPUT_PATH = pathlib.Path("stress_test_output.jpg")
NUM_RUNS    = 50

print(f"=== bitstream md5: {hashlib.md5(BIT_PATH.read_bytes()).hexdigest()} ===")
print(f"=== stress test: {NUM_RUNS} back-to-back runs ===")

# Load image (same preprocessing as run_inference_hdl.py)
img_pil = Image.open(str(IMAGE_PATH)).convert("RGB").resize((256, 256))
image = np.array(img_pil, dtype=np.uint8)
print(f"  input image: {IMAGE_PATH} → shape={image.shape}")

ol = Overlay(str(BIT_PATH), ignore_version=True)
drv = TinyissimoYoloAcceleratorDriver(ol.tinyissimoyolo_accel_0)

# Run NUM_RUNS inferences
runs = []
t_start = time.time()
for i in range(NUM_RUNS):
    result = drv.run(image, conf_thresh=0.3, nms_thresh=0.45)
    # Hash the raw table for quick bit-exact comparison
    raw_bytes = result["raw_table"].tobytes()
    raw_md5 = hashlib.md5(raw_bytes).hexdigest()
    runs.append({
        "raw_md5": raw_md5,
        "cycle_count": result["cycle_count"],
        "num_boxes": len(result["boxes"]),
        "boxes": result["boxes"],
        "scores": result["scores"],
        "class_ids": result["class_ids"],
        "class_names": result["class_names"],
    })
    if i == 0:
        first_result = result  # keep full first result for visualization
t_end = time.time()

# Verify determinism
first_md5 = runs[0]["raw_md5"]
all_match = all(r["raw_md5"] == first_md5 for r in runs)
cycle_counts = set(r["cycle_count"] for r in runs)

print(f"\n=== stress test results ===")
print(f"  total time:    {t_end - t_start:.2f} s ({NUM_RUNS / (t_end - t_start):.1f} inferences/s wall-clock)")
print(f"  raw md5:       {first_md5}")
print(f"  all match:     {all_match}")
print(f"  cycle counts:  {cycle_counts}")
print(f"  detections:    {runs[0]['num_boxes']} boxes")

if not all_match:
    print(f"\n  FAILED — raw output diverged across runs!")
    distinct_md5s = {}
    for i, r in enumerate(runs):
        distinct_md5s.setdefault(r["raw_md5"], []).append(i)
    for md5, idxs in distinct_md5s.items():
        print(f"    md5={md5} runs={idxs[:5]}{'...' if len(idxs) > 5 else ''} ({len(idxs)} total)")
    sys.exit(1)

print(f"\n  PASS — all {NUM_RUNS} runs bit-identical\n")

# Report detections
print(f"=== detections (from run 0) ===")
for i, (box, score, cls_id, cls_name) in enumerate(zip(
    runs[0]["boxes"], runs[0]["scores"], runs[0]["class_ids"], runs[0]["class_names"]
)):
    print(f"  [{i}] {cls_name:>6s}  conf={score:.3f}  box=(x={box[0]:>3d}, y={box[1]:>3d}, w={box[2]:>3d}, h={box[3]:>3d})")

# Visualize using PIL instead of OpenCV so we don't add a dependency
print(f"\n=== drawing bounding boxes ===")
from PIL import ImageDraw, ImageFont

# Convert the (256,256,3) numpy uint8 to PIL Image for drawing
vis = Image.fromarray(image).copy()
draw = ImageDraw.Draw(vis)

for box, score, cls_id in zip(runs[0]["boxes"], runs[0]["scores"], runs[0]["class_ids"]):
    x, y, w, h = box
    cls_id = int(cls_id)
    name = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
    color_bgr = CLASS_COLORS.get(cls_id, (255, 255, 255))
    # PIL uses RGB, OpenCV uses BGR — reverse
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    # Clamp to image bounds
    x = max(0, min(255, x))
    y = max(0, min(255, y))
    x2 = max(0, min(255, x + w))
    y2 = max(0, min(255, y + h))

    # Draw box (width 2)
    draw.rectangle([x, y, x2, y2], outline=color_rgb, width=2)

    # Draw label above the box
    label = f"{name} {score:.2f}"
    tb = draw.textbbox((x, max(0, y - 12)), label)
    draw.rectangle(tb, fill=color_rgb)
    draw.text((x, max(0, y - 12)), label, fill=(255, 255, 255))

# Add a header banner with the build/stress info
header_lines = [
    f"bit: {hashlib.md5(BIT_PATH.read_bytes()).hexdigest()[:16]}...",
    f"stress: {NUM_RUNS} runs, all bit-identical",
    f"cycles: {runs[0]['cycle_count']}  ({NUM_RUNS / (t_end - t_start):.1f} inf/s)",
    f"dets: {runs[0]['num_boxes']} {','.join(runs[0]['class_names'][:5])}",
]

# Upscale by 3x so the text is readable
vis_up = vis.resize((768, 768), Image.NEAREST)
draw_up = ImageDraw.Draw(vis_up)

# Draw header banner at the top
banner_h = 64
draw_up.rectangle([0, 0, 768, banner_h], fill=(0, 0, 0))
for i, line in enumerate(header_lines):
    draw_up.text((4, 2 + i * 15), line, fill=(0, 255, 0))

vis_up.save(str(OUTPUT_PATH), quality=92)
print(f"  saved: {OUTPUT_PATH}")
print(f"  size:  {OUTPUT_PATH.stat().st_size} bytes")

ol.free()
print("\n=== stress test complete ===")
