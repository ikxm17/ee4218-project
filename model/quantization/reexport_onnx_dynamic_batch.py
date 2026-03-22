"""
Re-export TinyissimoYOLO v1-small .pt → ONNX with a dynamic batch dimension.

The original export via Ultralytics fixes batch=1. This script bypasses
Ultralytics' export pipeline and calls torch.onnx.export directly so that
the batch axis is left symbolic ("batch_size"), allowing any batch size at
inference time.

Usage
-----
    python reexport_onnx_dynamic_batch.py \
        --pt   path/to/weights/best.pt \
        --out  tinyissimo_v1_small_dynamic.onnx \
        --img-size 256 \
        [--opset 12]

Verification
------------
After export the script runs a quick onnxruntime check with batch sizes
1, 4, and 8 to confirm the dynamic axis works correctly.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-export TinyissimoYOLO .pt to ONNX with dynamic batch"
    )
    parser.add_argument("--pt", type=str, required=True,
                        help="Path to the trained .pt weights file")
    parser.add_argument("--out", type=str, default="tinyissimo_dynamic.onnx",
                        help="Output ONNX file path (default: tinyissimo_dynamic.onnx)")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Square input image size used during training (default: 256)")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset version (default: 12)")
    parser.add_argument("--channels", type=int, default=3,
                        help="Number of input channels - 1 for grayscale (default: 1)")
    parser.add_argument("--no-verify", action="store_true", default=False,
                        help="Skip onnxruntime verification step")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Thin wrapper: strips Ultralytics post-processing so torch.onnx is happy
# ---------------------------------------------------------------------------

class DetectWrapper(nn.Module):
    """
    Wraps an Ultralytics YOLO model so that only the raw backbone+head
    forward pass is traced, without the Ultralytics Detect post-processing
    that is not ONNX-friendly in older opsets.
    """

    def __init__(self, ultralytics_model):
        super().__init__()
        # .model is the actual nn.Sequential inside the YOLO wrapper
        self.model = ultralytics_model.model

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(args):
    pt_path = Path(args.pt)
    if not pt_path.exists():
        sys.exit(f"[ERROR] .pt file not found: {pt_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load Ultralytics YOLO model
    # ------------------------------------------------------------------
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] ultralytics is not installed. Run: pip install ultralytics")

    print(f"[info] Loading weights from: {pt_path}")
    yolo = YOLO(str(pt_path))
    yolo.model.eval().to(device)

    wrapped = DetectWrapper(yolo).eval().to(device)

    # ------------------------------------------------------------------
    # 2. Dummy input – batch=1 for tracing; batch axis made dynamic below
    # ------------------------------------------------------------------
    dummy = torch.zeros(
        1, args.channels, args.img_size, args.img_size,
        dtype=torch.float32, device=device
    )

    # ------------------------------------------------------------------
    # 3. torch.onnx.export with dynamic_axes
    # ------------------------------------------------------------------
    #   dynamic_axes format:
    #     { tensor_name: { axis_index: axis_label } }
    #
    #   We mark axis 0 of the input (and every output) as dynamic so any
    #   batch size is accepted at runtime.
    # ------------------------------------------------------------------
    print(f"[info] Tracing model …")
    with torch.no_grad():
        sample_out = wrapped(dummy)

    # Build dynamic axes dict for all outputs
    if isinstance(sample_out, (list, tuple)):
        output_names = [f"output_{i}" for i in range(len(sample_out))]
        dynamic_axes = {"images": {0: "batch_size"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}
    else:
        output_names = ["output_0"]
        dynamic_axes = {
            "images":   {0: "batch_size"},
            "output_0": {0: "batch_size"},
        }

    print(f"[info] Exporting to ONNX (opset {args.opset}) → {out_path}")
    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        opset_version=args.opset,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"[OK]   ONNX file written: {out_path}")

    return out_path, output_names


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(onnx_path, output_names, args):
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("[warn] onnxruntime not installed – skipping verification.")
        return

    try:
        import onnx
        model_proto = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_proto)
        print("[OK]   onnx.checker passed")
    except ImportError:
        print("[warn] onnx package not installed – skipping graph check.")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    print("[info] Running inference at batch sizes: 1, 4, 8 …")
    for bs in [1, 4, 8]:
        dummy_np = np.zeros(
            (bs, args.channels, args.img_size, args.img_size),
            dtype=np.float32
        )
        outputs = sess.run(None, {input_name: dummy_np})
        shapes = [o.shape for o in outputs]
        print(f"  batch={bs:2d}  output shapes: {shapes}")

    print("[OK]   Dynamic batch verification passed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    onnx_path, output_names = export(args)
    if not args.no_verify:
        verify(onnx_path, output_names, args)