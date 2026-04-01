#!/usr/bin/env python3
"""Dump every tensor in a TFLite model with shape, dtype, scale, and zero-point.

Usage:
    python scripts/tflite_tensor_map.py <model.tflite>
    python scripts/tflite_tensor_map.py <model.tflite> --activations-only
"""
import argparse
import numpy as np
import tflite_runtime.interpreter as tflite


def main():
    parser = argparse.ArgumentParser(description="TFLite tensor map dump")
    parser.add_argument("model", help="Path to .tflite model")
    parser.add_argument(
        "--activations-only",
        action="store_true",
        help="Only show activation tensors (skip weights/constants)",
    )
    args = parser.parse_args()

    interp = tflite.Interpreter(model_path=args.model)
    interp.allocate_tensors()

    tensors = interp.get_tensor_details()
    inp_indices = {d["index"] for d in interp.get_input_details()}
    out_indices = {d["index"] for d in interp.get_output_details()}

    header = f"{'Idx':>4} | {'Name':<65} | {'Shape':<25} | {'DType':<8} | {'Scale':>12} | {'ZP':>5} | Role"
    print(header)
    print("-" * len(header))

    for t in tensors:
        idx = t["index"]
        name = t["name"]
        shape = str(t["shape"].tolist())
        dtype = str(t["dtype"]).replace("<class 'numpy.", "").replace("'>", "")

        qp = t.get("quantization_parameters", {})
        scales = qp.get("scales", np.array([]))
        zps = qp.get("zero_points", np.array([]))

        if len(scales) == 1:
            scale_str = f"{scales[0]:.8f}"
            zp_str = f"{zps[0]}"
        elif len(scales) > 1:
            scale_str = f"[{len(scales)} ch]"
            zp_str = f"[{len(zps)} ch]"
        else:
            scale_str = "-"
            zp_str = "-"

        role = ""
        if idx in inp_indices:
            role = "INPUT"
        elif idx in out_indices:
            role = "OUTPUT"
        elif name.startswith("tfl.pseudo_qconst"):
            role = "weight"
        elif name.startswith("arith.constant"):
            role = "const"

        if args.activations_only and role in ("weight", "const"):
            continue

        if len(name) > 65:
            name = name[:62] + "..."

        print(f"{idx:4d} | {name:<65} | {shape:<25} | {dtype:<8} | {scale_str:>12} | {zp_str:>5} | {role}")


if __name__ == "__main__":
    main()
