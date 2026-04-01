#!/usr/bin/env python3
"""Detect TFLite buffer aliasing by running inference and comparing raw tensor bytes.

TFLite's arena allocator reuses buffers for tensors with non-overlapping lifetimes.
After invoke(), get_tensor() returns whatever currently occupies that buffer — NOT a
snapshot of the tensor at computation time. This script reveals which tensors share
the same underlying buffer, making their intermediate values unreliable to read.

Usage:
    python scripts/tflite_buffer_alias.py <model.tflite> --image <input.jpg>
    python scripts/tflite_buffer_alias.py <model.tflite>  # uses random input
"""
import argparse
import numpy as np
import tflite_runtime.interpreter as tflite


def load_input(interp, image_path=None):
    """Prepare input tensor matching the model's expected dtype and shape."""
    inp_detail = interp.get_input_details()[0]
    shape = inp_detail["shape"]
    dtype = inp_detail["dtype"]

    if image_path:
        from PIL import Image

        img = Image.open(image_path).convert("RGB").resize((shape[2], shape[1]))
        data = np.expand_dims(np.array(img, dtype=dtype), axis=0)
    else:
        if dtype == np.uint8:
            data = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        elif dtype == np.int8:
            data = np.random.randint(-128, 128, size=shape, dtype=np.int8)
        else:
            data = np.random.randn(*shape).astype(np.float32)

    return data


def find_alias_groups(interp):
    """Find groups of tensors that share the same raw int8 bytes after invoke()."""
    tensors = interp.get_tensor_details()
    inp_indices = {d["index"] for d in interp.get_input_details()}
    out_indices = {d["index"] for d in interp.get_output_details()}

    # Only check non-weight activation tensors
    activation_indices = []
    for t in tensors:
        if t["name"].startswith(("tfl.pseudo_qconst", "arith.constant")):
            continue
        activation_indices.append(t["index"])

    checked = set()
    groups = []

    for i in activation_indices:
        if i in checked:
            continue
        ti = interp.get_tensor(i).flatten()
        group = [i]
        for j in activation_indices:
            if j <= i or j in checked:
                continue
            tj = interp.get_tensor(j).flatten()
            if ti.size == tj.size and np.array_equal(ti, tj):
                group.append(j)
        if len(group) > 1:
            for idx in group:
                checked.add(idx)
            groups.append(group)

    return groups


def short_name(name):
    """Extract a readable short name from a TFLite tensor name."""
    lower = name.lower()
    if "sigmoid" in lower:
        return "Sigmoid"
    if "multiply" in lower or "/mul" in lower:
        return "Mul"
    if "max_pool" in lower:
        return "MaxPool"
    if "convolution" in lower or ("add" in lower and "conv" in lower):
        return "Conv"
    if "softmax" in lower:
        return "Softmax"
    if "reshape" in lower:
        return "Reshape"
    if "transpose" in lower:
        return "Transpose"
    if "concat" in lower:
        return "Concat"
    if "strided" in lower:
        return "StridedSlice"
    if "sub" in lower:
        return "Sub"
    parts = name.split("/")
    return parts[-1][:25] if parts else name[:25]


def main():
    parser = argparse.ArgumentParser(description="TFLite buffer alias detector")
    parser.add_argument("model", help="Path to .tflite model")
    parser.add_argument("--image", help="Input image (optional, uses random if omitted)")
    args = parser.parse_args()

    interp = tflite.Interpreter(model_path=args.model)
    interp.allocate_tensors()

    input_data = load_input(interp, args.image)
    interp.set_tensor(interp.get_input_details()[0]["index"], input_data)
    interp.invoke()

    tensors = interp.get_tensor_details()
    tensor_map = {t["index"]: t for t in tensors}
    inp_indices = {d["index"] for d in interp.get_input_details()}
    out_indices = {d["index"] for d in interp.get_output_details()}

    groups = find_alias_groups(interp)

    print("=" * 70)
    print("TFLite Buffer Aliasing Report")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Total tensors: {len(tensors)}")
    print(f"Aliased groups found: {len(groups)}")
    print()

    if not groups:
        print("No aliased intermediate tensors detected.")
        return

    print("Aliased tensor groups (identical raw bytes after invoke):")
    print("-" * 70)

    for i, group in enumerate(groups):
        shape = interp.get_tensor(group[0]).shape
        members = []
        for idx in group:
            t = tensor_map[idx]
            sn = short_name(t["name"])
            role = ""
            if idx in inp_indices:
                role = " [INPUT]"
            elif idx in out_indices:
                role = " [OUTPUT]"
            members.append(f"{sn}#{idx}{role}")

        print(f"  Group {i+1}: {' == '.join(members)}  shape={list(shape)}")

    # Sigmoid-Mul specific analysis
    print()
    print("=" * 70)
    print("SiLU (Sigmoid * Mul) Aliasing Detail")
    print("=" * 70)
    print()
    print("TFLite fuses x*sigmoid(x) into a single SiLU kernel. The Sigmoid")
    print("intermediate is never stored separately — both tensor indices point")
    print("to the Mul (SiLU) output buffer.")
    print()
    print(f"{'Sig#':<6} {'Mul#':<6} {'Shape':<25} {'Sig scale':>12} {'Mul scale':>12} {'Sig zp':>7} {'Mul zp':>7}")
    print("-" * 80)

    for group in groups:
        sig_idx = mul_idx = None
        for idx in group:
            name = tensor_map[idx]["name"].lower()
            if "sigmoid" in name and sig_idx is None:
                sig_idx = idx
            elif ("multiply" in name or "/mul" in name) and mul_idx is None:
                mul_idx = idx
        if sig_idx is not None and mul_idx is not None:
            sqp = tensor_map[sig_idx]["quantization_parameters"]
            mqp = tensor_map[mul_idx]["quantization_parameters"]
            shape = list(interp.get_tensor(sig_idx).shape)
            ss = sqp["scales"][0] if len(sqp["scales"]) else 0
            ms = mqp["scales"][0] if len(mqp["scales"]) else 0
            sz = sqp["zero_points"][0] if len(sqp["zero_points"]) else 0
            mz = mqp["zero_points"][0] if len(mqp["zero_points"]) else 0
            print(f"{sig_idx:<6} {mul_idx:<6} {str(shape):<25} {ss:>12.8f} {ms:>12.8f} {sz:>7} {mz:>7}")

    print()
    print("WARNING: Only INPUT and OUTPUT tensors are reliable after invoke().")
    print("Intermediate tensor data may belong to a different (later) operation")
    print("that reused the same buffer slot.")


if __name__ == "__main__":
    main()
