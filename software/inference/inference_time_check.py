"""
benchmark_inference.py — TinyissimoYOLO ONNX inference time benchmark
----------------------------------------------------------------------
Measures inference latency and throughput of your exported ONNX model
using random input tensors (no webcam or dataset required).

Reports:
  - Warmup runs (excluded from stats)
  - Mean, median, std, min, max latency in ms
  - Throughput in FPS
  - Per-stage breakdown: preprocess / inference / postprocess

Requirements:
    pip install onnxruntime numpy opencv-python

Usage:
    python benchmark_inference.py --model best.onnx
    python benchmark_inference.py --model best.onnx --runs 1000 --img-size 256
    python benchmark_inference.py --model best.onnx --use-webcam-frame
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ── Reuse preprocessing from inference script ─────────────────────────────────

def letterbox(img: np.ndarray, target: int):
    h, w    = img.shape[:2]
    scale   = target / max(h, w)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((target, target, 3), 114, dtype=np.uint8)
    pad_x   = (target - new_w) // 2
    pad_y   = (target - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def preprocess(frame: np.ndarray, img_size: int):
    img, scale, pad = letterbox(frame, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, scale, pad


def dummy_postprocess(output: np.ndarray, conf_threshold: float = 0.01) -> int:
    """Minimal postprocess just to measure decode cost — returns detection count."""
    pred        = output[0].T                          # [anchors, 4+nc]
    class_scores = pred[:, 4:]
    confidences  = class_scores.max(axis=1)
    return int((confidences >= conf_threshold).sum())


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(
    session:    ort.InferenceSession,
    input_name: str,
    img_size:   int,
    n_runs:     int,
    n_warmup:   int,
    source_frame: np.ndarray,
) -> dict:
    """
    Run n_warmup + n_runs inferences, return timing stats in milliseconds.
    Measures three stages separately:
        preprocess  — letterbox + normalise + transpose
        inference   — ONNX session.run()
        postprocess — output decode
    """
    pre_times  = []
    inf_times  = []
    post_times = []
    total_times = []

    print(f"\nWarmup ({n_warmup} runs) ...", end="", flush=True)
    for _ in range(n_warmup):
        tensor, _, _ = preprocess(source_frame, img_size)
        session.run(None, {input_name: tensor})
    print(" done")

    print(f"Benchmarking ({n_runs} runs) ...", end="", flush=True)
    for i in range(n_runs):
        t_total_start = time.perf_counter()

        # Preprocess
        t0            = time.perf_counter()
        tensor, scale, pad = preprocess(source_frame, img_size)
        t1            = time.perf_counter()

        # Inference
        outputs       = session.run(None, {input_name: tensor})
        t2            = time.perf_counter()

        # Postprocess
        _             = dummy_postprocess(outputs[0])
        t3            = time.perf_counter()

        pre_times.append((t1 - t0) * 1000)
        inf_times.append((t2 - t1) * 1000)
        post_times.append((t3 - t2) * 1000)
        total_times.append((t3 - t_total_start) * 1000)

        if (i + 1) % (n_runs // 10) == 0:
            print(".", end="", flush=True)

    print(" done\n")

    def stats(times):
        arr = np.array(times)
        return {
            "mean":   float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std":    float(np.std(arr)),
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
            "p95":    float(np.percentile(arr, 95)),
            "p99":    float(np.percentile(arr, 99)),
        }

    return {
        "preprocess":  stats(pre_times),
        "inference":   stats(inf_times),
        "postprocess": stats(post_times),
        "total":       stats(total_times),
        "throughput_fps": 1000.0 / float(np.mean(total_times)),
    }


def print_stats(label: str, s: dict, width: int = 12) -> None:
    print(f"  {label:<14} "
          f"mean={s['mean']:>{width}.3f}ms  "
          f"median={s['median']:>{width}.3f}ms  "
          f"std={s['std']:>{width}.3f}ms  "
          f"min={s['min']:>{width}.3f}ms  "
          f"max={s['max']:>{width}.3f}ms  "
          f"p95={s['p95']:>{width}.3f}ms  "
          f"p99={s['p99']:>{width}.3f}ms")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TinyissimoYOLO ONNX inference time"
    )
    parser.add_argument("--model",      type=str,   default="best.onnx",
                        help="Path to ONNX model (default: best.onnx)")
    parser.add_argument("--img-size",   type=int,   default=256,
                        help="Model input size — must match training imgsz (default: 256)")
    parser.add_argument("--runs",       type=int,   default=500,
                        help="Number of benchmark runs (default: 500)")
    parser.add_argument("--warmup",     type=int,   default=50,
                        help="Warmup runs excluded from stats (default: 50)")
    parser.add_argument("--use-webcam-frame", action="store_true",
                        help="Capture one real webcam frame as input instead of random noise")
    parser.add_argument("--camera",     type=int,   default=0,
                        help="Camera index if --use-webcam-frame (default: 0)")
    parser.add_argument("--cpu-only",   action="store_true",
                        help="Force CPU execution provider only")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Model     : {args.model}")
    print(f"Input size: {args.img_size}x{args.img_size}")
    print(f"Runs      : {args.runs}  (+ {args.warmup} warmup)")

    providers = (
        ["CPUExecutionProvider"]
        if args.cpu_only else
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    session    = ort.InferenceSession(args.model, providers=providers)
    input_name = session.get_inputs()[0].name
    out_shape  = session.get_outputs()[0].shape

    print(f"Provider  : {session.get_providers()[0]}")
    print(f"Input     : {input_name}  {session.get_inputs()[0].shape}")
    print(f"Output    : {session.get_outputs()[0].name}  {out_shape}")

    # ── Build source frame ────────────────────────────────────────────────────
    if args.use_webcam_frame:
        cap = cv2.VideoCapture(args.camera)
        ret, source_frame = cap.read()
        cap.release()
        if not ret or source_frame is None:
            print("WARNING: Could not read webcam frame, falling back to random noise.")
            source_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        else:
            print(f"Source    : webcam frame {source_frame.shape}")
    else:
        # Realistic random noise frame (same resolution as typical webcam)
        source_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        print(f"Source    : synthetic random frame {source_frame.shape}")

    # ── Run benchmark ─────────────────────────────────────────────────────────
    results = run_benchmark(
        session, input_name, args.img_size,
        n_runs=args.runs, n_warmup=args.warmup,
        source_frame=source_frame,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    sep = "─" * 110
    print(sep)
    print(f"  {'Stage':<14} {'mean':>16}   {'median':>16}   {'std':>16}   "
          f"{'min':>16}   {'max':>16}   {'p95':>16}   {'p99':>16}")
    print(sep)
    for stage in ("preprocess", "inference", "postprocess", "total"):
        print_stats(stage, results[stage])
    print(sep)
    print(f"\n  Throughput : {results['throughput_fps']:.1f} FPS  "
          f"(based on mean total latency)\n")

    # ── CPU vs GPU comparison hint ────────────────────────────────────────────
    if not args.cpu_only and "CUDA" in session.get_providers()[0]:
        print("  Tip: run with --cpu-only to compare CPU vs GPU latency.\n")
    elif args.cpu_only:
        print("  Tip: remove --cpu-only to benchmark with GPU (if available).\n")


if __name__ == "__main__":
    main()