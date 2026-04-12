"""Benchmark TFLite inference with and without XNNPACK delegate.

Usage (on Kria board):
    /opt/ee4218/ee4218-venv/bin/python3 bench_tflite_xnnpack.py [--model MODEL] [--runs N] [--threads T]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import tflite_runtime.interpreter as tflite

_HERE = Path(__file__).resolve().parent
_DEFAULT_MODEL = str(
    _HERE.parent / "models" / "tflite" / "tinyissimo_ptq_full_integer_quant.tflite"
)

WARMUP = 10
RUNS = 100


def make_interpreter(model_path: str, *, use_xnnpack: bool, num_threads: int):
    """Create a TFLite interpreter with or without XNNPACK."""
    if use_xnnpack:
        # Explicitly enable XNNPACK delegate
        delegate = tflite.load_delegate("libXNNPACK.so.0") if False else None
        # The portable way: use experimental_delegates or fall back to the
        # num_threads kwarg.  tflite_runtime exposes XNNPACK via the built-in
        # XNNPack delegate option when available.
        try:
            xnnpack = tflite.load_delegate("XNNPACK")
        except (ValueError, OSError):
            xnnpack = None

        if xnnpack:
            interp = tflite.Interpreter(
                model_path=model_path,
                num_threads=num_threads,
                experimental_delegates=[xnnpack],
            )
        else:
            # Fallback: XNNPACK is built-in for float models but there's no
            # separate .so to load on most tflite_runtime builds.  We signal
            # "enabled" by NOT setting the disable flag — see below.
            interp = tflite.Interpreter(
                model_path=model_path, num_threads=num_threads
            )
            print("  (XNNPACK .so not loadable — using built-in default)")
    else:
        # Disable XNNPACK by using the BUILTIN_WITHOUT_DEFAULT_DELEGATES resolver
        interp = tflite.Interpreter(
            model_path=model_path,
            num_threads=num_threads,
            experimental_op_resolver_type=tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
    interp.allocate_tensors()
    return interp


def bench(interp, input_data: np.ndarray, warmup: int, runs: int) -> list[float]:
    """Run warmup + timed inference, return per-run times in ms."""
    in_idx = interp.get_input_details()[0]["index"]
    interp.set_tensor(in_idx, input_data)

    for _ in range(warmup):
        interp.invoke()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interp.invoke()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def stats(times: list[float]) -> dict:
    a = np.array(times)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


def main():
    parser = argparse.ArgumentParser(description="TFLite XNNPACK benchmark")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    # Prepare dummy input matching the model
    tmp = tflite.Interpreter(model_path=args.model)
    tmp.allocate_tensors()
    in_det = tmp.get_input_details()[0]
    dtype = in_det["dtype"]
    shape = in_det["shape"]
    del tmp

    if dtype == np.uint8:
        input_data = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    else:
        input_data = np.random.rand(*shape).astype(np.float32)

    print(f"Model:   {Path(args.model).name}")
    print(f"Input:   {shape} {dtype.__name__}")
    print(f"Threads: {args.threads}  |  Warmup: {args.warmup}  |  Runs: {args.runs}")
    print()

    for label, use_xnn in [("WITHOUT XNNPACK", False), ("WITH XNNPACK (default)", True)]:
        print(f"--- {label} ---")
        interp = make_interpreter(args.model, use_xnnpack=use_xnn, num_threads=args.threads)
        times = bench(interp, input_data, args.warmup, args.runs)
        s = stats(times)
        print(f"  mean:   {s['mean']:7.2f} ms")
        print(f"  median: {s['median']:7.2f} ms")
        print(f"  std:    {s['std']:7.2f} ms")
        print(f"  min:    {s['min']:7.2f} ms")
        print(f"  max:    {s['max']:7.2f} ms")
        print(f"  p95:    {s['p95']:7.2f} ms")
        print(f"  p99:    {s['p99']:7.2f} ms")
        print()

    # Also test thread scaling without XNNPACK
    print("--- THREAD SCALING (no XNNPACK) ---")
    for t in [1, 2, 4]:
        interp = make_interpreter(args.model, use_xnnpack=False, num_threads=t)
        times = bench(interp, input_data, args.warmup, args.runs)
        s = stats(times)
        print(f"  {t} thread(s): mean={s['mean']:.2f}ms  median={s['median']:.2f}ms  p95={s['p95']:.2f}ms")


if __name__ == "__main__":
    main()
