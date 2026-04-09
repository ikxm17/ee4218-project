# CPU Inference Runners

TFLite and ONNX inference runners for the host/dev-machine, plus the harnesses
used to compare CPU reference output against the HDL/HLS accelerator.

## Entry points

| Script | Purpose |
|--------|---------|
| `run_inference.py`          | Standard TFLite inference on a single image |
| `tflite_inference.py`       | Lower-level TFLite wrapper used by the demo and CLI runners |
| `run_inference_hdl.py`      | Run inference through the HDL accelerator (requires PYNQ + board) |
| `compare_hdl_vs_tflite.py`  | Side-by-side HDL vs TFLite output comparison on the board |
| `demo_runners.py`           | Multi-backend orchestrator used by the demo GUI — exposes `CPURunner`, `HDLRunner`, `TFLiteRunner`, `HLSRunner` as a common interface |
| `inference_time_check.py`   | Benchmark TFLite inference latency on the A53 cores |
| `webcam_inference.py`       | Webcam-driven inference loop for live demos |
| `quick_webcam_test.py`      | Smoke test for webcam capture + preprocessing |
| `inspect_intermediates.py`  | Dump and visualize per-layer TFLite activations for debugging |
| `visualise_outputs.py`      | Draw bounding boxes on an input image |

## Test harness

`tests/` holds pytest-based off-board regression tests that run without the
board and without the FPGA:

- `test_run_inference_hdl_offboard.py` — verifies the `run_inference_hdl.py`
  code paths that don't touch hardware (config loading, preprocessing, box
  decoding)
- `test_compare_hdl_vs_tflite_offboard.py` — exercises the comparison logic
  against synthetic reference tensors

Run with:

```bash
python -m pytest software/inference/tests/
```

## Input data

`data/input_image.jpg` is the canonical regression image used by the
equivalence gates in [`scripts/`](../../scripts/).
[`scripts/fetch-coco-demo-images.py`](../../scripts/fetch-coco-demo-images.py)
downloads additional COCO val2017 images for the demo GUI.

## Related

- **End-to-end equivalence gate** → [`scripts/verify_hdl_vs_tflite_boxes.py`](../../scripts/verify_hdl_vs_tflite_boxes.py)
- **On-board silicon vs TFLite diff** → [`scripts/diag_accel_silicon_vs_tflite.py`](../../scripts/diag_accel_silicon_vs_tflite.py)
- **Demo GUI** → [`software/gui/demo/`](../gui/demo/)
