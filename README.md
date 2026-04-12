# TinyissimoYOLO on Kria KV260

EE4218 team project — FPGA-accelerated YOLO object detection on a Xilinx Kria
KV260 Vision AI starter kit. Implements **TinyissimoYOLO** (17-layer quantized
int8 CNN) across two interchangeable accelerator backends (hand-written HDL and
Vitis HLS C++), driven by PYNQ on the Cortex-A53 PS alongside an IMX219 camera
preprocessing pipeline.

## Architecture at a glance

- **Accelerator (PL)** — 17-layer TinyissimoYOLO inference engine with two
  backends selectable at runtime via the driver `set_engine` API:
  - `hdl` — hand-written SystemVerilog in [`hardware/rtl/`](hardware/rtl/)
  - `hls` — Vitis HLS C++ in [`hardware/hls/`](hardware/hls/)
- **Camera pipeline (PL)** — IMX219 CSI-2 RX → Demosaic → Gamma LUT → VDMA → DDR,
  wrapped by [`software/overlay/camera.py`](software/overlay/camera.py).
- **Quantization (host)** — PTQ from PyTorch → ONNX → TFLite full-integer int8 in
  [`software/training/`](software/training/), with per-layer weight ROM
  generation for the HDL backend.
- **Runtime (PS)** — PYNQ overlay wrapper in
  [`software/overlay/`](software/overlay/), a FastAPI WebSocket demo GUI in
  [`software/gui/demo/`](software/gui/demo/), and a CPU baseline in
  [`software/inference/`](software/inference/).
- **Verification** — bit-accurate end-to-end golden gate
  ([`scripts/verify_hdl_vs_tflite_boxes.py`](scripts/verify_hdl_vs_tflite_boxes.py))
  proving both accelerator backends produce detections matching the TFLite
  reference.

The most recent integrated build lives under
[`hardware/output/preserved/playground_FINAL/`](hardware/output/preserved/playground_FINAL/)
(`playground_hdl_hls_wrapper.{bit,hwh,xsa,dts,dtbo}`) — the first bitstream
carrying both the HDL and HLS backends together with runtime `engine_sel`.
See [`hardware/output/preserved/README.md`](hardware/output/preserved/README.md)
for the full build roster (`FIRST` / `SECOND` / `FINAL`) and restoration
instructions.

## Getting started

- **Board bring-up** → [`setup/README.md`](setup/README.md) — Ubuntu flash,
  networking, PYNQ, TFLite, ONNX
- **Weight generation + HDL build pipeline** →
  [`hardware/scripts/README.md`](hardware/scripts/README.md) — TFLite → ROM →
  golden → verify flow
- **Camera streamer** → [`software/gui/README.md`](software/gui/README.md) —
  FastAPI WebSocket frame streamer

## Running the demo GUI

The demo GUI is a browser-based inference comparison app that runs all three
backends (TFLite, HDL, HLS) on selectable images and displays bounding-box
overlays side-by-side with per-runner timing breakdowns.

### On the Kria board

Run from the **repository root**:

```bash
sudo XILINX_XRT=/usr \
    /opt/ee4218/ee4218-venv/bin/python3 \
    software/gui/demo/app.py \
    --bitstream hardware/output/preserved/playground_FINAL/playground_hdl_hls_wrapper.bit \
    --model-path software/models/tflite/tinyissimo_ptq_full_integer_quant.tflite \
    --image-dir software/models/demo_images \
    --host 0.0.0.0 --port 8000
```

Then open `http://kria-01:8000` (or `http://<board-ip>:8000`) in a browser on
the same LAN.

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--bitstream` | *(required)* | Path to `.bit` file (with matching `.hwh` alongside) |
| `--model-path` | *(required)* | TFLite full-integer quantized model |
| `--image-dir` | *(required)* | Directory of test images to select from in the UI |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Bind port |

### Notes

- **`sudo` is required** — PYNQ needs `/dev/mem` access to program the FPGA.
- **`XILINX_XRT=/usr`** must be set for PYNQ device discovery under sudo.
- The overlay and TFLite interpreter load once at startup; per-click latency is
  inference + post-processing only.

## Repository layout

```
hardware/
  rtl/              Hand-written SystemVerilog for the HDL accelerator
  hls/              Vitis HLS C++ source + synthesized IP for the HLS accelerator
  ip_repo/          Packaged IP (derived cache synced from rtl/)
  vivado/           Vivado project and block design
  weights/          Quantized weight ROMs (.mem / .coe) and layer config
  testbench/        RTL testbenches and per-layer golden vectors
  scripts/          Weight-gen and golden-gen pipeline (Python)
  constraints/      XDC pin / IO constraints
  output/           Vivado build artifacts — .bit / .hwh / .xsa (gitignored)

software/
  training/         PTQ quantization pipeline (PyTorch → ONNX → TFLite)
  inference/        TFLite / ONNX CPU inference runners and test harness
  overlay/          PYNQ overlay wrappers and PL IP drivers
  gui/              FastAPI WebSocket camera streamer + demo GUI
  models/           Trained and quantized model artifacts (.pt / .onnx / .tflite)

scripts/            On-board diagnostics + TFLite equivalence gates
setup/              KV260 board bring-up (flash, network, PYNQ, runtimes)
docs/               Reference documentation (Xilinx PGs, KV260, IMX219)
```

## Verification gates

Two offline gates run on the dev host and must pass before claiming the
accelerator matches TFLite:

```bash
# Per-layer int8 tensor diff (expected to diverge ≤ 8 LSB by design)
python scripts/verify_goldens_vs_tflite.py --max-lsb 8

# End-to-end bounding-box equivalence (load-bearing)
python scripts/verify_hdl_vs_tflite_boxes.py
```

Both scripts require Python 3.10+ (the transitive `software.overlay.drivers`
import uses PEP 604 union syntax). See
[`hardware/scripts/README.md`](hardware/scripts/README.md) for the full weight →
ROM → golden → verify pipeline and why the per-layer gate reports `WITHIN-TOL`
rather than bit-exact.

## License and credits

EE4218 team project, 2026.
