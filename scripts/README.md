# On-board Diagnostics and TFLite Equivalence Gates

Root-level scripts for deploying bitstreams to the board, running on-board
diagnostics, and the offline TFLite equivalence gates that validate the HDL
accelerator.

For the **weight-gen and golden-gen pipeline** (TFLite → ROM → golden), see
[`hardware/scripts/README.md`](../hardware/scripts/README.md). This README
covers the runtime / diagnostic side.

## Deployment

| Script | Purpose |
|--------|---------|
| `deploy-overlay.sh` | Extract `.bit` / `.hwh` from a `.xsa`, generate the I2C-only camera DTBO, rsync everything to the board |
| `sync-ip-src.sh`    | Mirror `hardware/rtl/*.{v,sv}` into `hardware/ip_repo/src/` before a Vivado build (does NOT sync `.mem` / `.coe` — those must be copied manually) |
| `generate-dtbo.py`  | Build the camera-pipeline device tree overlay (I2C node only — ZOCL/AFI come from `pynq.dtbo` at boot) |
| `inspect-hwh.sh`    | Pretty-print the IPs declared in a `.hwh` file for debugging address maps |

## Offline equivalence gates (run on the dev host)

| Script | Gate |
|--------|------|
| `verify_goldens_vs_tflite.py` | Per-layer int8 tensor diff between `hardware/testbench/inference_hdl/golden_layer*_uram.mem` and TFLite intermediates. Supports `--max-lsb N` to allow rounding-boundary slack (expected `WITHIN-TOL` up to 8 LSB) |
| `verify_hdl_vs_tflite_boxes.py` | **Load-bearing**: end-to-end detection equivalence gate. Runs both the HDL golden pipeline and TFLite to final bounding boxes, confirms detections match at 0.3 confidence threshold within a 1-pixel corner tolerance |

Both require Python 3.10+ (see
[`../software/overlay/README.md#dev-host-caveat`](../software/overlay/README.md#dev-host-caveat)).

## On-board diagnostics (run as root on the board)

Run via:

```bash
echo <passwd> | sudo -S XILINX_XRT=/usr /opt/ee4218/ee4218-venv/bin/python3 \
  scripts/<script>.py
```

| Script | Purpose |
|--------|---------|
| `diag_accel_silicon_vs_tflite.py`    | **Primary on-board gate** — per-layer URAM readback diff vs TFLite reference, supports `--engine {hdl,hls,both}` |
| `diag_accel_layer_bisect.py`         | Binary-search the first layer where silicon diverges from TFLite |
| `diag_accel_layer0_full_capture.py`  | Full layer-0 URAM snapshot (all 128×128×16 bytes) |
| `diag_accel_layer0_max1.py`          | Layer-0 with `max_layers_run=1` to isolate layer-0 compute |
| `diag_accel_per_layer_sweep.py`      | Sweep `max_layers_run` from 1..17, capturing each layer's output |
| `diag_accel_shift_analysis.py`       | Quantization parameter sweep to localize rounding divergence |
| `diag_accel_impulse_response.py`     | Synthetic inputs (zeros / ones / max) to exercise rare code paths |
| `diag_accel_determinism.py`          | Run the same inference 10× to check bit-exact repeatability |
| `diag_accel_dbg_capture.py`          | Capture debug probe points during a run |
| `diag_accel_full_preload_sweep.py`   | URAM preload permutation sweep |
| `diag_accel_preload_check.py`        | Verify the preload path writes the bytes the HDL reads |
| `diag_accel_sim_vs_silicon.py`       | Compare simulation golden output against live silicon readback |
| `probe_hdl_vs_tflite_raw.py`         | Dump raw final-layer tensor from both HDL and TFLite, byte-level diff |
| `diag_csi2.py`                       | Read CSI-2 RX registers to verify D-PHY lock and lane config |
| `verify_csi2_registers.py`           | Cross-check CSI-2 register values against expected defaults |
| `probe_vpss.py`                      | Probe VPSS Multi-Scaler register space (probing only — IP is hardware-bypassed) |
| `stress_test_and_visualize.py`       | Inference loop with live bounding-box overlay for stress/latency testing |

## TFLite host utilities

| Script | Purpose |
|--------|---------|
| `tflite_buffer_alias.py`   | Alias TFLite buffer memory to avoid reallocation between runs |
| `tflite_tensor_map.py`     | Map TFLite op indices to human-readable layer names |
| `fetch-coco-demo-images.py`| Download COCO val2017 images for the demo GUI |

## Vivado TCL helpers

| Script | Purpose |
|--------|---------|
| `rebuild_bitstream.tcl` | Full headless synth + impl + bitstream rebuild (OOC-aware). **Heavy** — takes tens of minutes |
| `run_sim_headless.tcl`  | Launch `xsim` on the default simulation top without opening the GUI |
| `uram_attr_check.tcl`   | Inspect URAM attributes post-PAR to verify placement and configuration |
| `uram_path_check.tcl`   | Verify URAM connectivity in the routed design |
| `probe_uram_hier.tcl`   | Hierarchy debug for URAM placement |
| `dsp48_attr_check.tcl`  | Inspect DSP48 attributes to verify INIT, PATTERN, and USE_MULT settings |
