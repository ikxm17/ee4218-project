# RTL Testbenches

System-level and unit testbenches for the hand-written HDL pipeline and
packaged accelerator IP.

## Testbench index

| File | Scope | Notes |
|------|-------|-------|
| `tb_tinyissimoyolo_accel.sv` | End-to-end system test of the packaged accelerator IP | Drives AXI-Lite preload of URAM, starts inference, reads back the result window via AXI-Lite, and compares against `inference_hdl/golden_layer*_uram.mem` on a per-layer basis. Includes cascade bisection (Step 3.7) for localizing first-diverging layers and a Step 5a AXI-Lite readback path |
| `tb_inference_hdl.sv`        | Standalone test of `inference_hdl.sv` | Preloads layer-0 pixels via `pixels_layer0.mem` and runs the inference FSM without the AXI-Lite / packaging glue |
| `tb_conv3d.sv`               | Unit test of the `conv3d.v` engine | Exercises the 3×3 kernel cone |
| `tb_convolver.sv`            | Unit test of the generic MAC cone | Single-layer smoke test |
| `tb_ram_verify.sv`           | Memory primitive test | Verifies the `sdp_ram.sv` wrapper |

## Golden vectors

Per-layer simulation goldens live in
[`inference_hdl/`](inference_hdl/). They are generated offline by
`hardware/scripts/generate_conv3d_golden.py` from the canonical TFLite model +
quantization parameters, and consumed by `tb_tinyissimoyolo_accel.sv` and
`tb_inference_hdl.sv`.

- `pixels_layer0.mem` — INT8 input image (URAM layout)
- `golden_layer{N}_uram.mem` (N=0..16) — expected URAM content after each layer

Regenerate via:

```bash
python hardware/scripts/generate_conv3d_golden.py --lut
```

See [`hardware/scripts/README.md`](../scripts/README.md) for the full weight →
ROM → golden pipeline.

## Running a testbench headlessly

The project's Vivado XPR has `tinyissimoyolo_accel` wired up as the default
simulation top (set by `fix(testbench): prevent AXI read port hijack` —
`cb2525b`). Launch a headless simulation via:

```bash
vivado -mode batch -source scripts/run_sim_headless.tcl -nojournal -nolog
```

Per the `feedback_vivado_sim` memory, GUI simulation is the default for
interactive debugging; headless Tcl is fine for CI and regression runs.

## AXI read port hijack — cautionary note

`inference_top.sv` documents a hazard (see `result_rd_a` / `result_rd_b`): the
AXI-Lite result readback mux can steal a URAM read port from the compute path
if triggered while inference is in flight. Testbenches must only read layer-N
results *after* `STATUS.done` is asserted, or use the ping-pong window for
layers that have already completed. `tb_tinyissimoyolo_accel.sv` Step 3.7
cascade bisection respects this constraint.
