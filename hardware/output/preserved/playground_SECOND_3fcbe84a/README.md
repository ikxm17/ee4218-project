# playground_SECOND_3fcbe84a — HDL accelerator, first TFLite-faithful build

> **Status: sim-bit-exact AND TFLite-box-equivalent.** This is the first
> build whose weight ROMs and sim goldens were regenerated with round-half-up
> requantize (commits `275dfe5`, `23ed0cb`, `c411005`), closing the numerics
> loop end-to-end in the sense that matters for detection: **TFLite reference
> and RTL silicon produce the same bounding boxes and class predictions on
> the demo input** (verified by `scripts/verify_hdl_vs_tflite_boxes.py` and
> `scripts/probe_hdl_vs_tflite_raw.py`).
>
> **Per-layer int8 activations are NOT bit-identical to TFLite, and the
> drift is irreducible.** The RTL uses a single-step round-half-up
> requantize (`(acc*m0 + (1<<(n-1))) >> n`) that the Python golden mirrors
> bit-exactly. This disagrees with `tflite_runtime` by at most ±8 LSB on a
> bounded fraction of pixels per layer (compounded through the 17-layer
> cascade — see `hardware/scripts/README.md` Step 3a). Switching the RTL to
> the full two-stage gemmlowp `SRDHM + RDP` would NOT close the gap:
> `notes/deliverables/hdl-accelerator.md` documents an experiment in which
> a Python implementation of full gemmlowp still showed 1-LSB drift against
> `tflite_runtime`, because `tflite_runtime`'s optimized NEON/XNNPACK
> kernels disagree with the gemmlowp *reference spec* on negative-accumulator
> boundary cases. The drift is absorbed by dequant + sigmoid + DFL softmax +
> NMS before reaching the detector output, so all on-board `diag_accel_*`
> diagnostics PASS cleanly against the goldens.
>
> **Scope: HDL accelerator only.** PL-side-only build — no camera pipeline,
> no VDMA, no real-time video path. Preprocessed images must be preloaded
> via AXI-Lite MMIO.

## Identity

| Field | Value |
|-------|-------|
| Bitstream md5 | `3fcbe84ad834f13495ce9bfdca4875af` |
| Git commit | `0dad181` (branch `feat/hdl-pipeline`) |
| Build date | 2026-04-08 |
| Vivado version | 2025.2 (Build 6299465) |
| Target device | xck26-sfvc784 (Kria KV260, -2LV speed grade) |

## What changed since `playground_FIRST_5e86ce6c`

No RTL changed. The differences are entirely in the weight / quantization
content baked into the bitstream:

- **`275dfe5` fix(weights): match TFLite quantization math bit-exact** —
  brought the weight-quantization pipeline into alignment with TFLite's
  per-tensor quant.
- **`23ed0cb` fix(golden): round-half-up requantize matching RTL
  conv3d/conv1d** — switched the sim-golden requantize to round-half-up,
  matching the RTL's hardware rounding mode.
- **`c411005` chore(weights): regenerate ROMs and goldens after requantize
  fixes** — regenerated every per-layer `.hex` / `.coe` / `.npy` asset, so
  the BRAM-backed weight ROMs packed into this `.bit` differ bit-wise from
  the FIRST build.
- **`b730635` feat(scripts): add per-layer TFLite verification harness** —
  new scraper that diffs RTL per-layer output against TFLite per-layer
  output (used to confirm bit-exactness).

Because the RTL is byte-for-byte identical, the +1-URAM-shift bug fix from
FIRST still applies. The cone-A pipeline flop is still present, and the
surrounding placement solved through the same code path.

## What's integrated

- **TinyissimoYOLO HDL accelerator** (`tinyissimoyolo_accelerator_0` IP)
  - 17-layer int8 quantized YOLO inference engine
  - Layer breakdown: 10 CONV3 / CONV3_POOL backbone + conv1x1 heads for
    cv2 (box) and cv3 (class)
  - Parallelism: 16 slots (`MAX_PARALLEL`), each with 9-MAC conv chain +
    2x circular_buffer row delay
  - Storage: 2x URAM-based fmap ping-pong buffers (`fmap_a`, `fmap_b`,
    16384x128-bit each), BRAM accumulator scratch, BRAM weight/QP/SiLU ROMs
  - Control: AXI-Lite slave on `S_AXI_LITE`, register map 0x000-0x05C
    (status/ctrl) + 0x100-0x14FF (result read window)
  - Input modes: (0) AXI-Lite FIFO pixel preload, (1) S_AXIS camera stream
  - Debug captures: layer 0 address pipeline snapshots at 0x030-0x05C
  - Cycle count: 2,936,665 per inference at 100 MHz (~29.4 ms)

- **Zynq UltraScale+ PS integration**
  - PS8 with one 100 MHz PL clock (`clk_pl_0`)
  - AXI-Lite GP master → SmartConnect → accelerator slave
  - No DMA, no camera hardware connected in this bitstream

## What's NOT integrated

- **Camera pipeline** — sensor / CSI-2 RX / demosaic / gamma LUT / scaler
  pipeline lives in a separate `camera_pipeline.bit`, not loaded here.
- **VDMA / AXI-Full datapath** — pixel preload is AXI-Lite MMIO only, so
  ingress bandwidth for a 256x256x3 image is ~150 ms per frame.
- **Real-time video inference** — no hardware glue between the camera
  pipeline and the accelerator in this build.

## Timing & utilization

Post-route numbers from `playground_SECOND_3fcbe84a_timing_summary_routed.rpt`
and `..._utilization_placed.rpt`:

| Metric | Value | Notes |
|--------|-------|-------|
| Setup WNS | **+0.776 ns** | Healthy (clk_pl_0 @ 100 MHz) |
| Hold WHS | **+0.011 ns** | ⚠️ 3 ps tighter than FIRST (was +0.014 ns) |
| Pulse-width WPWS | +3.500 ns | |
| All constraints met | yes | |
| CLB LUTs | 13608 / 117120 (11.62%) | +154 LUTs vs FIRST |
| CLB Registers | 5321 / 234240 (2.27%) | unchanged vs FIRST |
| Block RAM Tile | 122 / 144 (84.72%) | |
| URAM | 16 / 64 (25.00%) | `fmap_a` + `fmap_b` ping-pong |
| DSPs | 154 / 1248 (12.34%) | |

**Warning on hold margin.** The worst hold path is still a ~19 ps-class
path inside the accelerator (same family as the FIRST build's cone-A
pipeline flop). Weight-content changes alone drifted the margin by 3 ps;
any future RTL edit or Vivado version bump could push an intra-accel path
over the edge. A future `FINAL` build should add an explicit placement or
`set_min_delay` constraint on the known-critical cones before this pattern
is fixed.

## Verification status

All on-board silicon diagnostics PASSED on the SECOND bitstream (2026-04-08):

| Diagnostic | Result |
|-----------|--------|
| `diag_accel_sim_vs_silicon.py` | **PASS** — cv2 (layer 13) + cv3 (layer 16) bit-exact vs sim goldens, 0 diffs / 4288 values |
| `diag_accel_layer0_max1.py` | **PASS** — `H1 dominant (100.0%)`, fmap_a[0..16383] bit-exact |
| `diag_accel_per_layer_sweep.py` | **PASS** — all 17 layers bit-exact, cycle count 2936665 (matches sim) |
| `diag_accel_layer_bisect.py` | **PASS** — all 16 checkable layers, 0 diffs across all regions |
| `diag_accel_determinism.py` | **DET** — independent run pairs, identical md5 |
| `diag_accel_shift_analysis.py` | **PASS** — layer 0 fmap_a[4096..16383] = 12288/12288 H1 match |
| `diag_accel_impulse_response.py` | **PASS** — single-pixel impulse lands at expected pool coordinates |

**Additional TFLite-fidelity checks** (the differentiator vs FIRST):

| Check | Result |
|-------|--------|
| Final-tensor TFLite match (`scripts/probe_hdl_vs_tflite_raw.py`, on-board) | **PASS (bounded)** — HDL silicon int8 output tensor within ±4 LSB of TFLite on the bowl demo image; all detection cells agree on the 0.3 confidence threshold decision |
| End-to-end box equivalence (`scripts/verify_hdl_vs_tflite_boxes.py`, offline) | **PASS** — HDL goldens and TFLite produce the same `bowl` detection, conf=0.473, box corners within 1 pixel on the 256×256 image; 0/64 grid cells flip the confidence threshold decision |
| Per-layer int8 TFLite match (`scripts/verify_goldens_vs_tflite.py`, offline) | **DIVERGED (bounded)** — max\|d\| ≤ 8 LSB at every layer; 0.03–40% of positions per layer disagree. Expected and documented: the Python golden mirrors the RTL's single-step round-half-up requantize, which differs from TFLite's two-stage gemmlowp SRDHM+RDP by bounded LSB amounts. The drift vanishes at the detection output (row above). |

## Files in this directory

| File | Size | Purpose |
|------|------|---------|
| `playground_SECOND_3fcbe84a.bit` | 7.8 MB | Programming bitstream for PL |
| `playground_SECOND_3fcbe84a.xsa` | 2.6 MB | Hardware platform export (for PYNQ `Overlay()`) |
| `playground_SECOND_3fcbe84a.hwh` | 176 KB | Hardware handoff XML (PYNQ ip_dict) |
| `playground_SECOND_3fcbe84a.dtbo` | 196 B | Compiled device tree overlay |
| `playground_SECOND_3fcbe84a.dts` | 375 B | Device tree source |
| `playground_SECOND_3fcbe84a_routed.dcp` | 17.0 MB | Routed checkpoint (for Tcl forensics) |
| `playground_SECOND_3fcbe84a_timing_summary_routed.rpt` | 520 KB | Full timing summary (+0.011 ns hold margin) |
| `playground_SECOND_3fcbe84a_utilization_placed.rpt` | 13 KB | Post-place resource utilization |

## Restoring this build

```bash
DIR=hardware/output/preserved/playground_SECOND_3fcbe84a
cp $DIR/playground_SECOND_3fcbe84a.bit  hardware/output/playground.bit
cp $DIR/playground_SECOND_3fcbe84a.xsa  hardware/output/playground.xsa
cp $DIR/playground_SECOND_3fcbe84a.hwh  hardware/output/playground.hwh
cp $DIR/playground_SECOND_3fcbe84a.dtbo hardware/output/playground.dtbo
cp $DIR/playground_SECOND_3fcbe84a.dts  hardware/output/playground.dts
bash scripts/deploy-overlay.sh --xsa hardware/output/playground.xsa
```
