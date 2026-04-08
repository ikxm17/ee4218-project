# hardware/scripts — weight & golden generation

Task-oriented runbook for regenerating every synthesis-ready asset the
TinyissimoYOLO HDL accelerator consumes: the int8 weight / bias / multiplier
ROMs baked into BRAM, the per-layer SiLU LUTs, the SystemVerilog layer
config header, and the per-layer sim goldens the RTL testbench diffs against.

For the *why* — quantization math, URAM packing, SiLU derivation, the
round-half-up requantize fix story — read
[`notes/deliverables/hdl-accelerator.md`](../../notes/deliverables/hdl-accelerator.md).
This README is the *how*.

## Canonical TFLite reference model

```
software/models/tflite/tinyissimo_ptq_full_integer_quant.tflite
```

This specific file is the **only** TFLite model the HDL generators read.
It is a full-integer quantized TinyissimoYOLO with:

- int8 weights, per-channel weight scales
- int8 activations, per-tensor activation scales, `zp_in = -128`
- int32 per-channel biases

It is produced upstream by [`software/training/quantize.py`](../../software/training/quantize.py),
which runs PTQ on the PyTorch checkpoint, exports ONNX, and then shells out
to `onnx2tf` for the ONNX → TFLite conversion. The other `.tflite` variants
in `software/models/tflite/` (float32, float16, int16-activation) are
experimental and are **not** validated against the HDL.

---

## Step 1 — Regenerate weight ROMs

```bash
python hardware/scripts/generate_hdl_weights.py \
    software/models/tflite/tinyissimo_ptq_full_integer_quant.tflite \
    --out hardware/weights/hdl/
```

`generate_hdl_weights.py` writes the following into `hardware/weights/hdl/`:

| File | Contents |
|------|----------|
| `weight_rom.{coe,mem}` | 128-bit packed conv2d weight ROM, all 17 layers concatenated |
| `bias_rom.{coe,mem}` | int32 per-channel bias with `-zp_in × Σ(weight)` correction baked in |
| `m0_rom.{coe,mem}` | Q31 per-channel multiplier, frexp-decomposed to match gemmlowp |
| `nshift_rom.{coe,mem}` | per-channel right-shift amount |
| `zp_in_rom.{coe,mem}`, `zp_out_rom.{coe,mem}` | per-layer zero points |
| `silu_lut.{coe,mem}` | precomputed SiLU LUT (17 × 256 entries), matches TFLite's LOGISTIC+MUL two-stage path |
| `layer_config.svh` | SystemVerilog header with the 17-layer config table |
| `rom_summary.json` | human-readable offset + size metadata |
| `weight_rom_golden.npz` | unpacked weights as a NumPy archive (consumed by Step 2) |

Pass `--verify` to also run an in-script round-trip check after generation.

**Critical subtlety — bias correction baked into the ROM.** TFLite computes
`acc = Σ((input - zp_in) × weight) + bias_raw`. The HDL MAC implements the
simpler `acc = Σ(input × weight) + bias_adjusted` where
`bias_adjusted = bias_raw - zp_in × Σ(weight)`. The generator applies this
correction before packing the bias ROM. This is the reason you must
regenerate the ROMs whenever the weights or `zp_in` change.

## Step 2 — Regenerate sim goldens

```bash
python hardware/scripts/generate_conv3d_golden.py \
    --model software/models/tflite/tinyissimo_ptq_full_integer_quant.tflite \
    --image software/inference/data/input_image.jpg \
    --golden hardware/weights/hdl/weight_rom_golden.npz \
    --lut hardware/weights/hdl/silu_lut.mem \
    --out hardware/testbench/inference_hdl/ \
    --num-layers 17
```

`--lut` must point at the `silu_lut.mem` file produced by Step 1; the
reference pipeline replays SiLU activation through the exact same LUT
bytes that the HDL SiLU ROM loads at elaboration time.

`generate_conv3d_golden.py` runs a pure-Python reference inference whose
arithmetic mirrors the HDL exactly, and writes into
`hardware/testbench/inference_hdl/`:

| File | Contents |
|------|----------|
| `pixels_layer0.mem` | 256 × 256 × 3 int8 input bytes (exact bytes the RTL sees after uint8→int8 conversion) |
| `golden_layer0_uram.mem` … `golden_layer16_uram.mem` | 17 per-layer URAM-format goldens — 128-bit words holding 16 int8 channels each |
| `golden_ch_out0.mem` | legacy single-channel layer-0 golden; superseded by the per-layer files above and gitignored |

The reference pipeline exactly mirrors the RTL's `u_conv3d` / `u_conv1d`:

1. Zero-pad input with `zp_in` (padding = `k // 2`)
2. MAC: `sum = Σ(input × weight)` — **no** zp_in subtraction, since the
   correction is baked into `bias_rom` upstream
3. Accumulate: `acc = sum + bias`
4. Requantize: `out = ((acc × m0 + nudge) >> nshift) + zp_out`, clamped to int8,
   where `nudge = 1 << (nshift - 1)` gives **round-half-up** matching the RTL
5. SiLU activation: `out = silu_lut[layer_idx × 256 + (out + 128)]` (skipped for
   `CONV1_LIN` layers — the cv2/cv3 detection heads)
6. 2×2 max-pool stride 2, for `CONV3_POOL` layers only

Detection head branching: layers 14–16 (cv2) read from layer 10's output, not
from layer 13. The generator restores the layer-10 activation when it reaches
layer 14.

**Critical subtlety — the round-half-up nudge.** Without the `(1 << (nshift - 1))`
nudge, the requantize path rounds toward zero and diverges from the RTL by 1 LSB
at rounding boundaries. This was one of the four bugs fixed by commit `23ed0cb`.

## Step 3 — Verify goldens against TFLite (offline gate)

```bash
python scripts/verify_goldens_vs_tflite.py
```

[`scripts/verify_goldens_vs_tflite.py`](../../scripts/verify_goldens_vs_tflite.py)
runs the TFLite interpreter layer-by-layer on `pixels_layer0.mem` and diffs
each intermediate tensor against the matching `golden_layerN_uram.mem`. It
classifies every layer as one of:

- `BIT-EXACT` — zero mismatches against the TFLite tensor
- `CLOSE (max<=1)` — every diff is ≤ 1 LSB (still treated as failure)
- `DIVERGED` — at least one diff is > 1 LSB

The script exits non-zero if **any** layer is not BIT-EXACT. It is the
**offline invariant gate**: a failure here means the Python pipeline
(weight generator, golden generator, or both) drifted from TFLite — not
that the RTL is wrong. Catch it here before spending ~30 minutes on a
bitstream rebuild.

Expected output on a healthy pipeline:

```
=== summary: 17/17 bit-exact, 0 failed ===
```

The on-board cousin of this script is
[`scripts/diag_accel_silicon_vs_tflite.py`](../../scripts/diag_accel_silicon_vs_tflite.py),
which runs the same diff against live URAM reads from the accelerator on the
Kria board.

### Known dev-host caveats

1. **TFLite interpreter version sensitivity.** `verify_goldens_vs_tflite.py`
   prefers `tflite_runtime` and falls back to `tensorflow.lite`. The two
   interpreters do not always agree bit-for-bit on every quantized op; small
   (≤ 8 LSB) divergences across interpreter versions will make the gate
   report `DIVERGED` even when the pipeline is healthy. Use the same
   interpreter (and ideally the same version) that was used when the
   goldens were last regenerated. If you only have `tflite_runtime`
   available, treat `mean|d| ≪ 1 LSB` as a healthy pattern even if the
   gate's verdict is not BIT-EXACT.

2. **Python 3.10+ required for the import chain.** The script imports
   `software.overlay.tests.checks` for the URAM unpack helper, which in
   turn pulls in `software.overlay.drivers.__init__` and the `Imx219Driver`
   class. The driver uses Python 3.10+ union syntax (`int | None`), so the
   project's `ee4218` conda env (Python 3.8) cannot import it. Run the
   verification gate from a Python 3.10+ environment that has either
   `tflite_runtime` or full `tensorflow` installed.

## Step 4 — Sync ROMs into the Vivado IP cache (easy to miss)

After Step 1, the new `.mem` / `.coe` files live in `hardware/weights/hdl/`,
but Vivado's IP packager keeps its own copy inside `hardware/ip_repo/src/`.
[`scripts/sync-ip-src.sh`](../../scripts/sync-ip-src.sh) mirrors
`hardware/rtl/*.{v,sv}` into that cache but **does not** sync `.mem` / `.coe`
files — those have to be copied by hand, or updated via the Vivado IP
packager UI.

Without this sync, the next bitstream rebuild will happily re-synthesize with
**stale** weight ROMs and every on-board diagnostic will fail in a way that
looks like an RTL bug. See `notes/deliverables/hdl-accelerator.md` for the
three-layer cache hazard that makes this bite.

Safe procedure after regenerating weights:

```bash
# 1. Sync RTL sources into the IP packager staging area
bash scripts/sync-ip-src.sh

# 2. Mirror the fresh ROM files manually (paths may need tweaking per host)
cp hardware/weights/hdl/*.{mem,coe} hardware/ip_repo/src/
cp hardware/weights/hdl/layer_config.svh hardware/ip_repo/src/

# 3. Re-package the IP in Vivado (or re-run the packaging tcl) before synth
```

## Full regeneration workflow

```bash
# 1. Regenerate weight ROMs
python hardware/scripts/generate_hdl_weights.py \
    software/models/tflite/tinyissimo_ptq_full_integer_quant.tflite \
    --out hardware/weights/hdl/

# 2. Regenerate sim goldens
python hardware/scripts/generate_conv3d_golden.py \
    --model software/models/tflite/tinyissimo_ptq_full_integer_quant.tflite \
    --image software/inference/data/input_image.jpg \
    --golden hardware/weights/hdl/weight_rom_golden.npz \
    --lut hardware/weights/hdl/silu_lut.mem \
    --out hardware/testbench/inference_hdl/ \
    --num-layers 17

# 3. Offline TFLite verification gate — fail fast if the Python pipeline drifts
python scripts/verify_goldens_vs_tflite.py

# 4. Sync into the IP packager cache (see Step 4 for manual ROM copies)
bash scripts/sync-ip-src.sh
```

## Companion scripts in this directory

- `dump_wt_mem_init.tcl` — Vivado tcl utility for dumping the weight ROM
  `INIT_*` attributes out of a routed `.dcp`, used for post-place-and-route
  forensics to confirm the bitstream actually baked the expected weights.
- `dump_all_mem_init.tcl` — broader version that dumps every `INIT_*` on every
  BRAM/URAM cell in the routed checkpoint.

These are diagnostic-only; they are not part of the regeneration pipeline.

## Related documentation

- [`notes/deliverables/hdl-accelerator.md`](../../notes/deliverables/hdl-accelerator.md) —
  design rationale, quantization pipeline math, URAM packing, SiLU LUT
  derivation, the requantize bug history, and the ROM cache hazard.
- [`software/training/`](../../software/training/) — upstream PTQ and
  ONNX → TFLite conversion (the pipeline that produces the canonical `.tflite`).
- [`hardware/weights/VERILOG_GUIDE.txt`](../../hardware/weights/VERILOG_GUIDE.txt) —
  legacy per-layer `.hex` format from the first-generation integration, kept
  for debugging traces. Not part of the production ROM pipeline.
