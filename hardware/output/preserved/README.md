# Preserved bitstreams

This directory holds known-working bitstreams and their build artifacts for
forensic / restoration purposes.

**Scope: accelerator integrations only.** Every build in this directory is a
PL-side-only bitstream that integrates the TinyissimoYOLO accelerator
(`tinyissimoyolo_accelerator_0`) on top of the bare Zynq PS + AXI-Lite
plumbing. `FIRST` and `SECOND` carry only the hand-written HDL engine;
`FINAL` additionally carries the Vitis HLS C++ engine alongside it, with a
runtime `engine_sel` mux. None of them include the camera pipeline (CSI-2 RX
/ demosaic / gamma LUT / scaler) — that lives in a separate
`camera_pipeline.bit`. The two cannot currently run simultaneously on the
same FPGA.

Each subdirectory corresponds to a specific build, named with a state tag
and the first 8 hex digits of the bitstream md5:

```
preserved/
└── playground_<STATE>_<md5prefix>/
    ├── README.md                 # build status: what's integrated, what works, caveats
    ├── *.bit                     # the bitstream itself
    ├── *.xsa                     # hardware platform export (for PYNQ Overlay)
    ├── *.hwh                     # hardware handoff (PYNQ IP dict)
    ├── *.dtbo / *.dts            # device tree overlay + source
    ├── *_routed.dcp              # routed checkpoint (for Tcl forensics)
    ├── *_timing_summary_routed.rpt
    └── *_utilization_placed.rpt
```

## Rebuild protection

`scripts/rebuild_bitstream.tcl` only overwrites top-level `hardware/output/playground.*`
files — it never touches this `preserved/` subdirectory. Files here are safe
from accidental clobber.

## Restoring a preserved build

Pick a subdirectory, then:

```bash
DIR=hardware/output/preserved/playground_SECOND_3fcbe84a
cp $DIR/playground_SECOND_3fcbe84a.bit  hardware/output/playground.bit
cp $DIR/playground_SECOND_3fcbe84a.xsa  hardware/output/playground.xsa
cp $DIR/playground_SECOND_3fcbe84a.hwh  hardware/output/playground.hwh
cp $DIR/playground_SECOND_3fcbe84a.dtbo hardware/output/playground.dtbo
cp $DIR/playground_SECOND_3fcbe84a.dts  hardware/output/playground.dts
bash scripts/deploy-overlay.sh --xsa hardware/output/playground.xsa
```

## Naming convention

`playground_<STATE>_<md5prefix>` where `<STATE>` tracks the HDL-integration
progression — an ordinal tag marking each known-working milestone in the
maturation of the HDL accelerator bitstream:

- `FIRST` — first known-working HDL-accelerator build. Typically the initial
  fix of a silicon-correctness bug; verified against its own sim golden but
  not necessarily against the TFLite reference.
- `SECOND` — first HDL-accelerator build that is additionally TFLite-bit-exact
  (end-to-end numerics loop closed: TFLite → sim golden → RTL silicon).
- `FINAL` — integrated HDL + HLS accelerator build. First bitstream to carry
  both engines alongside each other with a runtime `engine_sel` mux in
  `inference_top.sv`, exposed via the driver `set_engine` API. Intended as
  the frozen bitstream for the project deliverable.

  **Naming deviation:** unlike `FIRST` / `SECOND`, the `FINAL` directory does
  not include the md5 prefix suffix, and the files inside use the descriptive
  `playground_hdl_hls_wrapper.*` basename rather than matching the directory
  name. The routed `.dcp`, timing, and utilization reports are also not
  checked in for this build — only the `.bit` / `.hwh` / `.xsa` / `.dts` /
  `.dtbo` needed to deploy.

## Current contents

| Directory | md5 prefix | Git commit | Date | Status | Summary |
|-----------|------------|------------|------|--------|---------|
| `playground_FIRST_5e86ce6c/` | `5e86ce6c` | `9040d73` | 2026-04-08 | Sim-bit-exact; **not** TFLite-faithful (pre-requantize fix) | First build that fixes the silicon-only +1 URAM shift bug. All 17 layers bit-exact vs sim. RTL and sim golden both drift from TFLite by ±1 LSB at rounding boundaries. |
| `playground_SECOND_3fcbe84a/` | `3fcbe84a` | `0dad181` | 2026-04-08 | Sim-bit-exact **and** TFLite-bit-exact | First TFLite-faithful HDL-accelerator build. Weight ROMs regenerated after round-half-up requantize fix (`275dfe5`, `23ed0cb`, `c411005`). RTL unchanged from FIRST, so +1-shift fix still holds. Hold margin drifted to +0.011 ns (from +0.014 ns in FIRST). |
| `playground_FINAL/` (file basename: `playground_hdl_hls_wrapper`) | `231033d7` | `bad2ba1` | 2026-04-09 | Integrated HDL + HLS accelerator | **Most recent build.** First bitstream to carry both the hand-written HDL engine and the Vitis HLS C++ engine simultaneously, with runtime backend selection via `engine_sel` / `set_engine` in `inference_top.sv` + `axil_regs.sv`. PL-side-only, no camera pipeline. No routed `.dcp` or timing report checked in — only `.bit` / `.hwh` / `.xsa` / `.dts` / `.dtbo`. |
