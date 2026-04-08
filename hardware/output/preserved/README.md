# Preserved bitstreams

This directory holds known-working bitstreams and their build artifacts for
forensic/restoration purposes. Each subdirectory corresponds to a specific
build, named with the first 8 hex digits of the bitstream md5:

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
DIR=hardware/output/preserved/playground_FIXED_5e86ce6c
cp $DIR/playground_FIXED_5e86ce6c.bit  hardware/output/playground.bit
cp $DIR/playground_FIXED_5e86ce6c.xsa  hardware/output/playground.xsa
cp $DIR/playground_FIXED_5e86ce6c.hwh  hardware/output/playground.hwh
cp $DIR/playground_FIXED_5e86ce6c.dtbo hardware/output/playground.dtbo
cp $DIR/playground_FIXED_5e86ce6c.dts  hardware/output/playground.dts
bash scripts/deploy-overlay.sh --xsa hardware/output/playground.xsa
```

## Naming convention

`playground_<STATE>_<md5prefix>` where `<STATE>` is a human-readable tag:
- `FIXED` — a build where a known bug has been fixed
- `BASELINE` — reference build before a new feature
- `REGRESSION` — a build reproducing a regression for debugging

## Current contents

| Directory | md5 prefix | Git commit | Date | Summary |
|-----------|------------|------------|------|---------|
| `playground_FIXED_5e86ce6c/` | `5e86ce6c` | `9040d73` | 2026-04-08 | First build that fixes the silicon-only +1 URAM shift bug. All 17 layers bit-exact vs sim. |
