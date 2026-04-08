# playground_FIXED_5e86ce6c — known-working TinyissimoYOLO HDL accelerator build

## Identity

| Field | Value |
|-------|-------|
| Bitstream md5 | `5e86ce6c86b6ad3c0f9da3ed37025770` |
| Git commit | `9040d73` (branch `feat/hdl-pipeline`, tag `fix/+1-shift-heisenbug`) |
| Build date | 2026-04-08 |
| Vivado version | 2025.2 (Build 6299465) |
| Target device | xck26-sfvc784 (Kria KV260, -2LV speed grade) |

## What's integrated

- **TinyissimoYOLO HDL accelerator** (`tinyissimoyolo_accelerator_0` IP)
  - 17-layer int8 quantized YOLO inference engine
  - Layer breakdown: 10 CONV3/CONV3_POOL backbone + conv1x1 heads for cv2 (box) and cv3 (class)
  - Parallelism: 16 slots (`MAX_PARALLEL`), each with 9-MAC conv chain + 2x circular_buffer row delay
  - Storage: 2x URAM-based fmap ping-pong buffers (`fmap_a`, `fmap_b`, 16384x128-bit each), single BRAM accumulator scratch, BRAM weight/QP/SiLU ROMs
  - Control: AXI-Lite slave on `S_AXI_LITE` with register map at 0x000-0x05C (status/ctrl) and 0x100-0x14FF (result read window)
  - Input modes: (0) AXI-Lite FIFO pixel preload, (1) S_AXIS camera stream
  - Debug captures: layer 0 address pipeline snapshots at 0x030-0x05C (conv_res, pool_out, rmw_s0, out_buf_wr, rmw_base, pp_wr_offset, ch_out, h_out)
  - Cycle count: 2,936,665 per inference at 100 MHz (~29.4 ms)

- **Zynq UltraScale+ PS integration**
  - PS8 with one 100 MHz PL clock
  - AXI-Lite GP master → SmartConnect → accelerator slave
  - No DMA, no camera hardware connected in this bitstream

- **Cycle count / utilization**
  - CLB LUTs: 13454 / 117120 (11.49%)
  - CLB Registers: 5321 / 234240 (2.27%)
  - URAM: (see full rpt)
  - BRAM: (see full rpt)

## What's NOT integrated

- **Camera pipeline** — the image sensor / CSI-2 RX / demosaic / gamma LUT / multi-scaler pipeline is in a SEPARATE bitstream (`camera_pipeline.bit`, not in this directory). The two cannot currently run simultaneously on the same FPGA.
- **VDMA / AXI-Full datapath** — preload is via AXI-Lite MMIO only, limiting pixel ingress bandwidth to ~150 ms for a 256x256x3 image
- **Real-time video inference** — the HDL accelerator in this build cannot consume camera frames directly without a hardware glue layer

## Verification status

All on-board silicon verification PASSED with this bitstream (2026-04-08):

| Diagnostic | Result |
|-----------|--------|
| `diag_accel_sim_vs_silicon.py` | **PASS** — cv2 (layer 13) + cv3 (layer 16) both bit-exact vs sim goldens, 0 diffs / 4288 values |
| `diag_accel_layer0_max1.py` | **PASS** — `H1 dominant (100.0%)`, fmap_a[0..16383] bit-exact |
| `diag_accel_per_layer_sweep.py` | **PASS** — all 17 layers BIT-EXACT, cycle count 2936665 (matches sim) |
| `diag_accel_layer_bisect.py` | **PASS** — all 16 checkable layers, 0 diffs across all regions |
| `diag_accel_determinism.py` | **DET** — 3 independent run pairs, same md5 |
| `diag_accel_shift_analysis.py` | **PASS** — layer 0 fmap_a[4096..16383] = 12288/12288 H1 match |
| `diag_accel_impulse_response.py` | **PASS** — single-pixel impulse lands at expected pool coordinates |

## Fix history (the Heisenbug story)

This build is the first known-working bitstream after the +1 URAM shift bug
("silicon[k] == golden[(k+1) mod 16384]") was discovered. The fix emerged
from adding debug capture logic to `inference_hdl.sv`, `inference_top.sv`,
and `axil_regs.sv` (commits `be176d7` → `824e0f7` → `9040d73`). No
functional RTL was modified — only debug capture flops and signal taps
were added. The specific capture added in `9040d73` (rmw_base_addr +
curr_pp_wr_offset + curr_ch_out + h_out snapshots) appears to be the
load-bearing change that tipped placement/routing into the working region.

### Suspected root cause

The FIXED build's timing report shows:
- **Setup WNS: +0.896 ns** (healthy)
- **Hold WHS: +0.014 ns** (on the razor's edge)

The tightest hold paths within the accelerator are:
1. **0.019 ns**: `u_conv3d/ACC_write_address_reg[9]` → `u_conv3d/ACC_write_address_d_reg[9]`
2. 0.019 ns: `u_inference_hdl/rmw_s0_addr_reg[7]` → `dbg_rmw_s0_addr_0_reg[7]`
3. 0.024 ns: `u_inference_hdl/u_max_pool/out_cnt_reg[10]` → `pool_addr_reg[10]`
4. 0.027-0.035 ns: various rmw_s0/max_pool/circular_buffer paths

Path #1 is the cone-A pipeline fix flop added in commit `dd77efd`. The
source and destination are in physically adjacent slices (SLICE_X51Y81 →
SLICE_X52Y81), with data delay 0.247 ns and clock skew 0.125 ns — leaving
only **19 picoseconds** of hold margin.

The broken builds likely had even less margin on this path (or a similar
one), causing a real hold violation that post-impl-timing sim did not
report because STA said "pass" with a few picoseconds of positive slack.
Adding the new debug capture flops changed Vivado's placement just enough
to gain ~5 picoseconds of slack on the critical path, eliminating the
physical hold violation.

**This fix is fragile.** A future rebuild (with any RTL change or even a
different Vivado version) could re-break the same path. The proper fix
would be to add an explicit placement constraint, a `set_max_delay` /
`set_min_delay`, or a `DONT_TOUCH` on the cone-A flops to guarantee
timing margin.

## Files in this directory

| File | Size | Purpose |
|------|------|---------|
| `playground_FIXED_5e86ce6c.bit` | 7.8 MB | Programming bitstream for PL |
| `playground_FIXED_5e86ce6c.xsa` | 2.3 MB | Hardware platform export (for PYNQ `Overlay()`) |
| `playground_FIXED_5e86ce6c.hwh` | 176 KB | Hardware handoff XML (PYNQ ip_dict) |
| `playground_FIXED_5e86ce6c.dtbo` | 196 B | Compiled device tree overlay |
| `playground_FIXED_5e86ce6c.dts` | 375 B | Device tree source |
| `playground_FIXED_5e86ce6c_routed.dcp` | 16.9 MB | Routed checkpoint (for Tcl forensics) |
| `playground_FIXED_5e86ce6c_timing_summary_routed.rpt` | 540 KB | Full timing summary (shows the 0.019 ns hold margin) |
| `playground_FIXED_5e86ce6c_utilization_placed.rpt` | 13 KB | Post-place resource utilization |

## Restoring this build

```bash
DIR=hardware/output/preserved/playground_FIXED_5e86ce6c
cp $DIR/playground_FIXED_5e86ce6c.bit  hardware/output/playground.bit
cp $DIR/playground_FIXED_5e86ce6c.xsa  hardware/output/playground.xsa
cp $DIR/playground_FIXED_5e86ce6c.hwh  hardware/output/playground.hwh
cp $DIR/playground_FIXED_5e86ce6c.dtbo hardware/output/playground.dtbo
cp $DIR/playground_FIXED_5e86ce6c.dts  hardware/output/playground.dts
bash scripts/deploy-overlay.sh --xsa hardware/output/playground.xsa
```
