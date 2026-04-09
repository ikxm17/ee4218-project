# PYNQ Overlay and PL IP Drivers

Python-side glue for running the Kria KV260 design on-board: PYNQ overlay
loading, per-IP register drivers, frame buffer management, and the camera
pipeline orchestrator.

## Entry points

| File | Role |
|------|------|
| `camera.py`  | `CameraOverlay` — orchestrates bitstream load, IMX219 power-on, sensor init, PL IP configuration, stream start, and health checks |
| `buffers.py` | DDR frame buffer allocation and view helpers (raw / RGB / inference input) |

## PL IP drivers (`drivers/`)

Each driver targets one Xilinx/user IP block in the PL. The class-level
`IP_VLNV` / `IP_NAME` fields are used by the overlay's IP audit.

| Driver | IP | Purpose |
|--------|-----|---------|
| `csi2_rx.py`                    | `mipi_csi2_rx_subsystem:6.0` | MIPI CSI-2 receiver configuration, D-PHY lock detection, soft reset sequence |
| `demosaic.py`                   | `v_demosaic:1.1`             | Bayer → RGB demosaicing, bayer phase select |
| `gamma_lut.py`                  | `v_gamma_lut:1.1`            | 256-entry gamma LUT load + bypass |
| `vdma.py`                       | `axi_vdma:6.3`               | S2MM/MM2S DMA channel setup, frame-store management, DMASR status decoding |
| `imx219.py`                     | `axi_iic:2.1` (I2C bus)      | IMX219 sensor register table writes, mode select |
| `vpss.py`                       | `v_proc_ss:2.3`              | Multi-Scaler driver (currently unused — hardware bypass; see `_vpss_coeff.py` for Lanczos tap table) |
| `tinyissimoyolo_accelerator.py` | `user.org:user:tinyissimoyolo_accelerator` | HDL + HLS accelerator control — `set_engine`, `set_mode`, URAM preload/readback, result window decode, `nms()` post-processing |
| `_hls_common.py`                | (helper)                     | Shared HLS-backed driver logic |
| `_imx219_regs.py`               | (helper)                     | IMX219 register table constants |
| `_vpss_coeff.py`                | (helper)                     | Lanczos 6-tap / 64-phase scaler coefficients |

## Tests (`tests/`)

Split into **off-board** (pure unit tests, no PYNQ) and **on-board** (require
bitstream + live PL):

```
tests/
  test_*_offboard.py   # import-safe, run anywhere
  test_*_onboard.py    # requires sudo + XILINX_XRT + overlay
  checks.py            # shared URAM unpack / register helpers
  conftest.py          # fixtures (skip onboard when not on board)
```

Run off-board tests:

```bash
python -m pytest software/overlay/tests/ -k offboard
```

On-board test invocation pattern:

```bash
echo <passwd> | sudo -S XILINX_XRT=/usr /opt/ee4218/ee4218-venv/bin/python3 \
  -m pytest software/overlay/tests/test_csi2_rx_onboard.py
```

## Dev-host caveat

`drivers/__init__.py` imports every driver unconditionally, including
`imx219.py` which uses Python 3.10+ union syntax (`int | None`). Dev-host
scripts that import this package need Python 3.10+ — the project's `ee4218`
conda env (Python 3.8) cannot import it; use `claude-utils` or any 3.10+ env.
