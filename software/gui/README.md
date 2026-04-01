# Camera Streamer

FastAPI WebSocket server that streams camera frames to a browser.

## Dependencies

```bash
source /opt/ee4218/ee4218-venv/bin/activate
pip install fastapi "uvicorn[standard]" opencv-python-headless
```

## Usage

```bash
# On board (hardware or test mode — auto-detects overlay)
sudo /opt/ee4218/ee4218-venv/bin/python3 -m uvicorn software.gui.app:app --host 0.0.0.0 --port 8000

# On host (test mode only — no PYNQ)
python -m uvicorn software.gui.app:app --host 127.0.0.1 --port 8000
```

Open `http://<board-ip>:8000` in a browser.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EE4218_OVERLAY_PATH` | `hardware/output/camera_pipeline.bit` | Bitstream path |
| `EE4218_GUI_HOST` | `0.0.0.0` | Bind address |
| `EE4218_GUI_PORT` | `8000` | Bind port |
| `EE4218_JPEG_QUALITY` | `80` | JPEG compression (1-100) |
| `EE4218_MAX_FPS` | `30` | Frame rate cap |

## WebSocket Protocol

**Endpoint:** `ws://<host>:8000/ws/camera`

- Server sends binary JPEG frames
- Client sends JSON control messages:
  - `{"action": "switch_buffer", "buffer": "viz"}` — 720p visualization
  - `{"action": "switch_buffer", "buffer": "inference"}` — 256x256 inference input
  - `{"action": "toggle_gamma", "bypass": true}` — toggle Gamma LUT bypass
