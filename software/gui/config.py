"""Camera streamer configuration."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OVERLAY_PATH = os.environ.get(
    "EE4218_OVERLAY_PATH",
    str(PROJECT_ROOT / "hardware" / "output" / "camera_pipeline.bit"),
)

HOST = os.environ.get("EE4218_GUI_HOST", "0.0.0.0")
PORT = int(os.environ.get("EE4218_GUI_PORT", "8000"))
JPEG_QUALITY = int(os.environ.get("EE4218_JPEG_QUALITY", "80"))
MAX_FPS = int(os.environ.get("EE4218_MAX_FPS", "30"))

# Buffer resolutions (height, width, channels)
BUFFER_SHAPES = {
    "viz": (720, 1280, 3),
    "inference": (256, 256, 3),
}
