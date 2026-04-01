"""FastAPI camera streamer — WebSocket video + REST status."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from . import config
from .frame_source import FrameSource, create_frame_source

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

source: Optional[FrameSource] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global source
    source = create_frame_source()
    logger.info("Camera streamer started in %s mode", source.mode)
    yield
    source.close()
    logger.info("Camera streamer stopped")


app = FastAPI(title="EE4218 Camera Streamer", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def status():
    return JSONResponse(
        {
            "mode": source.mode if source else "uninitialized",
            "buffers": list(config.BUFFER_SHAPES.keys()),
            "max_fps": config.MAX_FPS,
            "jpeg_quality": config.JPEG_QUALITY,
        }
    )


@app.websocket("/ws/camera")
async def camera_ws(ws: WebSocket):
    await ws.accept()
    current_buffer = "viz"
    frame_interval = 1.0 / config.MAX_FPS
    logger.info("WebSocket client connected")

    try:
        while True:
            t0 = time.monotonic()

            # Check for control messages (non-blocking)
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                current_buffer = _handle_control(msg, current_buffer)
            except (asyncio.TimeoutError, WebSocketDisconnect):
                pass

            # Capture and send frame
            frame = source.get_frame(current_buffer)
            ok, jpeg = cv2.imencode(
                ".jpg", frame[:, :, ::-1],  # RGB -> BGR for cv2
                [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY],
            )
            if ok:
                await ws.send_bytes(jpeg.tobytes())

            # Rate limiting
            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


def _handle_control(msg: str, current_buffer: str) -> str:
    """Process a JSON control message from the client."""
    try:
        data = json.loads(msg)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON from client: %s", msg)
        return current_buffer

    action = data.get("action")
    if action == "switch_buffer":
        buf = data.get("buffer", current_buffer)
        if buf in config.BUFFER_SHAPES:
            logger.info("Switched to buffer: %s", buf)
            return buf
        logger.warning("Unknown buffer: %s", buf)
    elif action == "toggle_gamma":
        bypass = data.get("bypass", False)
        source.set_gamma_bypass(bypass)
    else:
        logger.warning("Unknown action: %s", action)

    return current_buffer
