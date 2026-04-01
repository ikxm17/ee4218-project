#!/usr/bin/env bash
# Install application Python packages into the shared venv
#
# Adds packages required by software/ that aren't part of a dedicated
# framework script (03-pynq, 04-tflite, 05-onnxruntime).
set -euo pipefail

VENV_DIR="/opt/ee4218/ee4218-venv"

echo "=== Application packages ==="

if [ ! -x "$VENV_DIR/bin/python3" ]; then
    echo "ERROR: venv not found at $VENV_DIR — run 03-pynq.sh first."
    exit 1
fi

echo "Installing application packages..."
# Pin opencv and fastapi to avoid breaking pynq transitive deps:
#   pynqutils requires numpy<2.0 — opencv>=4.12 needs numpy>=2
#   pynqmetadata requires pydantic==1.9.1 — fastapi>=0.126 needs pydantic>=2.7
"$VENV_DIR/bin/pip" install \
    "opencv-python-headless<=4.11.0.86" \
    "fastapi>=0.100,<=0.125.0" \
    "uvicorn[standard]"

# ── Sanity checks ───────────────────────────────────────────────────
echo "Running import checks..."
"$VENV_DIR/bin/python3" -c "import cv2; print(f'  opencv {cv2.__version__} OK')"
"$VENV_DIR/bin/python3" -c "import fastapi; print(f'  fastapi {fastapi.__version__} OK')"
"$VENV_DIR/bin/python3" -c "import uvicorn; print(f'  uvicorn {uvicorn.__version__} OK')"

echo "Application packages setup complete."
