#!/usr/bin/env bash
# Install ONNX Runtime for software inference on A53
#
# Adds onnxruntime to the shared venv created by 03-pynq.sh.
# Includes a smoke test using a pre-built SqueezeNet model to verify
# that inference works end-to-end (load → session → run → output).
set -euo pipefail

VENV_DIR="/opt/ee4218/ee4218-venv"
ORT_VERSION="1.23.2"
SMOKE_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-12.onnx"
SMOKE_MODEL="/tmp/squeezenet1.0-12.onnx"

echo "=== ONNX Runtime setup ==="

# ── Idempotent check ─────────────────────────────────────────────────
if "$VENV_DIR/bin/python3" -c \
    "import onnxruntime; assert onnxruntime.__version__ == '$ORT_VERSION'" 2>/dev/null; then
    echo "onnxruntime $ORT_VERSION already installed, skipping."
else
    echo "Installing onnxruntime..."
    "$VENV_DIR/bin/pip" install onnxruntime=="$ORT_VERSION"
fi

# Quick import sanity check
"$VENV_DIR/bin/python3" -c \
    "import onnxruntime as ort; print('onnxruntime', ort.__version__, 'providers:', ort.get_available_providers())"

# ── Inference smoke test ─────────────────────────────────────────────
# Download a pre-built SqueezeNet 1.0, run inference with random input,
# and verify we get output with the expected shape.
echo "Running inference smoke test..."
wget -q "$SMOKE_URL" -O "$SMOKE_MODEL"
"$VENV_DIR/bin/python3" - "$SMOKE_MODEL" <<'PYEOF'
import sys, numpy as np
import onnxruntime as ort

session = ort.InferenceSession(sys.argv[1])
inp = session.get_inputs()[0]
out = session.get_outputs()[0]

# Replace dynamic dims (None or symbolic strings) with 1
shape = [d if isinstance(d, int) else 1 for d in inp.shape]
expected_out = tuple(d if isinstance(d, int) else 1 for d in out.shape)

# Random input matching expected shape
test_input = np.random.randn(*shape).astype(np.float32)
result = session.run([out.name], {inp.name: test_input})[0]

assert result.shape == expected_out, f"Output shape mismatch: {result.shape} vs {expected_out}"
print(f"Inference OK — input {tuple(shape)} float32 → output {result.shape}")
PYEOF

rm -f "$SMOKE_MODEL"
echo "ONNX Runtime setup complete."
