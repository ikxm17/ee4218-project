#!/usr/bin/env bash
# Install TFLite Runtime for software inference on A53
#
# Adds tflite-runtime to the shared venv created by 03-pynq.sh.
# Includes a smoke test using a pre-built MobileNet model to verify
# that inference works end-to-end (load → allocate → invoke → output).
set -euo pipefail

VENV_DIR="/opt/ee4218/ee4218-venv"
SMOKE_URL="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz"
SMOKE_MODEL="/tmp/mobilenet_v1_1.0_224_quant.tflite"

echo "=== TFLite Runtime setup ==="

# ── Idempotent check ─────────────────────────────────────────────────
if "$VENV_DIR/bin/python3" -c \
    "import tflite_runtime; assert tflite_runtime.__version__ == '2.14.0'" 2>/dev/null; then
    echo "tflite-runtime 2.14.0 already installed, skipping."
else
    echo "Installing tflite-runtime..."
    "$VENV_DIR/bin/pip" install tflite-runtime==2.14.0
fi

# Quick import sanity check
"$VENV_DIR/bin/python3" -c \
    "import tflite_runtime.interpreter as tflite; print('tflite-runtime', tflite_runtime.__version__, 'OK')"

# ── Inference smoke test ─────────────────────────────────────────────
# Download a pre-built quantized MobileNet V1, run inference with
# random input, and verify we get output with the expected shape.
echo "Running inference smoke test..."
wget -q "$SMOKE_URL" -O /tmp/mobilenet_quant.tgz
tar xzf /tmp/mobilenet_quant.tgz -C /tmp mobilenet_v1_1.0_224_quant.tflite
"$VENV_DIR/bin/python3" - "$SMOKE_MODEL" <<'PYEOF'
import sys, numpy as np
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path=sys.argv[1], num_threads=4)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

# Random input matching expected shape and dtype
test_input = np.random.randint(0, 256, size=inp['shape'], dtype=inp['dtype'])
interpreter.set_tensor(inp['index'], test_input)
interpreter.invoke()

result = interpreter.get_tensor(out['index'])
assert result.shape == tuple(out['shape']), f"Output shape mismatch: {result.shape}"
print(f"Inference OK — input {tuple(inp['shape'])} {inp['dtype'].__name__} → output {tuple(out['shape'])}")
PYEOF

rm -f "$SMOKE_MODEL" /tmp/mobilenet_quant.tgz
echo "TFLite Runtime setup complete."
