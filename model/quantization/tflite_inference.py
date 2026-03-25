import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import os

# --- Configuration ---
CLASS_NAMES = {0: "chair", 1: "bowl", 2: "cup"}
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}

# CHOOSE YOUR MODEL HERE
MODEL_PATH = "tl/tinyissimo_ptq_full_integer_quant.tflite"     # FULL INT8 - w forced quantization of DFL
# MODEL_PATH = "tl/tinyissimo_ptq_integer_quant.tflite"           
IMAGE_PATH = "/home/leeey/Downloads/ee4218-project/model/quantization/coco3/images/val2017/000000551660.jpg"

def decode_dfl(dfl_data):
    """Softmax + Weighted Sum for DFL 16-bin distribution."""
    # Subtract max for numerical stability (important for INT8-forced models)
    exp_data = np.exp(dfl_data - np.max(dfl_data, axis=-1, keepdims=True))
    prob = exp_data / np.sum(exp_data, axis=-1, keepdims=True)
    return np.sum(prob * np.arange(16), axis=-1)

def post_process(raw_data, is_full_int8, scale=1.0, zp=0, input_size=256):
    grid_size = 8
    stride = input_size / grid_size
    boxes, scores, class_ids = [], [], []

    # 1. Manual Dequantization for Full INT8
    # The 'int8' model (Float I/O) is already float32 here, so scale=1.0/zp=0
    data = (raw_data[0].astype(np.float32) - zp) * scale

    for y in range(grid_size):
        for x in range(grid_size):
            cell = data[y, x, :]
            
            # 2. Class Confidence (Sigmoid on logits)
            conf_scores = 1 / (1 + np.exp(-cell[64:]))
            cls_id = np.argmax(conf_scores)
            max_conf = conf_scores[cls_id]
            
            if max_conf > 0.3:
                # 3. Box Geometry (LTRB)
                ltrb = decode_dfl(cell[:64].reshape(4, 16))
                
                # Grid-to-Pixel Projection
                cx, cy = (x + 0.5) * stride, (y + 0.5) * stride
                x1, y1 = cx - ltrb[0] * stride, cy - ltrb[1] * stride
                x2, y2 = cx + ltrb[2] * stride, cy + ltrb[3] * stride
                
                boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
                scores.append(float(max_conf))
                class_ids.append(cls_id)
                
    return boxes, scores, class_ids

# --- Main Logic ---
print(f"Loading Model: {MODEL_PATH}")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_det = interpreter.get_input_details()[0]
output_det = interpreter.get_output_details()[0]

# Detect Model Type
is_full_int8 = (input_det['dtype'] == np.uint8)
print(f"Mode: {'FULL INT8 (Uint8 Input)' if is_full_int8 else 'INT8 (Float32 Input)'}")

# 1. Preprocess
original_img = cv2.imread(IMAGE_PATH)
input_img = cv2.resize(original_img, (256, 256))
rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

if is_full_int8:
    # Full INT8 expects raw 0-255 bytes
    input_data = np.expand_dims(rgb_img, axis=0).astype(np.uint8)
else:
    # INT8 (Float I/O) expects normalized 0.0-1.0
    input_data = np.expand_dims(rgb_img, axis=0).astype(np.float32) / 255.0

# 2. Inference
interpreter.set_tensor(input_det['index'], input_data)
interpreter.invoke()
raw_output = interpreter.get_tensor(output_det['index'])

# 3. Post-Process (Handling Quantization Parameters if Full INT8)
scale, zp = output_det['quantization'] if is_full_int8 else (1.0, 0)
boxes, scores, class_ids = post_process(raw_output, is_full_int8, scale, zp)

# 4. NMS and Draw
display_img = cv2.resize(original_img, (512, 512))
scale_f = 512 / 256
indices = cv2.dnn.NMSBoxes(boxes, scores, 0.3, 0.45)

if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = [int(v * scale_f) for v in boxes[i]]
        cls_id = class_ids[i]
        color = CLASS_COLORS[cls_id]
        label = f"{CLASS_NAMES[cls_id]} {scores[i]:.2f}"
        
        cv2.rectangle(display_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(display_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imshow("Tinyissimo Detection", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()