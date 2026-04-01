import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import json
import argparse
import os

def compute_iou(boxA, boxB):
    # box = [x1, y1, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    # IoU = Area of Overlap / Area of Union
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def nms(boxes, scores, iou_threshold=0.45):
    if not boxes:
        return []

    # Sort indices by score in descending order
    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        
        # Compare current box i with the rest
        remaining_idxs = idxs[1:]
        ious = [compute_iou(boxes[i], boxes[j]) for j in remaining_idxs]
        
        # Keep only indices where IoU is below the threshold
        filtered_idxs = []
        for idx, iou in zip(remaining_idxs, ious):
            if iou <= iou_threshold:
                filtered_idxs.append(idx)
        
        idxs = np.array(filtered_idxs)
        
    return keep

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

def main():
    parser = argparse.ArgumentParser()
    _here = os.path.dirname(os.path.abspath(__file__))
    _default_model = os.path.join(_here, "..", "models", "tflite", "tinyissimo_ptq_full_integer_quant.tflite")
    _default_image = os.path.join(_here, "data", "input_image.jpg")
    parser.add_argument("--model-path", type=str, default=_default_model,
                        help="Path to TFLite model")
    parser.add_argument("--input-image", type=str, default=_default_image,
                        help="Path to input image")
    parser.add_argument("--result-dir", type=str, default="results",
                        help="Path to output results")
    args = parser.parse_args()
    
    # define model type
    is_full_int8 =True

    # Setup
    interpreter = tflite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]['index']
    out_details = interpreter.get_output_details()

    # Load/Preprocess
    img = Image.open(args.input_image).convert('RGB').resize((256, 256))
    input_data = np.expand_dims(np.array(img), axis=0)
    if interpreter.get_input_details()[0]['dtype'] != np.uint8:
        is_full_int8 = False
        input_data = input_data.astype(np.float32) / 255.0

    # Inference
    interpreter.set_tensor(in_idx, input_data)
    interpreter.invoke()

    # 3. Post-Process (Handling Quantization Parameters if Full INT8)
    output_det = out_details[0]
    raw_output = interpreter.get_tensor(output_det['index'])
    scale, zp = output_det['quantization'] if is_full_int8 else (1.0, 0)
    boxes, scores, class_ids = post_process(raw_output, is_full_int8, scale, zp)

    indices = nms(boxes, scores, iou_threshold=0.45)

    # Save all tensors and metadata
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)
    
    # Convert to numpy arrays for clean saving
    np.save(os.path.join(args.result_dir, "boxes.npy"), np.array(boxes))
    np.save(os.path.join(args.result_dir, "scores.npy"), np.array(scores))
    np.save(os.path.join(args.result_dir, "class_ids.npy"), np.array(class_ids))
    np.save(os.path.join(args.result_dir, "indices.npy"), np.array(indices))

    # Optional: Still save raw outputs for debugging if needed
    for i, det in enumerate(out_details):
        np.save(os.path.join(args.result_dir, f"raw_out_{i}.npy"), interpreter.get_tensor(det['index']))
    
    meta = {
        "is_full_int8": (interpreter.get_input_details()[0]['dtype'] == np.uint8),
        "scales": [float(d['quantization'][0]) for d in out_details],
        "zps": [int(d['quantization'][1]) for d in out_details]
    }
    with open(args.result_dir +"/"+ "meta.json", "w") as f:
        json.dump(meta, f)
    print(f"Saved results to {args.result_dir}")

if __name__ == "__main__":
    main()