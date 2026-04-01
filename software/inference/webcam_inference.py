"""
webcam_inference.py — TinyissimoYOLO ONNX webcam demo
------------------------------------------------------
Requirements:
    pip install onnxruntime numpy opencv-python

Usage:
    python webcam_inference.py --model best.onnx --conf 0.01
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES  = ["chair", "bowl", "cup"]
CLASS_COLORS = [
    (0,   200, 255),
    (0,   255, 128),
    (255, 100,  80),
]

# ── Pre/post-processing ───────────────────────────────────────────────────────

def letterbox(img, target):
    h, w    = img.shape[:2]
    scale   = target / max(h, w)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((target, target, 3), 114, dtype=np.uint8)
    pad_x   = (target - new_w) // 2
    pad_y   = (target - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def preprocess(frame, img_size):
    img, scale, pad = letterbox(frame, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, scale, pad


def xywh2xyxy(boxes):
    out       = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms(boxes, scores, iou_thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    kept   = []
    while order.size > 0:
        i = order[0]
        kept.append(i)
        xx1   = np.maximum(x1[i], x1[order[1:]])
        yy1   = np.maximum(y1[i], y1[order[1:]])
        xx2   = np.minimum(x2[i], x2[order[1:]])
        yy2   = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return kept


def postprocess(output, scale, pad, orig_shape, conf_thr, iou_thr):
    pred         = output[0].T              # [anchors, 4+nc]
    boxes_xywh   = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids    = np.argmax(class_scores, axis=1)
    confidences  = class_scores[np.arange(len(class_scores)), class_ids]

    mask = confidences >= conf_thr
    if not mask.any():
        return []

    boxes_xywh  = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]
    boxes_xyxy  = xywh2xyxy(boxes_xywh)

    orig_h, orig_w = orig_shape
    pad_x, pad_y   = pad
    results = []

    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        kept     = nms(boxes_xyxy[cls_mask], confidences[cls_mask], iou_thr)
        for i in kept:
            box  = boxes_xyxy[cls_mask][i]
            conf = confidences[cls_mask][i]
            x1   = int(max(0, min((box[0] - pad_x) / scale, orig_w - 1)))
            y1   = int(max(0, min((box[1] - pad_y) / scale, orig_h - 1)))
            x2   = int(max(0, min((box[2] - pad_x) / scale, orig_w - 1)))
            y2   = int(max(0, min((box[3] - pad_y) / scale, orig_h - 1)))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append({
                "box":        (x1, y1, x2, y2),
                "conf":       float(conf),
                "class_id":   int(cls),
                "class_name": CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else str(cls),
            })
    return results


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color  = CLASS_COLORS[det["class_id"] % len(CLASS_COLORS)]
        label  = f"{det['class_name']} {det['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(y1, th + 6)
        cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 4, ly + bl - 4), color, -1)
        cv2.putText(frame, label, (x1 + 2, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, fps, n_det, conf_thr, model_name):
    cv2.rectangle(frame, (0, 0), (340, 60), (20, 20, 20), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}  Det: {n_det}  Conf: {conf_thr:.2f}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, Path(model_name).name,
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return frame


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str,   default="best.onnx")
    parser.add_argument("--img-size", type=int,   default=256)
    parser.add_argument("--conf",     type=float, default=0.3)
    parser.add_argument("--iou",      type=float, default=0.45)
    parser.add_argument("--camera",   type=int,   default=0)
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
    session    = ort.InferenceSession(args.model, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"Input  : {input_name} {session.get_inputs()[0].shape}")
    print(f"Output : {session.get_outputs()[0].name} {session.get_outputs()[0].shape}")
    print(f"Provider: {session.get_providers()[0]}")

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    print("Press Q to quit.")

    fps_avg = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        t0 = time.perf_counter()

        # Save original frame before any modification
        orig_frame = frame.copy()

        # ── Preprocess: letterbox + normalise ─────────────────────────────────
        # letterbox() is called here directly so we keep the BGR letterboxed
        # image (debug_img) without going through the float32 tensor conversion
        debug_img, scale, pad = letterbox(orig_frame, args.img_size)
        debug_img = debug_img.copy()   # own buffer for drawing

        # Build the model tensor from the same letterboxed image
        inp = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        inp = inp.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))
        tensor = np.expand_dims(inp, axis=0)

        outputs    = session.run(None, {input_name: tensor})
        detections = postprocess(
            outputs[0], scale, pad, frame.shape[:2],
            args.conf, args.iou,
        )

        # ── Draw on full-res frame ─────────────────────────────────────────────
        draw_detections(frame, detections)

        elapsed = time.perf_counter() - t0
        fps_avg = 0.1 * (1.0 / max(elapsed, 1e-6)) + 0.9 * fps_avg
        draw_hud(frame, fps_avg, len(detections), args.conf, args.model)

        # ── Draw on 256px model-input debug window ────────────────────────────
        # Boxes are mapped back from original-frame coords → letterboxed space
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            lx1 = int(x1 * scale + pad[0])
            ly1 = int(y1 * scale + pad[1])
            lx2 = int(x2 * scale + pad[0])
            ly2 = int(y2 * scale + pad[1])
            color = CLASS_COLORS[det["class_id"] % len(CLASS_COLORS)]
            label = f"{det['class_name']} {det['conf']:.2f}"
            cv2.rectangle(debug_img, (lx1, ly1), (lx2, ly2), color, 1)
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            ly_txt = max(ly1, th + 4)
            cv2.rectangle(debug_img, (lx1, ly_txt - th - 4),
                          (lx1 + tw + 2, ly_txt + bl - 2), color, -1)
            cv2.putText(debug_img, label, (lx1 + 1, ly_txt - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("Model input (256x256)", debug_img)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()