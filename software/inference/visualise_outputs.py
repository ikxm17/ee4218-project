import numpy as np
import cv2
import os

# --- Configuration ---
CLASS_NAMES = {0: "chair", 1: "bowl", 2: "cup"}
# BGR Format for OpenCV: (Blue, Green, Red)
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)} 
RESULT_DIR = "results"
IMAGE_PATH = "data/input_image.jpg" # Ensure this matches your Kria input

def laptop_visualize():
    # 1. Load the results saved from Kria
    try:
        boxes = np.load(os.path.join(RESULT_DIR, "boxes.npy"))
        scores = np.load(os.path.join(RESULT_DIR, "scores.npy"))
        class_ids = np.load(os.path.join(RESULT_DIR, "class_ids.npy"))
        indices = np.load(os.path.join(RESULT_DIR, "indices.npy"))
    except FileNotFoundError as e:
        print(f"Error: Could not find result files in {RESULT_DIR}. {e}")
        return

    # 2. Load and Prepare Image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Could not load image at {IMAGE_PATH}")
        # Fallback: create a black canvas if image is missing
        img = np.zeros((256, 256, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (256, 256))

    print(f"Found {len(indices)} objects after NMS.")

    # 3. Drawing Loop
    for i in indices:
        # Indices in indices.npy refer to the original rows in boxes/scores/class_ids
        x, y, w, h = boxes[i]
        cls_id = int(class_ids[i])
        conf = scores[i]

        # Get class specific info
        label_text = CLASS_NAMES.get(cls_id, f"Unknown({cls_id})")
        color = CLASS_COLORS.get(cls_id, (255, 255, 255)) # Default white
        
        # Draw Box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Create Label with Background for Readability
        label = f"{label_text}: {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        
        # Draw label background rectangle
        cv2.rectangle(img, (x, y - label_h - baseline), (x + label_w, y), color, -1)
        # Draw text in white (or black depending on preference)
        cv2.putText(img, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 4. Show and Save
    cv2.imshow("Kria Inference Results", img)
    cv2.imwrite("detection_output.jpg", img)
    print("Saved visualization to detection_output.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    laptop_visualize()