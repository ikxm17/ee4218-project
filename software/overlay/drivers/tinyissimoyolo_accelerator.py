"""PYNQ driver for the TinyissimoYOLO hardware accelerator.

Supports both HDL and HLS inference engines (selected via MODE register).
Handles pixel write (via AXI-Lite FIFO), inference control, result readout,
and post-processing (DFL decode, sigmoid, NMS).

Usage:
    from pynq import MMIO
    accel = TinyissimoYoloAccelerator(MMIO(0xA00C_0000, 0x2000))

    # Load and run
    image = cv2.imread("test.jpg")
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = accel.run(image)
"""

from __future__ import annotations

import time

import numpy as np

# Detection class names and colors (BGR)
CLASS_NAMES = {0: "chair", 1: "bowl", 2: "cup"}
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}

# Output quantization (layers 13 and 16 share the same params)
OUTPUT_SCALE = 0.10655
OUTPUT_ZP = 10


class TinyissimoYoloAccelerator:
    """Driver for the tinyissimoyolo_accelerator_v1_0 IP."""

    # Register offsets
    _CTRL       = 0x000
    _STATUS     = 0x004
    _MODE       = 0x008
    _CYCLE_CNT  = 0x00C
    _LAYER_IDX  = 0x010
    _PIXEL_FIFO = 0x020
    _PIXEL_CNT  = 0x024
    _RESULT_BASE = 0x100

    # CTRL bits
    _CTRL_START     = 1 << 0
    _CTRL_FIFO_RST  = 1 << 1
    _CTRL_SOFT_RST  = 1 << 7

    # STATUS bits
    _STATUS_BUSY    = 1 << 0
    _STATUS_DONE    = 1 << 1
    _STATUS_IDLE    = 1 << 2
    _STATUS_PRELOAD = 1 << 3

    # Result layout: 320 URAM words × 128-bit = 1280 × 32-bit
    _CV2_WORDS = 256   # fmap_b[256..511]: 4 groups × 64 spatial
    _CV3_WORDS = 64    # fmap_b[512..575]: 1 group × 64 spatial
    _RESULT_WORDS = _CV2_WORDS + _CV3_WORDS  # 320 URAM words = 1280 32-bit reads

    def __init__(self, mmio):
        self._mmio = mmio

    @property
    def status(self) -> int:
        return self._mmio.read(self._STATUS)

    @property
    def cycle_count(self) -> int:
        return self._mmio.read(self._CYCLE_CNT)

    @property
    def layer_idx(self) -> int:
        return self._mmio.read(self._LAYER_IDX) & 0x1F

    @property
    def pixel_count(self) -> int:
        return self._mmio.read(self._PIXEL_CNT)

    def set_mode(self, mode: int):
        """Set input mode: 0 = AXI-Lite FIFO, 1 = S_AXIS camera."""
        self._mmio.write(self._MODE, mode & 0x1F)

    def soft_reset(self):
        self._mmio.write(self._CTRL, self._CTRL_SOFT_RST)

    def start(self):
        """Enter PRELOAD phase and reset pixel FIFO.

        In FIFO mode: call this BEFORE write_pixels().
        In S_AXIS mode: not needed (auto-preloads on SOF).
        """
        self._mmio.write(self._CTRL, self._CTRL_START | self._CTRL_FIFO_RST)

    def write_pixels(self, image_rgb: np.ndarray):
        """Write a 256x256 RGB8 image to the accelerator via FIFO.

        Converts uint8 (0-255) to int8 (-128..127) per model convention.
        Packs as 4 bytes per pixel: [R, G, B, pad].

        Args:
            image_rgb: (256, 256, 3) uint8 array, RGB channel order.
        """
        assert image_rgb.shape == (256, 256, 3), f"Expected (256,256,3), got {image_rgb.shape}"
        assert image_rgb.dtype == np.uint8, f"Expected uint8, got {image_rgb.dtype}"

        # uint8 → int8: subtract 128
        pixels_int8 = (image_rgb.astype(np.int16) - 128).astype(np.int8)

        # Pack as 4 bytes per pixel: [R, G, B, pad=0]
        packed = np.zeros((256 * 256, 4), dtype=np.int8)
        packed[:, :3] = pixels_int8.reshape(-1, 3)

        # Write to FIFO as 32-bit words (65536 writes)
        words = packed.view(np.uint32).flatten()
        for w in words:
            self._mmio.write(self._PIXEL_FIFO, int(w))

    def wait_done(self, timeout_s: float = 1.0) -> bool:
        """Poll STATUS until inference completes."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.status & self._STATUS_DONE:
                return True
        return False

    def read_results_raw(self) -> np.ndarray:
        """Read raw int8 detection results from URAM.

        Returns:
            (320, 16) int8 array — 320 URAM words, 16 bytes each.
            Rows 0-255: cv2 bbox logits (4 ch-groups × 64 spatial positions)
            Rows 256-319: cv3 class logits (1 ch-group × 64 spatial, only bytes 0-2 valid)
        """
        raw = np.zeros(self._RESULT_WORDS * 4, dtype=np.uint32)
        for i in range(self._RESULT_WORDS * 4):
            raw[i] = self._mmio.read(self._RESULT_BASE + i * 4)
        return raw.view(np.int8).reshape(self._RESULT_WORDS, 16)

    def unpack_detections(self, raw: np.ndarray) -> np.ndarray:
        """Unpack URAM-format results into (8, 8, 67) detection tensor.

        The URAM stores data in channel-group-major order:
          Group g, spatial position (y, x): raw[g*64 + y*8 + x, 0:16]
          Each row = 16 int8 channels for one spatial position.

        cv2: 4 groups × 64 spatial × 16 ch = 64 box channels
        cv3: 1 group × 64 spatial × 3 valid ch = 3 class channels

        Args:
            raw: (320, 16) int8 array from read_results_raw()

        Returns:
            (8, 8, 67) float32 array, dequantized.
        """
        # cv2: reshape (256, 16) → (4 groups, 8, 8, 16) → transpose → (8, 8, 64)
        cv2 = raw[:256].reshape(4, 8, 8, 16).transpose(1, 2, 0, 3).reshape(8, 8, 64)

        # cv3: reshape (64, 16) → (8, 8, 16) → keep first 3 channels
        cv3 = raw[256:320].reshape(8, 8, 16)[:, :, :3]

        # Combine and dequantize
        detection = np.concatenate(
            [cv2.astype(np.float32), cv3.astype(np.float32)], axis=-1
        )
        return (detection - OUTPUT_ZP) * OUTPUT_SCALE

    def run(self, image_rgb: np.ndarray, conf_thresh: float = 0.3,
            nms_thresh: float = 0.45) -> dict:
        """End-to-end inference: write image → run → read → post-process.

        Args:
            image_rgb: (256, 256, 3) uint8 RGB image.
            conf_thresh: Confidence threshold for detections.
            nms_thresh: IoU threshold for NMS.

        Returns:
            dict with keys: boxes, scores, class_ids, class_names, cycle_count
        """
        self.set_mode(0)
        self.start()
        self.write_pixels(image_rgb)

        if not self.wait_done(timeout_s=1.0):
            raise TimeoutError("Inference did not complete within 1 second")

        cycles = self.cycle_count
        raw = self.read_results_raw()
        data = self.unpack_detections(raw)

        boxes, scores, class_ids = post_process(data, conf_thresh)
        boxes, scores, class_ids = nms(boxes, scores, class_ids, nms_thresh)

        return {
            "boxes": boxes,
            "scores": scores,
            "class_ids": class_ids,
            "class_names": [CLASS_NAMES.get(c, f"cls{c}") for c in class_ids],
            "cycle_count": cycles,
        }


def decode_dfl(dfl_data: np.ndarray) -> np.ndarray:
    """Softmax + weighted sum for DFL 16-bin distribution."""
    exp_data = np.exp(dfl_data - np.max(dfl_data, axis=-1, keepdims=True))
    prob = exp_data / np.sum(exp_data, axis=-1, keepdims=True)
    return np.sum(prob * np.arange(16), axis=-1)


def post_process(data: np.ndarray, conf_thresh: float = 0.3,
                 input_size: int = 256) -> tuple:
    """Decode (8, 8, 67) detection tensor into bounding boxes.

    Args:
        data: (8, 8, 67) float32, already dequantized.
        conf_thresh: Minimum confidence to keep a detection.
        input_size: Image dimension (256).

    Returns:
        (boxes, scores, class_ids) — lists of detections.
    """
    grid_size = 8
    stride = input_size / grid_size
    boxes, scores, class_ids = [], [], []

    for y in range(grid_size):
        for x in range(grid_size):
            cell = data[y, x, :]

            # Sigmoid on class logits (last 3 elements)
            conf_scores = 1.0 / (1.0 + np.exp(-cell[64:]))
            cls_id = int(np.argmax(conf_scores))
            max_conf = float(conf_scores[cls_id])

            if max_conf > conf_thresh:
                # DFL decode: 64 logits → 4 LTRB distances
                ltrb = decode_dfl(cell[:64].reshape(4, 16))

                # Grid-to-pixel projection
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                x1 = cx - ltrb[0] * stride
                y1 = cy - ltrb[1] * stride
                x2 = cx + ltrb[2] * stride
                y2 = cy + ltrb[3] * stride

                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                scores.append(max_conf)
                class_ids.append(cls_id)

    return boxes, scores, class_ids


def nms(boxes: list, scores: list, class_ids: list,
        iou_thresh: float = 0.45) -> tuple:
    """Non-maximum suppression using OpenCV."""
    if not boxes:
        return [], [], []

    try:
        import cv2
        indices = cv2.dnn.NMSBoxes(boxes, scores, min(scores), iou_thresh)
        if len(indices) > 0:
            idx = indices.flatten()
            return (
                [boxes[i] for i in idx],
                [scores[i] for i in idx],
                [class_ids[i] for i in idx],
            )
    except ImportError:
        pass  # cv2 not available, skip NMS

    return boxes, scores, class_ids
