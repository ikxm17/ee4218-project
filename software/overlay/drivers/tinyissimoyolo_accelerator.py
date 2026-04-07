"""PYNQ driver for the TinyissimoYOLO hardware accelerator.

Wraps the user.org:user:tinyissimoyolo_accelerator:1.0 IP. Handles
mode select, pixel preload via the AXI-Lite FIFO, inference control,
result readout, and post-processing (DFL decode, sigmoid, NMS).

Usage:
    from pynq import Overlay
    from software.overlay.drivers import TinyissimoYoloAcceleratorDriver

    ov = Overlay("tinyissimoyolo.bit")
    accel = TinyissimoYoloAcceleratorDriver(ov.tinyissimoyolo_accel_0)

    image = Image.open("test.jpg").convert("RGB").resize((256, 256))
    detections = accel.run(np.array(image))
"""

from __future__ import annotations

import time

import numpy as np

# Detection class names and colors (BGR)
CLASS_NAMES = {0: "chair", 1: "bowl", 2: "cup"}
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}

# Output quantization (layers 13 and 16 share the same params,
# from hardware/weights/hdl/rom_summary.json)
OUTPUT_SCALE = 0.10655
OUTPUT_ZP = 10


class TinyissimoYoloAcceleratorDriver:
    """Driver for the TinyissimoYOLO HDL accelerator IP.

    Register map matches hardware/ip_repo/src/axil_regs.sv. Handshake
    matches hardware/testbench/tb_tinyissimoyolo_accel.sv.
    """

    IP_VLNV = "user.org:user:tinyissimoyolo_accelerator:1.0"
    IP_NAME = "tinyissimoyolo_accel_0"

    # Register offsets (axil_regs.sv address constants)
    _CTRL            = 0x000
    _STATUS          = 0x004
    _MODE            = 0x008
    _CYCLE_CNT       = 0x00C
    _LAYER_IDX       = 0x010
    _RESULT_BASE_REG = 0x014   # 14-bit URAM base for result window
    _RESULT_BUF_REG  = 0x018   # 0=fmap_a, 1=fmap_b
    _PIXEL_FIFO      = 0x020
    _PIXEL_CNT       = 0x024
    _RESULT_BASE     = 0x100   # AXI-Lite result region base

    # CTRL bits (auto-clearing one-shots in axil_regs.sv:142-143)
    _CTRL_START     = 1 << 0
    _CTRL_FIFO_RST  = 1 << 1
    _CTRL_SOFT_RST  = 1 << 7

    # STATUS bits (axil_regs.sv:264)
    _STATUS_BUSY    = 1 << 0
    _STATUS_DONE    = 1 << 1
    _STATUS_IDLE    = 1 << 2
    _STATUS_PRELOAD = 1 << 3

    # Result layout: 320 URAM words × 128-bit = 1280 × 32-bit
    _CV2_WORDS = 256   # fmap_b[256..511]: 4 groups × 64 spatial
    _CV3_WORDS = 64    # fmap_b[512..575]: 1 group × 64 spatial
    _RESULT_WORDS = _CV2_WORDS + _CV3_WORDS  # 320 URAM words = 1280 32-bit reads

    def __init__(self, ip):
        """Wrap a PYNQ IP/MMIO handle exposing read(offset)/write(offset, value).

        For PYNQ DefaultIP / MMIO handles the wrapper also exposes a
        numpy uint32 view via `.array`, which `read_results_raw()` uses
        for a single bulk readout instead of 1280 individual MMIO reads.
        """
        self._ip = ip

    @property
    def status(self) -> int:
        return self._ip.read(self._STATUS)

    @property
    def cycle_count(self) -> int:
        return self._ip.read(self._CYCLE_CNT)

    @property
    def layer_idx(self) -> int:
        return self._ip.read(self._LAYER_IDX) & 0x1F

    @property
    def pixel_count(self) -> int:
        return self._ip.read(self._PIXEL_CNT)

    def set_mode(self, mode: int):
        """Set input mode: 0 = AXI-Lite FIFO, 1 = S_AXIS camera."""
        if mode not in (0, 1):
            raise ValueError(f"mode must be 0 (FIFO) or 1 (S_AXIS), got {mode}")
        self._ip.write(self._MODE, mode)

    def soft_reset(self):
        """Pulse CTRL[7] (soft_reset, auto-clearing). Restores phase FSM to IDLE."""
        self._ip.write(self._CTRL, self._CTRL_SOFT_RST)

    def configure(self, mode: int = 0):
        """Bring the accelerator into a clean configured state.

        Issues a soft reset (clears phase FSM, FIFO accumulator, pixel
        counter) before latching the mode select. Soft reset must come
        first so MODE writes land into a freshly-reset register file.

        Args:
            mode: 0 = AXI-Lite FIFO preload (default for offline test),
                  1 = S_AXIS camera streaming.
        """
        self.soft_reset()
        self.set_mode(mode)

    def read_status(self) -> dict:
        """Decode the STATUS register into a dict.

        Mirrors the convention of VdmaDriver.read_status(). Bit layout
        from axil_regs.sv:264.
        """
        raw = self._ip.read(self._STATUS)
        return {
            "busy":         bool(raw & self._STATUS_BUSY),
            "done":         bool(raw & self._STATUS_DONE),
            "idle":         bool(raw & self._STATUS_IDLE),
            "preload_done": bool(raw & self._STATUS_PRELOAD),
            "raw":          raw,
        }

    def start(self):
        """Enter PRELOAD phase and reset pixel FIFO.

        In FIFO mode: call this BEFORE write_pixels().
        In S_AXIS mode: not needed (auto-preloads on SOF).
        """
        self._ip.write(self._CTRL, self._CTRL_START | self._CTRL_FIFO_RST)

    def write_pixels(self, image_rgb: np.ndarray):
        """Write a 256x256 RGB8 image to the accelerator via FIFO.

        Converts uint8 (0-255) to int8 (-128..127) per model convention.
        Packs each pixel as a 32-bit word `[R, G, B, pad=0]` (little-
        endian → on-the-wire `{0x00, B, G, R}`), matching the testbench
        at hardware/testbench/tb_tinyissimoyolo_accel.sv:317-322.

        Every word targets the SAME register offset (PIXEL_FIFO = 0x020),
        so PYNQ's `.array` slice trick — which writes to consecutive
        offsets — does not apply. The Python loop is the floor for the
        AXI-Lite-FIFO ingress path; ~150 ms wall time for 65 536 writes
        on Kria. An AXI-Full burst ingress would be the architectural
        fix; out of scope for the smoke test.

        Args:
            image_rgb: (256, 256, 3) uint8 array, RGB channel order.

        Raises:
            ValueError: if shape or dtype do not match the accelerator's
                input contract.
        """
        if image_rgb.shape != (256, 256, 3):
            raise ValueError(
                f"image_rgb must be (256,256,3), got {image_rgb.shape}")
        if image_rgb.dtype != np.uint8:
            raise ValueError(
                f"image_rgb must be uint8, got {image_rgb.dtype}")

        pixels_int8 = (image_rgb.astype(np.int16) - 128).astype(np.int8)
        packed = np.zeros((256 * 256, 4), dtype=np.int8)
        packed[:, :3] = pixels_int8.reshape(-1, 3)

        # Materialise as a Python list once so the hot loop avoids
        # per-iteration np.uint32 → int conversion.
        words = packed.view(np.uint32).flatten().tolist()
        for w in words:
            self._ip.write(self._PIXEL_FIFO, w)

    def wait_done(self, timeout_s: float = 1.0) -> bool:
        """Poll STATUS until inference completes."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.status & self._STATUS_DONE:
                return True
        return False

    def set_result_window(self, base_addr: int, buf: int = 1):
        """Slide the AXI-Lite result region over a different URAM range.

        The accelerator's result region (`_RESULT_BASE`..`_RESULT_BASE+0x14FF`)
        reads from `{buf, base_addr + offset}` where buf selects fmap_a (0)
        or fmap_b (1) and base_addr is the URAM word address. Defaults
        (base=256, buf=1) match the original cv2/cv3 layout. Use this for
        layer-by-layer silicon bisection — point the window at the URAM
        region a particular layer wrote, then call `read_results_raw()`
        and compare against `golden_layerN_uram.mem`.

        Args:
            base_addr: 14-bit URAM word address (0..16383)
            buf: 0 selects fmap_a, 1 selects fmap_b
        """
        self._ip.write(self._RESULT_BASE_REG, base_addr & 0x3FFF)
        self._ip.write(self._RESULT_BUF_REG, buf & 0x1)

    def read_window(self, base_addr: int, buf: int, num_words: int) -> np.ndarray:
        """Read `num_words` URAM words starting at `base_addr` from `buf`.

        The AXI-Lite result region only exposes `_RESULT_WORDS` (320) URAM
        words at a time, so for larger reads this method slides the
        window forward and concatenates the chunks. Returns a
        `(num_words, 16)` int8 array.
        """
        out = np.zeros((num_words, 16), dtype=np.int8)
        pos = 0
        while pos < num_words:
            chunk = min(self._RESULT_WORDS, num_words - pos)
            self.set_result_window(base_addr + pos, buf)
            out[pos:pos + chunk] = self.read_results_raw()[:chunk]
            pos += chunk
        return out

    def read_results_raw(self) -> np.ndarray:
        """Read raw int8 detection results from URAM in a single bulk copy.

        The result region (`_RESULT_BASE` .. `_RESULT_BASE + 0x14FF`) is
        addressed at consecutive 32-bit offsets, so we can use PYNQ's
        `.array` view (a numpy uint32 over the mmap) to copy 1280 words
        in one memcpy instead of 1280 individual MMIO syscalls. Falls
        back to the per-word loop if the handle does not expose `.array`
        (e.g. a custom mock that only implements `read`/`write`).

        Returns:
            (320, 16) int8 array — 320 URAM words, 16 bytes each.
            Rows 0-255:   cv2 bbox logits (4 groups × 64 spatial)
            Rows 256-319: cv3 class logits (1 group × 64 spatial,
                          only lanes 0..2 valid)
        """
        n_words32 = self._RESULT_WORDS * 4
        array_view = getattr(self._ip, "array", None)
        if array_view is not None:
            start = self._RESULT_BASE // 4
            raw = np.asarray(
                array_view[start:start + n_words32], dtype=np.uint32
            ).copy()
        else:
            raw = np.zeros(n_words32, dtype=np.uint32)
            for i in range(n_words32):
                raw[i] = self._ip.read(self._RESULT_BASE + i * 4)
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

        Calls `configure()` first so back-to-back invocations always start
        from a clean FSM state. Without the soft reset, stale URAM
        contents from a prior run leak into the next inference and the
        result becomes non-deterministic between calls.

        Args:
            image_rgb: (256, 256, 3) uint8 RGB image.
            conf_thresh: Confidence threshold for detections.
            nms_thresh: IoU threshold for NMS.

        Returns:
            dict with keys: boxes, scores, class_ids, class_names, cycle_count
        """
        self.configure(mode=0)
        self.start()
        self.write_pixels(image_rgb)

        if not self.wait_done(timeout_s=1.0):
            raise TimeoutError("Inference did not complete within 1 second")

        cycles = self.cycle_count
        raw_table = self.read_results_raw()
        raw_tensor = self.unpack_detections(raw_table)

        boxes, scores, class_ids = post_process(raw_tensor, conf_thresh)
        boxes, scores, class_ids = nms(boxes, scores, class_ids, nms_thresh)

        return {
            "boxes":       boxes,
            "scores":      scores,
            "class_ids":   class_ids,
            "class_names": [CLASS_NAMES.get(c, f"cls{c}") for c in class_ids],
            "cycle_count": cycles,
            # Raw outputs exposed so callers (smoke runner, golden
            # comparison) don't need to re-read MMIO.
            "raw_table":   raw_table,
            "raw_tensor":  raw_tensor,
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
