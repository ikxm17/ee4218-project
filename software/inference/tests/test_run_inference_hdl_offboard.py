"""Off-board tests for software/inference/run_inference_hdl.py helpers.

Covers the pure-logic functions (image preprocessing, result saving,
golden comparison). The PYNQ-dependent main() is exercised on the board.
"""

import json
import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

from software.inference.run_inference_hdl import (
    compare_against_golden,
    load_image,
    save_results,
)
from software.overlay.drivers.tinyissimoyolo_accelerator import (
    OUTPUT_SCALE,
    OUTPUT_ZP,
)

pytestmark = pytest.mark.offboard


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

def test_load_image_returns_uint8_256x256_rgb():
    """The canonical test image must produce a (256, 256, 3) uint8
    array — same as run_inference.py:119 — so the HDL and TFLite paths
    consume identical pre-quantisation data."""
    image_path = REPO_ROOT / "software" / "inference" / "data" / "input_image.jpg"
    if not image_path.exists():
        pytest.skip(f"Test image not present: {image_path}")
    img = load_image(str(image_path))
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

def _full_result(boxes, scores, class_ids, cycle_count):
    """Build a complete result dict matching what driver.run() returns."""
    return {
        "boxes":       boxes,
        "scores":      scores,
        "class_ids":   class_ids,
        "cycle_count": cycle_count,
        "raw_tensor":  np.zeros((8, 8, 67), dtype=np.float32),
        "raw_table":   np.zeros((320, 16), dtype=np.int8),
    }


def test_save_results_writes_expected_files(tmp_path):
    """save_results must produce the same on-disk layout as
    run_inference.py so the existing visualisation script and the
    comparison harness can read both result dirs interchangeably."""
    result = _full_result(
        boxes=[[10, 20, 30, 40], [50, 60, 70, 80]],
        scores=[0.9, 0.7],
        class_ids=[0, 2],
        cycle_count=12345,
    )
    save_results(str(tmp_path), result)

    assert (tmp_path / "boxes.npy").exists()
    assert (tmp_path / "scores.npy").exists()
    assert (tmp_path / "class_ids.npy").exists()
    assert (tmp_path / "indices.npy").exists()
    raw = np.load(tmp_path / "raw_out_0.npy")
    assert raw.shape == (1, 8, 8, 67)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["source"] == "hdl"
    assert meta["is_full_int8"] is True
    assert meta["scales"] == [OUTPUT_SCALE]
    assert meta["zps"] == [OUTPUT_ZP]
    assert meta["cycle_count"] == 12345


def test_save_results_indices_match_box_count(tmp_path):
    """driver.run() returns post-NMS detections, so indices must be
    a contiguous 0..N-1 sequence (one per kept box)."""
    result = _full_result(boxes=[[1, 2, 3, 4]], scores=[0.5],
                          class_ids=[0], cycle_count=1)
    save_results(str(tmp_path), result)
    indices = np.load(tmp_path / "indices.npy")
    assert indices.tolist() == [0]


# ---------------------------------------------------------------------------
# compare_against_golden
# ---------------------------------------------------------------------------

def _write_golden(path, words_lane012):
    """Build a golden .mem file from a (64, 3) int8 array."""
    with open(path, "w") as f:
        for w in words_lane012:
            byte0 = int(w[0]) & 0xFF
            byte1 = int(w[1]) & 0xFF
            byte2 = int(w[2]) & 0xFF
            word = byte0 | (byte1 << 8) | (byte2 << 16)
            f.write(f"{word:032x}\n")


def test_compare_against_golden_pass(tmp_path, capsys):
    """When the cv3 region matches the golden file byte-for-byte,
    compare_against_golden returns True."""
    raw = np.zeros((320, 16), dtype=np.int8)
    cv3 = np.random.RandomState(0).randint(-128, 128,
                                            size=(64, 3), dtype=np.int8)
    raw[256:320, :3] = cv3
    golden_path = tmp_path / "golden.mem"
    _write_golden(golden_path, cv3)

    assert compare_against_golden(raw, str(golden_path)) is True
    captured = capsys.readouterr().out
    assert "PASS" in captured


def test_compare_against_golden_fail(tmp_path, capsys):
    """A single-byte mismatch must report FAIL with a position breakdown."""
    raw = np.zeros((320, 16), dtype=np.int8)
    cv3 = np.zeros((64, 3), dtype=np.int8)
    raw[256:320, :3] = cv3
    raw[260, 1] = 5  # introduce a mismatch in word 4 lane 1
    golden_path = tmp_path / "golden.mem"
    _write_golden(golden_path, cv3)

    assert compare_against_golden(raw, str(golden_path)) is False
    captured = capsys.readouterr().out
    assert "FAIL" in captured
    assert "word=  4 lane=1" in captured
