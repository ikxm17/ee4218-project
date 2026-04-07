"""Off-board tests for software/inference/compare_hdl_vs_tflite.py.

The comparison logic is pure NumPy + filesystem reads, so it's fully
testable without hardware. Written test-first per TDD discipline.
"""

import json

import numpy as np
import pytest

from software.inference.compare_hdl_vs_tflite import (
    compare_detections,
    compare_raw_tensors,
    dequantise_tflite_raw,
    load_result_dir,
)

pytestmark = pytest.mark.offboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_result_dir(tmp_path, *, raw, boxes, scores, class_ids,
                      scales, zps, source="tflite"):
    np.save(tmp_path / "raw_out_0.npy", raw)
    np.save(tmp_path / "boxes.npy", np.array(boxes))
    np.save(tmp_path / "scores.npy", np.array(scores))
    np.save(tmp_path / "class_ids.npy", np.array(class_ids))
    np.save(tmp_path / "indices.npy", np.arange(len(boxes)))
    meta = {"source": source, "is_full_int8": True,
            "scales": scales, "zps": zps}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


# ---------------------------------------------------------------------------
# load_result_dir
# ---------------------------------------------------------------------------

def test_load_result_dir_returns_all_arrays(tmp_path):
    """load_result_dir must return raw, boxes, scores, class_ids, meta
    so the diff routines have everything they need without re-reading."""
    raw = np.zeros((1, 8, 8, 67), dtype=np.uint8)
    _write_result_dir(tmp_path, raw=raw, boxes=[[1,2,3,4]],
                      scores=[0.9], class_ids=[1],
                      scales=[0.10655], zps=[10])
    bundle = load_result_dir(str(tmp_path))
    assert bundle["raw"].shape == (1, 8, 8, 67)
    assert bundle["boxes"].tolist() == [[1, 2, 3, 4]]
    assert bundle["scores"].tolist() == [0.9]
    assert bundle["class_ids"].tolist() == [1]
    assert bundle["meta"]["source"] == "tflite"
    assert bundle["meta"]["scales"] == [0.10655]


# ---------------------------------------------------------------------------
# dequantise_tflite_raw
# ---------------------------------------------------------------------------

def test_dequantise_tflite_raw_uses_scale_and_zp():
    """dequantise = (raw - zp) * scale, matching the formula in
    run_inference.py:70."""
    raw = np.array([[10, 20, 30]], dtype=np.uint8)
    out = dequantise_tflite_raw(raw, scale=0.5, zp=10)
    np.testing.assert_allclose(out, [[0.0, 5.0, 10.0]])


def test_dequantise_tflite_raw_handles_int8():
    """Int8 quantisation paths produce the same float result."""
    raw = np.array([[-1, 0, 1]], dtype=np.int8)
    out = dequantise_tflite_raw(raw, scale=2.0, zp=-1)
    np.testing.assert_allclose(out, [[0.0, 2.0, 4.0]])


# ---------------------------------------------------------------------------
# compare_raw_tensors
# ---------------------------------------------------------------------------

def test_compare_raw_tensors_identical():
    """Two identical tensors → max|diff|=0, mean|diff|=0."""
    a = np.full((1, 8, 8, 67), 1.5, dtype=np.float32)
    b = np.full((1, 8, 8, 67), 1.5, dtype=np.float32)
    stats = compare_raw_tensors(a, b)
    assert stats["max_abs_diff"] == 0.0
    assert stats["mean_abs_diff"] == 0.0


def test_compare_raw_tensors_difference():
    """Element-wise diff stats are correct on a known offset."""
    a = np.zeros((1, 8, 8, 67), dtype=np.float32)
    b = np.full((1, 8, 8, 67), 0.5, dtype=np.float32)
    stats = compare_raw_tensors(a, b)
    assert stats["max_abs_diff"] == pytest.approx(0.5)
    assert stats["mean_abs_diff"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compare_detections
# ---------------------------------------------------------------------------

def test_compare_detections_perfect_match():
    """Same boxes, same classes → all matched, IoU=1, no extras."""
    boxes = np.array([[10, 10, 20, 20], [50, 50, 30, 30]])
    classes = np.array([0, 1])
    report = compare_detections(boxes_a=boxes, classes_a=classes,
                                boxes_b=boxes, classes_b=classes)
    assert report["matched"] == 2
    assert report["unmatched_a"] == 0
    assert report["unmatched_b"] == 0
    assert all(m["iou"] >= 0.99 for m in report["matches"])
    assert all(m["class_match"] for m in report["matches"])


def test_compare_detections_no_overlap():
    """Boxes far apart → IoU=0, no matches above threshold."""
    boxes_a = np.array([[10, 10, 20, 20]])
    boxes_b = np.array([[200, 200, 20, 20]])
    classes_a = np.array([0])
    classes_b = np.array([0])
    report = compare_detections(boxes_a=boxes_a, classes_a=classes_a,
                                boxes_b=boxes_b, classes_b=classes_b,
                                iou_thresh=0.5)
    assert report["matched"] == 0
    assert report["unmatched_a"] == 1
    assert report["unmatched_b"] == 1


def test_compare_detections_class_mismatch_still_matches_geometrically():
    """Same box, different class → matched=1 but class_match=False."""
    boxes = np.array([[10, 10, 20, 20]])
    report = compare_detections(boxes_a=boxes, classes_a=np.array([0]),
                                boxes_b=boxes, classes_b=np.array([1]))
    assert report["matched"] == 1
    assert report["matches"][0]["class_match"] is False
