"""Off-board tests for software/overlay/tests/checks.py helpers.

Covers golden .mem parsing used by the accelerator onboard test and
the run_inference_hdl smoke runner.
"""

import pathlib
import textwrap

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

from software.overlay.tests.checks import (
    find_accel_bitstream,
    load_golden_uram_mem,
)

pytestmark = pytest.mark.offboard


# ---------------------------------------------------------------------------
# find_accel_bitstream
# ---------------------------------------------------------------------------

def _make_hwh(path, contains_accel: bool):
    body = (
        '<?xml version="1.0"?><EDKSYSTEM><MODULES>'
        + ('<MODULE INSTANCE="tinyissimoyolo_accel_0" '
           'VLNV="user.org:user:tinyissimoyolo_accelerator:1.0"/>'
           if contains_accel else
           '<MODULE INSTANCE="v_demosaic_0" '
           'VLNV="xilinx.com:ip:v_demosaic:1.1"/>')
        + '</MODULES></EDKSYSTEM>'
    )
    path.write_text(body)


def test_find_accel_bitstream_picks_matching_hwh(tmp_path):
    """When two .bit files coexist, pick the one whose sibling .hwh
    references the accelerator VLNV. This protects the existing
    `overlay` fixture (which alphabetically picks camera_pipeline.bit)
    from being broken by adding the accelerator bitstream."""
    cam_bit = tmp_path / "camera_pipeline.bit"
    cam_hwh = tmp_path / "camera_pipeline.hwh"
    cam_bit.write_bytes(b"")
    _make_hwh(cam_hwh, contains_accel=False)

    accel_bit = tmp_path / "tinyissimoyolo.bit"
    accel_hwh = tmp_path / "tinyissimoyolo.hwh"
    accel_bit.write_bytes(b"")
    _make_hwh(accel_hwh, contains_accel=True)

    found = find_accel_bitstream(str(tmp_path))
    assert found == str(accel_bit)


def test_find_accel_bitstream_returns_none_when_missing(tmp_path):
    """If no .bit / .hwh references the accelerator, return None so the
    onboard fixture can pytest.skip cleanly."""
    cam_bit = tmp_path / "camera_pipeline.bit"
    cam_hwh = tmp_path / "camera_pipeline.hwh"
    cam_bit.write_bytes(b"")
    _make_hwh(cam_hwh, contains_accel=False)

    assert find_accel_bitstream(str(tmp_path)) is None


def test_find_accel_bitstream_handles_empty_dir(tmp_path):
    assert find_accel_bitstream(str(tmp_path)) is None


def test_find_accel_bitstream_skips_bit_without_sibling_hwh(tmp_path):
    """A .bit without a paired .hwh cannot be classified, so it must be
    ignored rather than crashing."""
    orphan_bit = tmp_path / "stale.bit"
    orphan_bit.write_bytes(b"")
    accel_bit = tmp_path / "tinyissimoyolo.bit"
    accel_hwh = tmp_path / "tinyissimoyolo.hwh"
    accel_bit.write_bytes(b"")
    _make_hwh(accel_hwh, contains_accel=True)
    assert find_accel_bitstream(str(tmp_path)) == str(accel_bit)


def test_load_golden_uram_mem_unpacks_lsb_first(tmp_path):
    """A 128-bit hex word `00..00 c4 b3 ec` (MSB→LSB) must produce
    int8 lanes [0xec, 0xb3, 0xc4, 0, 0, ..., 0] in lane order, with the
    first three bytes interpreted as signed (i.e. negative for 0x80+).

    This format mirrors hardware/testbench/inference_hdl/golden_layer16_uram.mem
    where layer 16 (cv3 head) holds 3 valid channels per spatial position
    in lanes 0..2 and zero-pad in lanes 3..15.
    """
    mem_file = tmp_path / "tiny_golden.mem"
    mem_file.write_text(textwrap.dedent("""\
        00000000000000000000000000c4b3ec
        00000000000000000000000000010203
    """))

    out = load_golden_uram_mem(str(mem_file), num_words=2)

    assert out.shape == (2, 16)
    assert out.dtype == np.int8
    # Word 0: lanes 0..2 = ec, b3, c4 (signed: -20, -77, -60)
    assert out[0, 0] == np.int8(-20)
    assert out[0, 1] == np.int8(-77)
    assert out[0, 2] == np.int8(-60)
    assert (out[0, 3:] == 0).all()
    # Word 1: lanes 0..2 = 03, 02, 01
    assert out[1, 0] == 3
    assert out[1, 1] == 2
    assert out[1, 2] == 1
    assert (out[1, 3:] == 0).all()


def test_load_golden_uram_mem_truncates_to_num_words(tmp_path):
    """Reading more lines than requested is silently ignored — the
    onboard test only consumes the cv3 region (64 words) of golden
    files that may hold more (e.g. cv2 layer 13 has 256 words)."""
    mem_file = tmp_path / "long.mem"
    lines = [f"{i:032x}" for i in range(10)]
    mem_file.write_text("\n".join(lines) + "\n")

    out = load_golden_uram_mem(str(mem_file), num_words=3)

    assert out.shape == (3, 16)
    # Word 0 = 0, word 1 = 1 (lane 0 = 1), word 2 = 2 (lane 0 = 2)
    assert out[0, 0] == 0
    assert out[1, 0] == 1
    assert out[2, 0] == 2


def test_load_golden_uram_mem_signed_full_range(tmp_path):
    """Lane bytes 0x00..0xFF map to signed int8 [-128..127].

    Lane 0 = 0x7F (127), lane 1 = 0x80 (-128), lane 2 = 0xFF (-1).
    """
    mem_file = tmp_path / "signed.mem"
    mem_file.write_text("00000000000000000000000000ff807f\n")

    out = load_golden_uram_mem(str(mem_file), num_words=1)

    assert out[0, 0] == np.int8(0x7F)   # 127
    assert out[0, 1] == np.int8(-128)   # 0x80
    assert out[0, 2] == np.int8(-1)     # 0xFF
    assert (out[0, 3:] == 0).all()


def test_load_golden_uram_mem_real_layer16_first_word():
    """Sanity check against the real golden file shipped with the testbench.

    First line of hardware/testbench/inference_hdl/golden_layer16_uram.mem
    is `...00c4b3ec` → cv3 channels at spatial (0,0): -20, -77, -60.
    """
    golden = (REPO_ROOT / "hardware" / "testbench" / "inference_hdl"
              / "golden_layer16_uram.mem")
    if not golden.exists():
        pytest.skip(f"Golden file not present: {golden}")
    out = load_golden_uram_mem(str(golden), num_words=64)
    assert out.shape == (64, 16)
    assert out[0, 0] == np.int8(-20)
    assert out[0, 1] == np.int8(-77)
    assert out[0, 2] == np.int8(-60)
