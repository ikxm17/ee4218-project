"""Shared pytest fixtures for PL driver testing.

Off-board tests use MockIP to exercise driver logic without hardware.
On-board tests use PYNQ to interact with real IPs on the Kria board.
"""

import os
import pathlib

import pytest


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "offboard: runs without hardware")
    config.addinivalue_line("markers", "onboard: requires Kria board")


# ---------------------------------------------------------------------------
# Board detection
# ---------------------------------------------------------------------------

def _on_kria() -> bool:
    """Return True when running on a Kria board with ZOCL loaded."""
    return pathlib.Path("/dev/dri/renderD128").exists()


def pytest_collection_modifyitems(config, items):
    """Auto-skip on-board tests when not on Kria."""
    if _on_kria():
        return
    skip = pytest.mark.skip(reason="not on Kria board (no /dev/dri/renderD128)")
    for item in items:
        if "onboard" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# MockIP — dict-backed read()/write() for off-board tests
# ---------------------------------------------------------------------------

class MockIP:
    """Minimal PYNQ IP mock: dict-backed register read/write."""

    def __init__(self, defaults: dict | None = None):
        self._regs: dict[int, int] = dict(defaults or {})

    def write(self, offset: int, value: int) -> None:
        self._regs[offset] = value

    def read(self, offset: int) -> int:
        return self._regs.get(offset, 0)

    @property
    def regs(self) -> dict[int, int]:
        """Direct access to register dict for assertions."""
        return self._regs


@pytest.fixture
def mock_ip():
    """Fresh MockIP instance."""
    return MockIP()


# ---------------------------------------------------------------------------
# On-board fixtures (session-scoped — overlay loads once)
# ---------------------------------------------------------------------------

# IPs that must NEVER be started (hardware bug — permanent AXI hang)
_NEVER_START_VLNVS = {"xilinx.com:ip:v_multi_scaler"}


@pytest.fixture(scope="session")
def overlay():
    """Load the PYNQ overlay (bitstream + hwh). Session-scoped."""
    pynq = pytest.importorskip("pynq", reason="pynq not available")
    import glob as globmod

    # Find the .bit file
    hw_dir = pathlib.Path(__file__).parents[3] / "hardware" / "output"
    bits = sorted(hw_dir.glob("*.bit"))
    if not bits:
        pytest.skip("No .bit file in hardware/output/")
    ol = pynq.Overlay(str(bits[0]))
    yield ol
    ol.free()


@pytest.fixture(scope="session")
def demosaic_ip(overlay):
    """PYNQ IP handle for v_demosaic_0."""
    return overlay.ip_dict.get("v_demosaic_0") and overlay.v_demosaic_0


@pytest.fixture(scope="session")
def gamma_lut_ip(overlay):
    """PYNQ IP handle for v_gamma_lut_0."""
    return overlay.ip_dict.get("v_gamma_lut_0") and overlay.v_gamma_lut_0


@pytest.fixture(scope="session")
def vdma0_ip(overlay):
    """MMIO handle for axi_vdma_0.

    Uses raw MMIO instead of PYNQ's AxiVDMA driver because the design
    doesn't connect VDMA interrupts (s2mm_introut), causing AxiVDMA
    __init__ to fail with AttributeError.
    """
    pynq = pytest.importorskip("pynq")
    ip_data = overlay.ip_dict.get("axi_vdma_0")
    if ip_data is None:
        pytest.skip("axi_vdma_0 not in overlay")
    return pynq.MMIO(ip_data["phys_addr"], ip_data["addr_range"])


@pytest.fixture(scope="session")
def vdma1_ip(overlay):
    """MMIO handle for axi_vdma_1."""
    pynq = pytest.importorskip("pynq")
    ip_data = overlay.ip_dict.get("axi_vdma_1")
    if ip_data is None:
        pytest.skip("axi_vdma_1 not in overlay")
    return pynq.MMIO(ip_data["phys_addr"], ip_data["addr_range"])


@pytest.fixture(scope="session")
def csi2_rx_mmio(overlay):
    """PYNQ MMIO handle for mipi_csi2_rx_subsyst_0."""
    pynq = pytest.importorskip("pynq")
    ip_data = overlay.ip_dict.get("mipi_csi2_rx_subsyst_0")
    if ip_data is None:
        pytest.skip("mipi_csi2_rx_subsyst_0 not in overlay")
    addr = ip_data["phys_addr"]
    size = ip_data["addr_range"]
    return pynq.MMIO(addr, size)


@pytest.fixture(scope="session")
def vpss_ip(overlay):
    """MMIO handle for v_proc_ss_0."""
    pynq = pytest.importorskip("pynq")
    ip_data = overlay.ip_dict.get("v_proc_ss_0")
    if ip_data is None:
        pytest.skip("v_proc_ss_0 not in overlay")
    return pynq.MMIO(ip_data["phys_addr"], ip_data["addr_range"])
