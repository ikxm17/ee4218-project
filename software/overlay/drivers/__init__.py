"""Per-IP-block drivers for the camera pipeline PL IPs and sensor.

Each driver class declares an IP_NAME matching its block design instance.
Use audit_drivers() to cross-reference against a .xsa/.hwh and detect
stale drivers or IPs that need new drivers.
"""

from .csi2_rx import Csi2RxDriver
from .demosaic import DemosaicDriver
from .gamma_lut import GammaLutDriver
from .imx219 import Imx219Driver
from .vdma import VdmaDriver

__all__ = [
    "Csi2RxDriver",
    "DemosaicDriver",
    "GammaLutDriver",
    "Imx219Driver",
    "VdmaDriver",
    "DRIVER_REGISTRY",
    "audit_drivers",
]

# Registry: maps block design IP instance names to driver classes.
# Built automatically from each driver's IP_NAME attribute.
DRIVER_REGISTRY = {
    cls.IP_NAME: cls
    for cls in [Csi2RxDriver, DemosaicDriver, GammaLutDriver, Imx219Driver, VdmaDriver]
}

# IPs that are infrastructure (no driver needed). Extend as the block
# design grows — these are filtered out of the "missing driver" report.
_INFRA_IPS = {
    "proc_sys_reset",
    "axi_interconnect",
    "smartconnect",
    "xlconcat",
    "zynq_ultra_ps_e",
    "clk_wiz",
    "axis_data_fifo",
    "axis_subset_converter",
}


def _parse_hwh(hwh_source: str) -> dict:
    """Parse a .hwh file (or extract from .xsa) and return IP metadata.

    Args:
        hwh_source: Path to a .hwh file or a .xsa file (ZIP containing .hwh).

    Returns:
        {ip_instance_name: {"vlnv": "vendor:lib:name:ver", "params": {...}}, ...}
    """
    import io
    import xml.etree.ElementTree as ET
    import zipfile

    if hwh_source.endswith(".xsa"):
        with zipfile.ZipFile(hwh_source) as zf:
            hwh_names = [n for n in zf.namelist() if n.endswith(".hwh")]
            if not hwh_names:
                raise FileNotFoundError(f"No .hwh found inside {hwh_source}")
            hwh_data = zf.read(hwh_names[0])
            root = ET.parse(io.BytesIO(hwh_data)).getroot()
    else:
        root = ET.parse(hwh_source).getroot()

    hw_ips = {}
    for module in root.iter("MODULE"):
        name = module.get("INSTANCE")
        if not name:
            continue
        vlnv = module.get("VLNV", "")
        # Extract selected parameters (address range, HLS flag, etc.)
        params = {}
        for param in module.iter("PARAMETER"):
            pname = param.get("NAME", "")
            pval = param.get("VALUE", "")
            if pname in ("C_BASEADDR", "C_HIGHADDR", "C_S_AXI_ADDR_WIDTH",
                         "C_CSI_EN_ACTIVELANES", "C_HS_LINE_RATE"):
                params[pname] = pval
        hw_ips[name] = {"vlnv": vlnv, "params": params}

    return hw_ips


def audit_drivers(xsa_or_hwh_path: str) -> dict:
    """Compare the driver registry against a .xsa or .hwh file.

    Parses the hardware description to extract IP instance names, then
    cross-references with DRIVER_REGISTRY.

    Args:
        xsa_or_hwh_path: Path to a .xsa (preferred) or .hwh file.

    Returns:
        {
            "source":   str,                                # path used
            "covered":  {ip_name: {"driver": cls, "vlnv": str}, ...},
            "stale":    {ip_name: {"driver": cls}, ...},
            "missing":  {ip_name: {"vlnv": str}, ...},
            "infra":    [ip_name, ...],
        }
    """
    hw_ips = _parse_hwh(xsa_or_hwh_path)
    hw_ip_names = set(hw_ips.keys())

    covered = {}
    stale = {}
    for ip_name, driver_cls in DRIVER_REGISTRY.items():
        if ip_name in hw_ip_names:
            covered[ip_name] = {
                "driver": driver_cls,
                "vlnv": hw_ips[ip_name]["vlnv"],
            }
        else:
            stale[ip_name] = {"driver": driver_cls}

    driver_ip_names = set(DRIVER_REGISTRY.keys())
    infra = []
    missing = {}
    for ip_name in sorted(hw_ip_names - driver_ip_names):
        if any(ip_name.startswith(prefix) for prefix in _INFRA_IPS):
            infra.append(ip_name)
        else:
            missing[ip_name] = {"vlnv": hw_ips[ip_name]["vlnv"]}

    return {
        "source": xsa_or_hwh_path,
        "covered": covered,
        "stale": stale,
        "missing": missing,
        "infra": infra,
    }
