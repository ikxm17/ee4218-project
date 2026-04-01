"""Per-IP-block drivers for the camera pipeline PL IPs and sensor.

Each driver class declares:
  - IP_VLNV: the Xilinx IP type (vendor:lib:name:ver) — used for audit
             matching. One driver class covers ALL instances of that type.
  - IP_NAME: the primary block design instance name — convenience only.

Drivers are instance-agnostic: the class takes any PYNQ IP handle.
Multiple instances of the same IP type reuse the same driver class.

Use audit_drivers() to cross-reference against a .xsa/.hwh and detect
IPs that need new drivers.
"""

from .csi2_rx import Csi2RxDriver
from .demosaic import DemosaicDriver
from .gamma_lut import GammaLutDriver
from .imx219 import Imx219Driver
from .vdma import VdmaDriver
from .vpss import VpssScalerDriver

__all__ = [
    "Csi2RxDriver",
    "DemosaicDriver",
    "GammaLutDriver",
    "Imx219Driver",
    "VdmaDriver",
    "VpssScalerDriver",
    "DRIVER_REGISTRY",
    "audit_drivers",
]

# All driver classes. Order doesn't matter.
_DRIVER_CLASSES = [
    Csi2RxDriver,
    DemosaicDriver,
    GammaLutDriver,
    Imx219Driver,
    VdmaDriver,
    VpssScalerDriver,
]

# Registry: maps IP VLNV (type) to driver class.
# One driver class covers all instances of that IP type.
DRIVER_REGISTRY = {cls.IP_VLNV: cls for cls in _DRIVER_CLASSES}

# IPs that are infrastructure (no driver needed). Matched by VLNV prefix.
_INFRA_VLNVS = {
    "xilinx.com:ip:proc_sys_reset",
    "xilinx.com:ip:axi_interconnect",
    "xilinx.com:ip:smartconnect",
    "xilinx.com:ip:xlconcat",
    "xilinx.com:ip:xlslice",
    "xilinx.com:ip:xlconstant",
    "xilinx.com:ip:zynq_ultra_ps_e",
    "xilinx.com:ip:clk_wiz",
    "xilinx.com:ip:axis_data_fifo",
    "xilinx.com:ip:axis_subset_converter",
    "xilinx.com:ip:axis_register_slice",
    "xilinx.com:ip:axis_broadcaster",
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
            # The .xsa contains sub-IP .hwh files alongside the top-level
            # design .hwh.  Use xsa.json topModuleName to find the right one.
            top_hwh = hwh_names[0]
            if "xsa.json" in zf.namelist():
                import json
                meta = json.loads(zf.read("xsa.json"))
                top_name = meta.get("topModuleName", "").removesuffix("_wrapper")
                for n in hwh_names:
                    if n == f"{top_name}.hwh":
                        top_hwh = n
                        break
            hwh_data = zf.read(top_hwh)
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


def _vlnv_type(vlnv: str) -> str:
    """Strip version from VLNV to get the IP type key.

    'xilinx.com:ip:axi_vdma:6.3' -> 'xilinx.com:ip:axi_vdma'
    """
    parts = vlnv.rsplit(":", 1)
    return parts[0] if len(parts) == 2 else vlnv


def audit_drivers(xsa_or_hwh_path: str) -> dict:
    """Compare the driver registry against a .xsa or .hwh file.

    Matches by IP type (VLNV without version), so multiple instances of the
    same IP (e.g. axi_vdma_0, axi_vdma_1) are covered by one driver class.

    Args:
        xsa_or_hwh_path: Path to a .xsa (preferred) or .hwh file.

    Returns:
        {
            "source":   str,
            "covered":  {ip_name: {"driver": cls, "vlnv": str}, ...},
            "stale":    {vlnv_type: {"driver": cls}, ...},
            "missing":  {ip_name: {"vlnv": str}, ...},
            "infra":    [ip_name, ...],
        }
    """
    hw_ips = _parse_hwh(xsa_or_hwh_path)

    # Build a set of VLNV types present in hardware
    hw_vlnv_types = {_vlnv_type(info["vlnv"]) for info in hw_ips.values()}

    # Check which driver VLNVs are present in hardware
    covered_vlnvs = set()
    stale = {}
    for vlnv, driver_cls in DRIVER_REGISTRY.items():
        vtype = _vlnv_type(vlnv)
        if vtype in hw_vlnv_types:
            covered_vlnvs.add(vtype)
        else:
            stale[vlnv] = {"driver": driver_cls}

    # Classify each hardware IP
    covered = {}
    missing = {}
    infra = []
    for ip_name, info in sorted(hw_ips.items()):
        vlnv = info["vlnv"]
        vtype = _vlnv_type(vlnv)

        if vtype in covered_vlnvs:
            covered[ip_name] = {
                "driver": DRIVER_REGISTRY.get(vlnv) or DRIVER_REGISTRY.get(
                    next(k for k in DRIVER_REGISTRY if _vlnv_type(k) == vtype)
                ),
                "vlnv": vlnv,
            }
        elif any(vlnv.startswith(prefix) for prefix in _INFRA_VLNVS):
            infra.append(ip_name)
        else:
            missing[ip_name] = {"vlnv": vlnv}

    return {
        "source": xsa_or_hwh_path,
        "covered": covered,
        "stale": stale,
        "missing": missing,
        "infra": infra,
    }
