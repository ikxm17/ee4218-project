"""Per-IP-block drivers for the camera pipeline PL IPs and sensor.

Each driver class declares an IP_NAME matching its block design instance.
Use audit_drivers() to cross-reference against a .hwh file and detect
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
}


def audit_drivers(hwh_path: str) -> dict:
    """Compare the driver registry against a .hwh file.

    Parses the .hwh (XML) to extract IP instance names, then
    cross-references with DRIVER_REGISTRY.

    Returns:
        {
            "covered":  {ip_name: DriverClass, ...},  # have driver + in hardware
            "stale":    {ip_name: DriverClass, ...},  # have driver, NOT in hardware
            "missing":  [ip_name, ...],                # in hardware, no driver
            "infra":    [ip_name, ...],                # in hardware, infrastructure (no driver needed)
        }
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(hwh_path)
    root = tree.getroot()

    # .hwh format: <MODULES><MODULE INSTANCE="ip_name" VLNV="..."> ...
    hw_ips = set()
    for module in root.iter("MODULE"):
        name = module.get("INSTANCE")
        if name:
            hw_ips.add(name)

    covered = {}
    stale = {}
    for ip_name, driver_cls in DRIVER_REGISTRY.items():
        if ip_name in hw_ips:
            covered[ip_name] = driver_cls
        else:
            stale[ip_name] = driver_cls

    driver_ip_names = set(DRIVER_REGISTRY.keys())
    infra = []
    missing = []
    for ip_name in sorted(hw_ips - driver_ip_names):
        # Check if any infrastructure prefix matches
        if any(ip_name.startswith(prefix) for prefix in _INFRA_IPS):
            infra.append(ip_name)
        else:
            missing.append(ip_name)

    return {
        "covered": covered,
        "stale": stale,
        "missing": missing,
        "infra": infra,
    }
