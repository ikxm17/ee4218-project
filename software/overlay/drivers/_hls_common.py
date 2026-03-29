"""Shared constants for Xilinx HLS-generated IPs.

All HLS IPs expose an AP_CTRL register at offset 0x00 with a common
bit layout for start/done/idle/ready handshaking.  The three constants
below cover the start + auto-restart pattern used by streaming IPs.
"""

AP_CTRL = 0x00
AP_CTRL_START = 0x01
AP_CTRL_AUTO_RESTART = 0x80


def hls_start(ip) -> None:
    """Write AP_CTRL to start an HLS IP with auto-restart enabled."""
    ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)
