"""Shared constants and helpers for Xilinx HLS-generated IPs.

All HLS IPs expose an AP_CTRL register at offset 0x00 with a common
bit layout for start/done/idle/ready handshaking.  The constants and
helpers below cover the start + auto-restart pattern and status
readback used by streaming video IPs.

Interrupt registers (GIE, IER, ISR) at 0x04-0x0C are also common
across all HLS IPs, though not used in normal streaming operation.
"""

# -- AP_CTRL register (offset 0x00) --
AP_CTRL = 0x00
AP_CTRL_START = 0x01         # bit 0 — ap_start
AP_CTRL_DONE = 0x02          # bit 1 — ap_done (read-only, clear-on-read)
AP_CTRL_IDLE = 0x04          # bit 2 — ap_idle (read-only)
AP_CTRL_READY = 0x08         # bit 3 — ap_ready (read-only)
AP_CTRL_AUTO_RESTART = 0x80  # bit 7 — auto_restart

# -- Interrupt registers (common to all HLS IPs) --
GIE = 0x04   # Global Interrupt Enable
IER = 0x08   # IP Interrupt Enable Register
ISR = 0x0C   # IP Interrupt Status Register


def hls_start(ip) -> None:
    """Write AP_CTRL to start an HLS IP with auto-restart enabled."""
    ip.write(AP_CTRL, AP_CTRL_START | AP_CTRL_AUTO_RESTART)


def hls_stop(ip) -> None:
    """Clear AP_CTRL to stop an HLS IP."""
    ip.write(AP_CTRL, 0x00)


def hls_read_status(ip) -> dict:
    """Read and decode AP_CTRL bits into a status dict.

    Returns:
        {"idle": bool, "done": bool, "ready": bool,
         "running": bool, "auto_restart": bool, "ap_ctrl_raw": int}
    """
    val = ip.read(AP_CTRL)
    idle = bool(val & AP_CTRL_IDLE)
    return {
        "idle": idle,
        "done": bool(val & AP_CTRL_DONE),
        "ready": bool(val & AP_CTRL_READY),
        "running": not idle and bool(val & (AP_CTRL_START | AP_CTRL_AUTO_RESTART)),
        "auto_restart": bool(val & AP_CTRL_AUTO_RESTART),
        "ap_ctrl_raw": val,
    }
