"""Network helpers for local phone access."""

from __future__ import annotations

import socket

LOCAL_IP_FALLBACK = "localhost"


def detect_local_ip() -> str:
    """Return the preferred LAN IP for clients on the same network.

    This uses a UDP socket connect to let the OS pick the outbound interface
    without sending any traffic. If detection fails, it falls back to
    ``localhost`` so the app still has a predictable URL.

    Returns:
        Best-effort LAN IP or ``localhost``.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except OSError:
        return LOCAL_IP_FALLBACK
    finally:
        sock.close()

    return ip or LOCAL_IP_FALLBACK
