"""Tests for local-network URL helpers."""

import socket

from src.utils.network import LOCAL_IP_FALLBACK, detect_local_ip


class _FakeSocket:
    """Minimal socket stub for IP detection tests."""

    def __init__(self, ip: str, should_fail: bool = False) -> None:
        self._ip = ip
        self._should_fail = should_fail

    def connect(self, _address) -> None:
        """Pretend to connect or raise an error."""
        if self._should_fail:
            raise OSError("network down")

    def getsockname(self) -> tuple[str, int]:
        """Return the configured local socket address."""
        return (self._ip, 12345)

    def close(self) -> None:
        """No-op close for the socket stub."""
        return None


def test_detect_local_ip_success(monkeypatch):
    """LAN IP detection should return the chosen outbound IP."""
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *_args, **_kwargs: _FakeSocket("192.168.0.24"),
    )

    assert detect_local_ip() == "192.168.0.24"


def test_detect_local_ip_fallback(monkeypatch):
    """LAN IP detection should fall back cleanly on socket failure."""
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *_args, **_kwargs: _FakeSocket("0.0.0.0", should_fail=True),
    )

    assert detect_local_ip() == LOCAL_IP_FALLBACK


def test_detect_local_ip_empty_string(monkeypatch):
    """Empty socket addresses should use the localhost fallback."""
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *_args, **_kwargs: _FakeSocket(""),
    )

    assert detect_local_ip() == LOCAL_IP_FALLBACK
