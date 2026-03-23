"""Tests for QR code generation."""

from src.utils.qr import build_qr_png

PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def test_build_qr_png_returns_png_bytes():
    """QR generation should return a PNG payload."""
    payload = build_qr_png("http://192.168.1.20:8000")

    assert payload.startswith(PNG_HEADER)
    assert len(payload) > len(PNG_HEADER)
