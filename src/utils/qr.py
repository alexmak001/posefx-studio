"""QR code helpers for local phone access."""

from __future__ import annotations

from io import BytesIO


def build_qr_png(content: str) -> bytes:
    """Build a PNG QR code for the given content.

    Args:
        content: Text or URL to encode into the QR.

    Returns:
        PNG bytes representing the QR image.

    Raises:
        RuntimeError: If the QR dependency is not installed.
    """
    try:
        import qrcode
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "QR support requires the 'qrcode[pil]' package. Run 'uv sync'."
        ) from exc

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(content)
    qr.make(fit=True)

    image = qr.make_image(fill_color="black", back_color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
