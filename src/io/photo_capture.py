"""Photo capture helpers for saving composited frames."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.engine import PartyEngine

logger = logging.getLogger(__name__)

JPEG_EXTENSION = ".jpg"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"


class PhotoCapture:
    """Saves composited frames as high-quality JPEG photos.

    Args:
        quality: JPEG quality from 0-100.
    """

    def __init__(self, quality: int = 95) -> None:
        self._quality = max(0, min(100, quality))

    def capture(
        self,
        frame: np.ndarray,
        output_dir: str | Path,
        stem: str | None = None,
    ) -> Path:
        """Save a frame as a JPEG photo.

        Args:
            frame: BGR frame to save.
            output_dir: Directory where the photo should be written.
            stem: Optional filename stem without extension.

        Returns:
            Absolute path to the saved photo.

        Raises:
            RuntimeError: If OpenCV fails to write the image.
        """
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)

        filename_stem = stem or f"photo_{datetime.now().strftime(TIMESTAMP_FORMAT)}"
        path = (directory / f"{filename_stem}{JPEG_EXTENSION}").resolve()

        ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            raise RuntimeError(f"Failed to save photo to {path}")

        logger.info("Saved photo: %s", path)
        return path

    def capture_with_countdown(self, engine: "PartyEngine", seconds: int = 3) -> None:
        """Request a countdown-driven capture through the engine.

        Args:
            engine: PartyEngine instance that owns countdown state.
            seconds: Countdown duration in seconds.
        """
        engine.trigger_photo(seconds=seconds)
