"""Video recorder for composited output frames."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FOURCC = "mp4v"
FILE_EXTENSION = ".mp4"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"


class VideoRecorder:
    """Records composited frames to an MP4 file."""

    def __init__(self) -> None:
        self._writer: cv2.VideoWriter | None = None
        self._output_path: Path | None = None
        self._resolution: tuple[int, int] | None = None
        self._fps = 0.0
        self._warned_resize = False

    @property
    def is_recording(self) -> bool:
        """Whether a recording is currently active."""
        return self._writer is not None

    def start(
        self,
        output_dir: str | Path,
        fps: float,
        resolution: tuple[int, int],
        stem: str | None = None,
    ) -> Path:
        """Start recording to a new MP4 file.

        Args:
            output_dir: Directory where the recording should be written.
            fps: Output frames per second.
            resolution: Frame size as (width, height).
            stem: Optional filename stem without extension.

        Returns:
            Absolute path to the recording file.

        Raises:
            RuntimeError: If a recording is already active or the writer cannot open.
        """
        if self.is_recording:
            raise RuntimeError("Recording is already active")

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)

        width, height = resolution
        filename_stem = stem or f"recording_{datetime.now().strftime(TIMESTAMP_FORMAT)}"
        output_path = (directory / f"{filename_stem}{FILE_EXTENSION}").resolve()
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*FOURCC),
            max(1.0, float(fps)),
            (int(width), int(height)),
        )

        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        self._writer = writer
        self._output_path = output_path
        self._resolution = (int(width), int(height))
        self._fps = max(1.0, float(fps))
        self._warned_resize = False
        logger.info(
            "Started recording: %s (%dx%d @ %.2f FPS)",
            output_path,
            width,
            height,
            self._fps,
        )
        return output_path

    def add_frame(self, frame: np.ndarray) -> None:
        """Append one frame to the active recording.

        Args:
            frame: BGR frame to write.
        """
        if not self.is_recording or self._writer is None or self._resolution is None:
            return

        expected_width, expected_height = self._resolution
        if frame.shape[1] != expected_width or frame.shape[0] != expected_height:
            if not self._warned_resize:
                logger.warning(
                    "Recording frame size %sx%s does not match expected %sx%s; resizing",
                    frame.shape[1],
                    frame.shape[0],
                    expected_width,
                    expected_height,
                )
                self._warned_resize = True
            frame = cv2.resize(frame, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)

        self._writer.write(frame)

    def stop(self) -> Path | None:
        """Stop the current recording and finalize the file.

        Returns:
            Saved recording path, or None if no recording was active.
        """
        if not self.is_recording or self._writer is None:
            return None

        self._writer.release()
        output_path = self._output_path
        self._writer = None
        self._output_path = None
        self._resolution = None
        self._fps = 0.0
        self._warned_resize = False

        if output_path is not None:
            logger.info("Stopped recording: %s", output_path)
        return output_path
