"""Video file input module for headless testing."""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoFileInput:
    """Opens a video file using OpenCV and provides frame capture.

    Loops back to the beginning when the video ends, allowing
    continuous testing without a webcam.

    Args:
        path: Path to the video file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")

        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self._path}")

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            "Video opened: %s (%dx%d @ %.1f fps, %d frames)",
            self._path.name, width, height, fps, frame_count,
        )

    def read(self) -> tuple[bool, np.ndarray]:
        """Read the next frame from the video.

        Loops back to the start when the video ends.

        Returns:
            Tuple of (success, frame). Frame is a BGR numpy array.
        """
        ok, frame = self._cap.read()
        if not ok:
            # Loop back to the beginning
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if ok:
                logger.debug("Video looped back to start")
        return ok, frame

    def release(self) -> None:
        """Release the video file."""
        if self._cap.isOpened():
            self._cap.release()
            logger.info("Video file released")
