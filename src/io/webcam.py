"""Webcam capture module."""

import logging

import cv2
import numpy as np

from src.utils.config import CameraConfig

logger = logging.getLogger(__name__)


class WebcamCapture:
    """Opens a camera device using OpenCV and provides frame capture.

    Args:
        config: Camera configuration with device_id, width, height, fps.
    """

    def __init__(self, config: CameraConfig) -> None:
        self._cap = cv2.VideoCapture(config.device_id)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera device {config.device_id}. "
                "Check that the camera is connected and not in use by another application."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self._cap.set(cv2.CAP_PROP_FPS, config.fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera opened: %dx%d @ %.1f fps (requested %dx%d @ %d fps)",
            actual_w, actual_h, actual_fps,
            config.width, config.height, config.fps,
        )

    def read(self) -> tuple[bool, np.ndarray]:
        """Read the next frame from the camera.

        Returns:
            Tuple of (success, frame). Frame is a BGR numpy array.
        """
        return self._cap.read()

    def release(self) -> None:
        """Release the camera device."""
        if self._cap.isOpened():
            self._cap.release()
            logger.info("Camera released")
