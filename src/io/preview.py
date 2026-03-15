"""Preview window module."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

QUIT_KEYS = {ord("q"), 27}  # 'q' and ESC


class PreviewWindow:
    """Displays frames in an OpenCV window.

    Args:
        window_name: Title of the display window.
    """

    def __init__(self, window_name: str = "posefx-studio") -> None:
        self._window_name = window_name
        self._last_key = -1

    def show(self, frame: np.ndarray) -> None:
        """Display a frame in the window.

        Args:
            frame: BGR image to display.
        """
        cv2.imshow(self._window_name, frame)
        self._last_key = cv2.waitKey(1) & 0xFF

    def should_quit(self) -> bool:
        """Check if the user pressed a quit key ('q' or ESC).

        Returns:
            True if the user wants to quit.
        """
        return self._last_key in QUIT_KEYS

    def destroy(self) -> None:
        """Close the preview window."""
        cv2.destroyAllWindows()
        logger.info("Preview window destroyed")
