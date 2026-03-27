"""Preview window module."""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

QUIT_KEYS = {ord("q"), 27}  # 'q' and ESC


def _has_display() -> bool:
    """Return True if a graphical display is available."""
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    # OpenCV headless build or no display set
    return False


def _get_screen_resolution() -> tuple[int, int] | None:
    """Return (width, height) of the primary display, or None if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["xrandr"],
            capture_output=True, text=True, timeout=3,
            env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")},
        )
        for line in result.stdout.splitlines():
            if "*" in line:
                parts = line.split()
                if parts:
                    w, h = parts[0].split("x")
                    return int(w), int(h)
    except Exception:
        pass
    return None


class PreviewWindow:
    """Displays frames in an OpenCV window, or no-ops when headless.

    Args:
        window_name: Title of the display window.
        fullscreen: If True, fill the entire screen.
    """

    def __init__(self, window_name: str = "posefx-studio", fullscreen: bool = False) -> None:
        self._window_name = window_name
        self._last_key = -1
        self._fullscreen = fullscreen
        self._screen_size: tuple[int, int] | None = None
        self._headless = not _has_display()
        if self._headless:
            logger.info("No display detected — preview window disabled (web stream still active)")
        elif fullscreen:
            self._screen_size = _get_screen_resolution()
            if self._screen_size:
                logger.info("Fullscreen mode: %dx%d", *self._screen_size)
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self._window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, frame: np.ndarray) -> None:
        """Display a frame in the window.

        Args:
            frame: BGR image to display.
        """
        if self._headless:
            return
        if self._fullscreen and self._screen_size is not None:
            sw, sh = self._screen_size
            fh, fw = frame.shape[:2]
            if fw != sw or fh != sh:
                frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self._window_name, frame)
        self._last_key = cv2.waitKey(1) & 0xFF

    @property
    def last_key(self) -> int:
        """The last key code captured by waitKey, or -1 if none."""
        return self._last_key

    def should_quit(self) -> bool:
        """Check if the user pressed a quit key ('q' or ESC).

        Returns:
            True if the user wants to quit.
        """
        return self._last_key in QUIT_KEYS

    def destroy(self) -> None:
        """Close the preview window."""
        if not self._headless:
            cv2.destroyAllWindows()
        logger.info("Preview window destroyed")
