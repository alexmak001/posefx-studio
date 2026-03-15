"""FPS tracking utilities."""

import time
from collections import deque


class FPSCounter:
    """Tracks rolling average FPS over the last N frames.

    Args:
        window_size: Number of frames to average over.
    """

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> None:
        """Record a frame timestamp. Call once per frame."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """Current rolling average FPS."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
