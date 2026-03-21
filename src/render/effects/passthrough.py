"""Passthrough renderer — clean camera feed with optional vignette."""

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext


class PassthroughRenderer(BaseRenderer):
    """Returns the original camera frame with a subtle bass-reactive vignette."""

    def __init__(self) -> None:
        self._vignette_cache: dict[tuple[int, int], np.ndarray] = {}

    def _get_vignette(self, h: int, w: int) -> np.ndarray:
        """Get or create a normalized vignette gradient for the given size."""
        key = (h, w)
        if key not in self._vignette_cache:
            y = np.linspace(-1, 1, h)
            x = np.linspace(-1, 1, w)
            xv, yv = np.meshgrid(x, y)
            # Distance from center, normalized 0-1
            dist = np.sqrt(xv ** 2 + yv ** 2)
            dist = np.clip(dist / dist.max(), 0, 1)
            self._vignette_cache[key] = dist.astype(np.float32)
        return self._vignette_cache[key]

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render passthrough with optional vignette.

        Args:
            ctx: Current frame data.

        Returns:
            Original frame, optionally with vignette darkening.
        """
        output = ctx.frame.copy()

        if ctx.bass_energy > 0.01:
            h, w = output.shape[:2]
            dist = self._get_vignette(h, w)
            # Corner darkness scales with bass: 0.2 (quiet) to 0.35 (loud)
            strength = 0.2 + ctx.bass_energy * 0.15
            darkening = 1.0 - dist * strength
            output = (output * darkening[:, :, np.newaxis]).astype(np.uint8)

        return output

    @property
    def name(self) -> str:
        return "Passthrough"
