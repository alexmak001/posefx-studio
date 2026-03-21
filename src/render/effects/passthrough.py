"""Passthrough renderer — returns the original camera frame unmodified."""

import numpy as np

from src.render.base import BaseRenderer, RenderContext


class PassthroughRenderer(BaseRenderer):
    """Camera-only mode. No effects applied."""

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Return the original frame unmodified.

        Args:
            ctx: Current frame data.

        Returns:
            Original camera frame.
        """
        return ctx.frame.copy()

    @property
    def name(self) -> str:
        return "Passthrough"
