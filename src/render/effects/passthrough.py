"""Passthrough renderer — brightened camera feed for dark room use."""

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext


class PassthroughRenderer(BaseRenderer):
    """Returns the camera frame with brightness boost for dark rooms.

    Applies a gentle brightness lift and optional bass-reactive
    warm glow to help visibility in low-light party environments.
    """

    def render(self, ctx: RenderContext) -> np.ndarray:
        output = ctx.frame.copy()

        # Brighten the frame for dark room visibility
        # Lift shadows: add flat brightness + gentle gamma correction
        bright_add = 10  # flat brightness boost
        output = cv2.add(output, np.full_like(output, bright_add))

        # Slight gamma lift to brighten midtones (gamma < 1 = brighter)
        gamma = 0.93
        lut = np.array([
            np.clip(((i / 255.0) ** gamma) * 255, 0, 255)
            for i in range(256)
        ], dtype=np.uint8)
        output = cv2.LUT(output, lut)

        # Bass-reactive warm pulse (subtle)
        if ctx.bass_energy > 0.1:
            warmth = int(ctx.bass_energy * 20)
            warm = output.copy()
            warm[:, :, 2] = np.clip(warm[:, :, 2].astype(np.int16) + warmth, 0, 255).astype(np.uint8)
            warm[:, :, 1] = np.clip(warm[:, :, 1].astype(np.int16) + warmth // 2, 0, 255).astype(np.uint8)
            output = warm

        return output

    @property
    def name(self) -> str:
        return "Passthrough"
