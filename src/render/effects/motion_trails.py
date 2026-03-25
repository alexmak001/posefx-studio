"""Motion trails renderer — rainbow afterimages trailing behind movement."""

from collections import deque

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext


class MotionTrailsRenderer(BaseRenderer):
    """Rainbow afterimages that trail behind movement.

    Stores recent frames of the person's mask. Each historical mask
    is drawn as a semi-transparent colored overlay with hue cycling
    through the spectrum. Covers the full body including face.
    """

    _MIN_TRAIL = 8
    _MAX_TRAIL = 14

    def __init__(self) -> None:
        self._history: deque[np.ndarray] = deque(maxlen=self._MAX_TRAIL)
        self._hue_offset = 0.0

    def render(self, ctx: RenderContext) -> np.ndarray:
        h, w = ctx.frame.shape[:2]
        output = ctx.frame.copy()

        bass = ctx.bass_energy
        trail_len = int(self._MIN_TRAIL + bass * (self._MAX_TRAIL - self._MIN_TRAIL))

        # Store current mask (full body including face)
        if ctx.mask and ctx.mask.num_people > 0:
            self._history.append(ctx.mask.combined_mask.copy())
        else:
            self._history.append(np.zeros((h, w), dtype=np.uint8))

        self._hue_offset = (self._hue_offset + 8) % 180

        num_trails = min(len(self._history), trail_len)
        if num_trails < 2:
            return output

        trail_layer = np.zeros((h, w, 3), dtype=np.float32)

        for i, mask in enumerate(list(self._history)[-num_trails:]):
            if not np.any(mask):
                continue

            progress = i / max(num_trails - 1, 1)

            hue = int((self._hue_offset + progress * 150) % 180)
            saturation = int(180 + bass * 75)
            value = 255

            hsv_pixel = np.array([[[hue, saturation, value]]], dtype=np.uint8)
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
            color = bgr_pixel[0, 0].astype(np.float32)

            # More transparent: lower alpha range
            alpha = 0.05 + progress * (0.2 + bass * 0.1)

            mask_f = mask.astype(np.float32)[:, :, np.newaxis]
            trail_layer += mask_f * color * alpha

        trail_clipped = np.clip(trail_layer, 0, 255).astype(np.uint8)
        # More transparent blend (0.5 instead of 0.7)
        output = cv2.addWeighted(output, 1.0, trail_clipped, 0.5, 0)

        # Draw current body on top (original pixels where current mask is)
        if ctx.mask and ctx.mask.num_people > 0:
            current = ctx.mask.combined_mask > 0
            output[current] = ctx.frame[current]

        # No composite_head — trails cover the face too
        return output

    @property
    def name(self) -> str:
        return "Motion Trails"

    @property
    def needs_mask(self) -> bool:
        return True

    @property
    def needs_pose(self) -> bool:
        return True
