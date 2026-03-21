"""Motion trails renderer — rainbow afterimages trailing behind movement."""

from collections import deque

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.utils import composite_head


class MotionTrailsRenderer(BaseRenderer):
    """Rainbow afterimages that trail behind movement.

    Stores recent frames of the person's mask. Each historical mask
    is drawn as a semi-transparent colored overlay with hue cycling
    through the spectrum. Fast movement creates long colorful streaks.
    """

    _MIN_TRAIL = 8
    _MAX_TRAIL = 14

    def __init__(self) -> None:
        self._history: deque[np.ndarray] = deque(maxlen=self._MAX_TRAIL)
        self._hue_offset = 0.0

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render motion trail effect.

        Args:
            ctx: Current frame data with mask.

        Returns:
            Frame with rainbow trails behind the body.
        """
        h, w = ctx.frame.shape[:2]
        original = ctx.frame.copy()
        output = ctx.frame.copy()

        bass = ctx.bass_energy

        # Determine trail length based on bass
        trail_len = int(self._MIN_TRAIL + bass * (self._MAX_TRAIL - self._MIN_TRAIL))

        # Store current mask
        if ctx.mask and ctx.mask.num_people > 0:
            self._history.append(ctx.mask.combined_mask.copy())
        else:
            self._history.append(np.zeros((h, w), dtype=np.uint8))

        # Advance hue
        self._hue_offset = (self._hue_offset + 8) % 180

        # Draw trails from oldest to newest
        num_trails = min(len(self._history), trail_len)
        if num_trails < 2:
            return composite_head(output, original, ctx.pose, ctx.frame.shape)

        trail_layer = np.zeros((h, w, 3), dtype=np.float32)

        for i, mask in enumerate(list(self._history)[-num_trails:]):
            if not np.any(mask):
                continue

            # Progress: 0 (oldest) to 1 (newest)
            progress = i / max(num_trails - 1, 1)

            # Hue cycles through spectrum: red → orange → yellow → green → cyan → blue → violet
            hue = int((self._hue_offset + progress * 150) % 180)
            saturation = int(180 + bass * 75)  # more saturated with bass
            value = 255

            # Create HSV color, convert to BGR
            hsv_pixel = np.array([[[hue, saturation, value]]], dtype=np.uint8)
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
            color = bgr_pixel[0, 0].astype(np.float32)

            # Opacity: oldest faint, newest strong. Bass boosts all.
            alpha = 0.1 + progress * (0.35 + bass * 0.15)

            # Tint the trail region
            mask_f = mask.astype(np.float32)[:, :, np.newaxis]
            trail_layer += mask_f * color * alpha

        # Blend trail onto output
        trail_clipped = np.clip(trail_layer, 0, 255).astype(np.uint8)
        output = cv2.addWeighted(output, 1.0, trail_clipped, 0.7, 0)

        # Draw current body normally on top (original pixels where current mask is)
        if ctx.mask and ctx.mask.num_people > 0:
            current = ctx.mask.combined_mask > 0
            output[current] = original[current]

        return composite_head(output, original, ctx.pose, ctx.frame.shape)

    @property
    def name(self) -> str:
        return "Motion Trails"

    @property
    def needs_mask(self) -> bool:
        return True

    @property
    def needs_pose(self) -> bool:
        return True
