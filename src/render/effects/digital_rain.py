"""Digital rain renderer — Matrix-style falling characters within body mask."""

import random
import string

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.utils import composite_head, get_head_mask

# Character set: mix of ASCII and katakana-like symbols
_CHARS = string.ascii_uppercase + string.digits + "{}[]|/\\<>@#$%&*"

# Column spacing in pixels
_COL_SPACING = 14
_CHAR_SCALE = 0.4
_TRAIL_LEN = 16


class _RainColumn:
    """A single column of falling characters."""

    __slots__ = ("x", "y", "speed", "chars", "head_y")

    def __init__(self, x: int, max_h: int) -> None:
        self.x = x
        self.y = random.randint(-_TRAIL_LEN * _COL_SPACING, 0)
        self.speed = random.uniform(3.0, 7.0)
        self.chars = [random.choice(_CHARS) for _ in range(_TRAIL_LEN)]
        self.head_y = self.y

    def update(self, speed_mult: float, max_h: int) -> None:
        """Advance the column downward."""
        self.y += self.speed * speed_mult
        self.head_y = self.y
        # Randomize a character occasionally
        if random.random() < 0.15:
            idx = random.randint(0, _TRAIL_LEN - 1)
            self.chars[idx] = random.choice(_CHARS)
        # Reset when fully off screen
        if self.y - _TRAIL_LEN * _COL_SPACING > max_h:
            self.y = random.randint(-_TRAIL_LEN * _COL_SPACING, -_COL_SPACING)
            self.speed = random.uniform(3.0, 7.0)


class DigitalRainRenderer(BaseRenderer):
    """Matrix-style falling green characters constrained to body mask.

    Characters stream downward in columns within the person's silhouette.
    Head characters at the front are bright white-green, trailing chars
    fade to dark green. Face region shows the real camera feed.
    """

    def __init__(self) -> None:
        self._columns: list[_RainColumn] = []
        self._initialized = False
        self._scanline_y = 0.0

    def _init_columns(self, w: int, h: int) -> None:
        """Create rain columns across the frame width."""
        self._columns = []
        for x in range(0, w, _COL_SPACING):
            self._columns.append(_RainColumn(x, h))
        self._initialized = True

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render digital rain effect.

        Args:
            ctx: Current frame data with mask and pose.

        Returns:
            Frame with Matrix rain inside body silhouette.
        """
        h, w = ctx.frame.shape[:2]
        original = ctx.frame.copy()
        output = ctx.frame.copy()

        if not self._initialized or len(self._columns) != w // _COL_SPACING:
            self._init_columns(w, h)

        if not ctx.mask or ctx.mask.num_people == 0:
            # Update columns even without mask so they keep moving
            for col in self._columns:
                col.update(1.0, h)
            return output

        bass = ctx.bass_energy
        combined = ctx.mask.combined_mask

        # Get head mask to exclude
        head = get_head_mask(ctx.pose, ctx.frame.shape) if ctx.pose else np.zeros((h, w), dtype=np.uint8)
        body_only = combined.copy()
        body_only[head > 0] = 0

        # Black out body region
        output[body_only > 0] = 0

        # Draw rain characters on a separate layer
        rain_layer = np.zeros((h, w, 3), dtype=np.uint8)

        speed_mult = 1.0 + bass * 1.5

        for col in self._columns:
            col.update(speed_mult, h)

            for i in range(_TRAIL_LEN):
                char_y = int(col.y - i * _COL_SPACING)
                if char_y < 0 or char_y >= h:
                    continue
                if col.x >= w:
                    continue

                # Only draw if inside body mask
                if body_only[char_y, min(col.x, w - 1)] == 0:
                    continue

                # Brightness: head of column is brightest
                if i == 0:
                    # Lead character: white-green, extra bright on bass
                    brightness = min(255, 200 + int(bass * 55))
                    color = (brightness // 2, brightness, brightness // 2)
                else:
                    fade = 1.0 - (i / _TRAIL_LEN)
                    g = int(200 * fade)
                    color = (0, max(g, 30), 0)

                cv2.putText(
                    rain_layer,
                    col.chars[i],
                    (col.x, char_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    _CHAR_SCALE,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        # Mask rain to body only
        rain_masked = cv2.bitwise_and(
            rain_layer, rain_layer,
            mask=body_only,
        )
        output = cv2.add(output, rain_masked)

        # Bass glitch scanline sweep
        if bass > 0.7:
            self._scanline_y += h * 0.05
            if self._scanline_y >= h:
                self._scanline_y = 0
            sy = int(self._scanline_y)
            scan_h = min(4, h - sy)
            if scan_h > 0:
                scanline_mask = body_only[sy:sy + scan_h]
                bright = np.full((scan_h, w, 3), (100, 255, 100), dtype=np.uint8)
                mask_3ch = scanline_mask[:, :, np.newaxis] > 0
                output[sy:sy + scan_h] = np.where(mask_3ch, bright, output[sy:sy + scan_h])

        return composite_head(output, original, ctx.pose, ctx.frame.shape)

    @property
    def name(self) -> str:
        return "Digital Rain"

    @property
    def needs_mask(self) -> bool:
        return True

    @property
    def needs_pose(self) -> bool:
        return True
