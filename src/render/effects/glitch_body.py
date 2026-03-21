"""Glitch body renderer — horizontal band distortion with RGB split."""

import random

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.utils import composite_head, get_head_mask


class GlitchBodyRenderer(BaseRenderer):
    """Body region sliced into horizontal bands with glitch distortion.

    Each strip is randomly offset horizontally with RGB channel
    separation. Some strips show noise or scanlines. Face stays clean.
    Bass energy intensifies the glitch.
    """

    _STRIP_HEIGHT = 16
    _INVERT_FRAMES = 0  # countdown for bass-triggered inversion

    def __init__(self) -> None:
        self._invert_countdown = 0

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render glitch body effect.

        Args:
            ctx: Current frame data with mask and pose.

        Returns:
            Frame with glitched body region and clean face.
        """
        h, w = ctx.frame.shape[:2]
        original = ctx.frame.copy()
        output = ctx.frame.copy()

        if not ctx.mask or ctx.mask.num_people == 0:
            return output

        bass = ctx.bass_energy
        combined = ctx.mask.combined_mask

        # Get head mask to exclude from glitch
        head = get_head_mask(ctx.pose, ctx.frame.shape) if ctx.pose else np.zeros((h, w), dtype=np.uint8)
        body_only = combined.copy()
        body_only[head > 0] = 0

        if not np.any(body_only):
            return output

        # Find body bounding box for efficiency
        rows = np.any(body_only, axis=1)
        cols = np.any(body_only, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Bass-reactive parameters
        max_offset = int(5 + bass * 25)
        channel_shift = int(2 + bass * 4)
        noise_chance = 0.1 + bass * 0.25

        # Heavy bass inversion flash
        if bass > 0.8 and self._invert_countdown <= 0:
            self._invert_countdown = 2
        if self._invert_countdown > 0:
            self._invert_countdown -= 1

        # Process body in horizontal strips
        for y in range(y_min, y_max + 1, self._STRIP_HEIGHT):
            y_end = min(y + self._STRIP_HEIGHT, y_max + 1)
            strip_mask = body_only[y:y_end, x_min:x_max + 1]

            if not np.any(strip_mask):
                continue

            # Random horizontal offset
            offset = random.randint(-max_offset, max_offset)
            strip = original[y:y_end, x_min:x_max + 1].copy()

            # RGB channel separation
            if channel_shift > 0:
                shifted = np.zeros_like(strip)
                # Red channel shifts right
                if offset + channel_shift < strip.shape[1]:
                    shifted[:, channel_shift:, 2] = strip[:, :strip.shape[1] - channel_shift, 2]
                else:
                    shifted[:, :, 2] = strip[:, :, 2]
                # Green stays centered
                shifted[:, :, 1] = strip[:, :, 1]
                # Blue channel shifts left
                if channel_shift < strip.shape[1]:
                    shifted[:, :strip.shape[1] - channel_shift, 0] = strip[:, channel_shift:, 0]
                else:
                    shifted[:, :, 0] = strip[:, :, 0]
                strip = shifted

            # Noise replacement for some strips
            if random.random() < noise_chance:
                noise = np.random.randint(0, 80, strip.shape, dtype=np.uint8)
                # Tint the noise cyan/magenta
                if random.random() > 0.5:
                    noise[:, :, 2] = np.clip(noise[:, :, 2] + 60, 0, 255)
                else:
                    noise[:, :, 1] = np.clip(noise[:, :, 1] + 60, 0, 255)
                strip = noise

            # Apply horizontal offset
            sw = strip.shape[1]
            if abs(offset) < sw:
                shifted_strip = np.zeros_like(strip)
                if offset > 0:
                    shifted_strip[:, offset:] = strip[:, :sw - offset]
                elif offset < 0:
                    shifted_strip[:, :sw + offset] = strip[:, -offset:]
                else:
                    shifted_strip = strip
                strip = shifted_strip

            # Bass inversion
            if self._invert_countdown > 0:
                strip = 255 - strip

            # Apply strip only where body mask is active
            mask_3ch = strip_mask[:, :, np.newaxis] > 0
            region = output[y:y_end, x_min:x_max + 1]
            output[y:y_end, x_min:x_max + 1] = np.where(mask_3ch, strip, region)

        # Scanline overlay on body (every 3rd row dimmed)
        scanline_mask = np.zeros(h, dtype=bool)
        scanline_mask[::3] = True
        scan_rows = scanline_mask & (np.any(body_only, axis=1))
        for y_idx in np.where(scan_rows)[0]:
            row_mask = body_only[y_idx] > 0
            output[y_idx, row_mask] = (output[y_idx, row_mask].astype(np.int16) * 7 // 10).astype(np.uint8)

        return composite_head(output, original, ctx.pose, ctx.frame.shape)

    @property
    def name(self) -> str:
        return "Glitch Body"

    @property
    def needs_mask(self) -> bool:
        return True

    @property
    def needs_pose(self) -> bool:
        return True
