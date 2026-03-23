"""Glitch body renderer — horizontal band distortion with RGB split."""

import random

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.utils import get_head_mask


class GlitchBodyRenderer(BaseRenderer):
    """Body region sliced into horizontal bands with glitch distortion.

    Each strip is randomly offset horizontally with RGB channel
    separation. Entire body including face is glitched, but the face
    area is blended at higher transparency so you can still see it.
    Bass energy intensifies the glitch.
    """

    _STRIP_HEIGHT = 16

    def __init__(self) -> None:
        self._invert_countdown = 0

    def render(self, ctx: RenderContext) -> np.ndarray:
        h, w = ctx.frame.shape[:2]
        original = ctx.frame.copy()
        output = ctx.frame.copy()

        if not ctx.mask or ctx.mask.num_people == 0:
            return output

        bass = ctx.bass_energy
        combined = ctx.mask.combined_mask

        # Use full body mask (including face)
        if not np.any(combined):
            return output

        # Get head mask for transparency blending later
        head = get_head_mask(ctx.pose, ctx.frame.shape) if ctx.pose else np.zeros((h, w), dtype=np.uint8)

        # Find body bounding box
        rows = np.any(combined, axis=1)
        cols = np.any(combined, axis=0)
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

        # Build glitched version of entire body
        glitched = output.copy()

        for y in range(y_min, y_max + 1, self._STRIP_HEIGHT):
            y_end = min(y + self._STRIP_HEIGHT, y_max + 1)
            strip_mask = combined[y:y_end, x_min:x_max + 1]

            if not np.any(strip_mask):
                continue

            offset = random.randint(-max_offset, max_offset)
            strip = original[y:y_end, x_min:x_max + 1].copy()

            # RGB channel separation
            if channel_shift > 0:
                shifted = np.zeros_like(strip)
                if offset + channel_shift < strip.shape[1]:
                    shifted[:, channel_shift:, 2] = strip[:, :strip.shape[1] - channel_shift, 2]
                else:
                    shifted[:, :, 2] = strip[:, :, 2]
                shifted[:, :, 1] = strip[:, :, 1]
                if channel_shift < strip.shape[1]:
                    shifted[:, :strip.shape[1] - channel_shift, 0] = strip[:, channel_shift:, 0]
                else:
                    shifted[:, :, 0] = strip[:, :, 0]
                strip = shifted

            # Noise replacement for some strips
            if random.random() < noise_chance:
                noise = np.random.randint(0, 80, strip.shape, dtype=np.uint8)
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

            # Apply strip where body mask is active
            mask_3ch = strip_mask[:, :, np.newaxis] > 0
            region = glitched[y:y_end, x_min:x_max + 1]
            glitched[y:y_end, x_min:x_max + 1] = np.where(mask_3ch, strip, region)

        # Scanline overlay on body (every 3rd row dimmed)
        scanline_mask = np.zeros(h, dtype=bool)
        scanline_mask[::3] = True
        scan_rows = scanline_mask & (np.any(combined, axis=1))
        for y_idx in np.where(scan_rows)[0]:
            row_mask = combined[y_idx] > 0
            glitched[y_idx, row_mask] = (glitched[y_idx, row_mask].astype(np.int16) * 7 // 10).astype(np.uint8)

        # For the face region: blend glitch at ~40% so face is still visible
        if np.any(head):
            head_bool = head > 0
            face_blend = cv2.addWeighted(
                glitched, 0.4, original, 0.6, 0
            )
            glitched[head_bool] = face_blend[head_bool]

        return glitched

    @property
    def name(self) -> str:
        return "Glitch Body"

    @property
    def needs_mask(self) -> bool:
        return True

    @property
    def needs_pose(self) -> bool:
        return True
