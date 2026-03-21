"""Particle fill renderer — person silhouette filled with swirling particles."""

import random
import time

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext

MAX_PARTICLES = 2500


class _FillParticle:
    """A single swirling particle inside the mask."""

    __slots__ = ("x", "y", "vx", "vy", "hue", "size")

    def __init__(self, x: float, y: float, hue: int) -> None:
        self.x = x
        self.y = y
        self.vx = random.uniform(-2.0, 2.0)
        self.vy = random.uniform(-2.0, 2.0)
        self.hue = hue
        self.size = random.randint(2, 4)


class ParticleFillRenderer(BaseRenderer):
    """Person mask filled with swirling colored particle dots.

    Points are scattered inside the mask region and drift each frame.
    Bass energy controls particle speed, density, and brightness.
    Background is fully blacked out for contrast.
    """

    def __init__(self) -> None:
        self._particles: list[_FillParticle] = []
        # Precompute a color LUT to avoid per-particle HSV conversion
        self._color_lut = np.zeros((180, 3), dtype=np.uint8)
        for h in range(180):
            hsv = np.array([[[h, 255, 255]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self._color_lut[h] = bgr[0, 0]

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render particle fill effect.

        Args:
            ctx: Current frame data with mask.

        Returns:
            Composited frame with particle-filled silhouette.
        """
        h, w = ctx.frame.shape[:2]

        if not ctx.mask or ctx.mask.num_people == 0:
            self._particles.clear()
            return ctx.frame.copy()

        mask = ctx.mask.combined_mask
        bass = ctx.bass_energy

        # Full black background for max contrast
        output = np.zeros((h, w, 3), dtype=np.uint8)

        # Spawn particles
        target_count = int(1200 + bass * 1300)
        spawn_needed = min(target_count - len(self._particles), 150)
        if spawn_needed > 0:
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                # Shift hue over time for rainbow cycling
                hue_offset = int(time.monotonic() * 30) % 180
                indices = np.random.randint(0, len(ys), size=spawn_needed)
                for idx in indices:
                    hue_val = (hue_offset + random.randint(-30, 30)) % 180
                    self._particles.append(
                        _FillParticle(float(xs[idx]), float(ys[idx]), hue_val)
                    )

        # Update and draw particles
        speed_mult = 1.0 + bass * 4.0
        alive: list[_FillParticle] = []

        for p in self._particles:
            p.x += p.vx * speed_mult + random.uniform(-0.5, 0.5)
            p.y += p.vy * speed_mult + random.uniform(-0.5, 0.5)

            ix, iy = int(p.x), int(p.y)
            if 0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0:
                alive.append(p)
                color = self._color_lut[p.hue]
                cv2.circle(output, (ix, iy), p.size, (int(color[0]), int(color[1]), int(color[2])), -1)

        self._particles = alive

        # Soft glow pass over all particles for extra pop
        if len(alive) > 0:
            glow = cv2.GaussianBlur(output, (9, 9), 0)
            output = cv2.add(output, glow)

        return output

    @property
    def name(self) -> str:
        return "Particle Fill"

    @property
    def needs_mask(self) -> bool:
        return True
