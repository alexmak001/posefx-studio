"""Particle dissolve renderer — rainbow glowing particles from body edges."""

import random
import time

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.utils import composite_head

_MAX_PARTICLES = 3000
_BASE_SPAWN = 60


def _hue_to_bgr(hue: float) -> tuple[int, int, int]:
    hsv = np.uint8([[[int(hue) % 180, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class _Particle:
    __slots__ = ("x", "y", "vx", "vy", "color", "life", "size")

    def __init__(self, x: float, y: float, hue: float) -> None:
        self.x = x
        self.y = y
        self.vx = random.uniform(-1.0, 1.0)
        self.vy = random.uniform(-1.0, 1.0)
        self.color = _hue_to_bgr(hue)
        self.life = random.randint(30, 70)
        self.size = random.randint(2, 4)

    def update(self) -> bool:
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.96
        self.vy *= 0.96
        self.life -= 1
        return self.life > 0


class ParticleDissolveRenderer(BaseRenderer):
    """Rainbow glowing particles launched from body edges.

    Faster movement spawns more particles. Particles cycle through
    the rainbow and have a glow effect.
    """

    def __init__(self) -> None:
        self._particles: list[_Particle] = []
        self._prev_mask: np.ndarray | None = None
        self._prev_center: tuple[float, float] | None = None

    def render(self, ctx: RenderContext) -> np.ndarray:
        h, w = ctx.frame.shape[:2]
        original = ctx.frame.copy()
        output = ctx.frame.copy()

        if not ctx.mask or ctx.mask.num_people == 0:
            self._draw_particles(output, h, w)
            self._prev_mask = None
            self._prev_center = None
            return output

        bass = ctx.bass_energy
        combined = ctx.mask.combined_mask
        t = time.monotonic()

        # Find body center
        body_coords = np.argwhere(combined > 0)
        if len(body_coords) > 0:
            center_y = float(np.mean(body_coords[:, 0]))
            center_x = float(np.mean(body_coords[:, 1]))
        else:
            center_x, center_y = w / 2, h / 2

        # Detect movement speed from center shift
        movement_speed = 0.0
        if self._prev_center is not None:
            dx = center_x - self._prev_center[0]
            dy = center_y - self._prev_center[1]
            movement_speed = min((dx ** 2 + dy ** 2) ** 0.5 / 10.0, 5.0)
        self._prev_center = (center_x, center_y)

        # Detect mask edge changes
        edge_kick = False
        if self._prev_mask is not None and self._prev_mask.shape == combined.shape:
            diff = cv2.absdiff(combined, self._prev_mask)
            if np.sum(diff) > 500:
                edge_kick = True
        self._prev_mask = combined.copy()

        # Spawn count scales with movement + bass
        spawn_count = int(_BASE_SPAWN * (1.0 + movement_speed * 2.0 + bass * 3.0))
        spawn_count = min(spawn_count, _MAX_PARTICLES - len(self._particles))

        # Spawn along contours with rainbow hue
        if spawn_count > 0:
            contours, _ = cv2.findContours(
                combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            base_hue = (t * 30) % 180  # cycle over time
            if contours:
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    per_contour = max(1, spawn_count // len(contours))
                    for j in range(per_contour):
                        if len(self._particles) >= _MAX_PARTICLES:
                            break
                        idx = random.randint(0, len(contour) - 1)
                        px, py = contour[idx][0]
                        hue = (base_hue + j * 5 + random.uniform(-10, 10)) % 180
                        self._particles.append(_Particle(float(px), float(py), hue))

        # Apply forces
        for p in self._particles:
            ix, iy = int(p.x), int(p.y)
            inside = (0 <= iy < h and 0 <= ix < w and combined[iy, ix] > 0)

            if inside:
                p.vx += random.uniform(-0.4, 0.4)
                p.vy += random.uniform(-0.4, 0.4)
            else:
                # Drift outward
                dx = p.x - center_x
                dy = p.y - center_y
                dist = max(1.0, (dx ** 2 + dy ** 2) ** 0.5)
                p.vx += dx / dist * 0.3
                p.vy += dy / dist * 0.3

            if edge_kick and inside:
                dx = p.x - center_x
                dy = p.y - center_y
                dist = max(1.0, (dx ** 2 + dy ** 2) ** 0.5)
                p.vx += dx / dist * 3.0
                p.vy += dy / dist * 3.0

            # Bass burst
            if bass > 0.3:
                dx = p.x - center_x
                dy = p.y - center_y
                dist = max(1.0, (dx ** 2 + dy ** 2) ** 0.5)
                kick = bass * 2.5 if bass <= 0.8 else bass * 5.0
                p.vx += dx / dist * kick * 0.3
                p.vy += dy / dist * kick * 0.3

            # Movement burst — faster body motion = more scatter
            if movement_speed > 1.0:
                p.vx += random.uniform(-1, 1) * movement_speed * 0.3
                p.vy += random.uniform(-1, 1) * movement_speed * 0.3

        self._draw_particles(output, h, w)

        return composite_head(output, original, ctx.pose, ctx.frame.shape)

    def _draw_particles(self, frame: np.ndarray, h: int, w: int) -> None:
        """Draw particles with glow effect."""
        # Draw on a separate layer for glow
        glow_layer = np.zeros_like(frame)

        alive = []
        for p in self._particles:
            if not p.update():
                continue
            ix, iy = int(p.x), int(p.y)
            if 0 <= iy < h and 0 <= ix < w:
                alpha = min(1.0, p.life / 30.0)
                color = tuple(int(c * alpha) for c in p.color)
                cv2.circle(glow_layer, (ix, iy), p.size, color, -1)
                alive.append(p)
            elif p.life > 5:
                alive.append(p)
        self._particles = alive

        # Glow pass
        if np.any(glow_layer):
            blurred = cv2.GaussianBlur(glow_layer, (11, 11), 0)
            frame[:] = cv2.add(frame, blurred)
            frame[:] = cv2.add(frame, glow_layer)

    @property
    def name(self) -> str:
        return "Particle Dissolve"

    @property
    def needs_mask(self) -> bool:
        return True

    @property
    def needs_pose(self) -> bool:
        return True
