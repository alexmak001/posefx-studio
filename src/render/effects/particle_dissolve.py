"""Particle dissolve renderer — body filled with scattering colored dots."""

import random

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.utils import composite_head, get_head_mask

# Warm particle palette (BGR)
_COLORS = [
    (100, 120, 255),  # coral
    (50, 190, 255),   # amber
    (150, 100, 255),  # pink
    (0, 200, 230),    # gold
    (130, 180, 255),  # peach
]

_MAX_PARTICLES = 2500
_SPAWN_BATCH = 80


class _Particle:
    """A single colored dot with position and velocity."""

    __slots__ = ("x", "y", "vx", "vy", "color", "life")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-0.5, 0.5)
        self.color = random.choice(_COLORS)
        self.life = random.randint(40, 80)

    def update(self) -> bool:
        """Advance one frame. Returns False when dead."""
        self.x += self.vx
        self.y += self.vy
        # Dampen velocity
        self.vx *= 0.95
        self.vy *= 0.95
        self.life -= 1
        return self.life > 0


class ParticleDissolveRenderer(BaseRenderer):
    """Body silhouette filled with thousands of colored dots.

    Particles inside the mask are attracted to stay within. When the body
    moves, edge particles scatter outward. Bass hits cause radial bursts.
    """

    def __init__(self) -> None:
        self._particles: list[_Particle] = []
        self._prev_mask: np.ndarray | None = None

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render particle dissolve effect.

        Args:
            ctx: Current frame data with mask.

        Returns:
            Frame with particle-filled body silhouette.
        """
        h, w = ctx.frame.shape[:2]
        original = ctx.frame.copy()
        output = ctx.frame.copy()

        if not ctx.mask or ctx.mask.num_people == 0:
            # Draw drifting particles and fade them
            self._update_and_draw(output, None, 0.0, h, w)
            self._prev_mask = None
            return output

        bass = ctx.bass_energy
        combined = ctx.mask.combined_mask

        # Get head mask
        head = get_head_mask(ctx.pose, ctx.frame.shape) if ctx.pose else np.zeros((h, w), dtype=np.uint8)
        body_only = combined.copy()
        body_only[head > 0] = 0

        # Black out body region
        output[body_only > 0] = 0

        # Detect mask edge movement for scatter kicks
        edge_kick = False
        if self._prev_mask is not None and self._prev_mask.shape == body_only.shape:
            diff = cv2.absdiff(body_only, self._prev_mask)
            if np.sum(diff) > 500:
                edge_kick = True
        self._prev_mask = body_only.copy()

        # Spawn particles inside mask
        mask_points = np.argwhere(body_only > 0)  # (N, 2) as (y, x)
        if len(mask_points) > 0 and len(self._particles) < _MAX_PARTICLES:
            count = min(_SPAWN_BATCH, _MAX_PARTICLES - len(self._particles))
            indices = np.random.randint(0, len(mask_points), count)
            for idx in indices:
                py, px = mask_points[idx]
                self._particles.append(_Particle(float(px), float(py)))

        # Apply forces to particles
        # Find body center for bass radial kick
        if len(mask_points) > 0:
            center_y = float(np.mean(mask_points[:, 0]))
            center_x = float(np.mean(mask_points[:, 1]))
        else:
            center_x, center_y = w / 2, h / 2

        for p in self._particles:
            ix, iy = int(p.x), int(p.y)
            inside = (0 <= iy < h and 0 <= ix < w and body_only[iy, ix] > 0)

            if inside:
                # Slight random jitter to keep them alive
                p.vx += random.uniform(-0.3, 0.3)
                p.vy += random.uniform(-0.3, 0.3)
            else:
                # Spring force back toward mask if nearby
                # Find nearest direction toward center
                dx = center_x - p.x
                dy = center_y - p.y
                dist = max(1.0, (dx ** 2 + dy ** 2) ** 0.5)
                p.vx += dx / dist * 0.8
                p.vy += dy / dist * 0.8

            # Edge scatter kick
            if edge_kick and inside:
                dx = p.x - center_x
                dy = p.y - center_y
                dist = max(1.0, (dx ** 2 + dy ** 2) ** 0.5)
                p.vx += dx / dist * 3.0
                p.vy += dy / dist * 3.0

            # Bass radial kick
            if bass > 0.3:
                dx = p.x - center_x
                dy = p.y - center_y
                dist = max(1.0, (dx ** 2 + dy ** 2) ** 0.5)
                kick = bass * 2.0
                if bass > 0.8:
                    kick = bass * 5.0
                p.vx += dx / dist * kick * 0.3
                p.vy += dy / dist * kick * 0.3

        self._update_and_draw(output, body_only, bass, h, w)

        return composite_head(output, original, ctx.pose, ctx.frame.shape)

    def _update_and_draw(self, frame: np.ndarray, mask: np.ndarray | None,
                         bass: float, h: int, w: int) -> None:
        """Update particle positions and draw them."""
        alive = []
        for p in self._particles:
            if not p.update():
                continue
            ix, iy = int(p.x), int(p.y)
            if 0 <= iy < h and 0 <= ix < w:
                alpha = min(1.0, p.life / 40.0)
                color = tuple(int(c * alpha) for c in p.color)
                cv2.circle(frame, (ix, iy), 2, color, -1)
                alive.append(p)
            elif p.life > 5:
                # Keep off-screen particles alive briefly
                alive.append(p)
        self._particles = alive

    @property
    def name(self) -> str:
        return "Particle Dissolve"

    @property
    def needs_mask(self) -> bool:
        return True

    @property
    def needs_pose(self) -> bool:
        return True
