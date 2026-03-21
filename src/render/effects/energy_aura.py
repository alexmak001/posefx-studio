"""Energy aura renderer — pulsing glow radiating from body outline."""

import random

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.utils import composite_head


class _AuraParticle:
    """A small dot that drifts upward from the aura edge."""

    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-2.5, -0.8)
        self.max_life = random.randint(20, 30)
        self.life = self.max_life
        self.color = random.choice([
            (0, 200, 255),    # gold (BGR)
            (0, 140, 255),    # orange
            (180, 50, 255),   # magenta
            (100, 220, 255),  # amber
        ])

    def update(self) -> bool:
        """Advance one frame. Returns False when dead."""
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life > 0


class EnergyAuraRenderer(BaseRenderer):
    """Pulsing energy field radiating from body outline.

    Uses distance transform for fast concentric glow rings around
    the body silhouette. Small particles drift upward along the aura.
    """

    def __init__(self) -> None:
        self._particles: list[_AuraParticle] = []

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render energy aura effect.

        Args:
            ctx: Current frame data with mask.

        Returns:
            Frame with energy aura around body silhouette.
        """
        h, w = ctx.frame.shape[:2]
        original = ctx.frame.copy()
        output = ctx.frame.copy()

        if not ctx.mask or ctx.mask.num_people == 0:
            self._draw_particles(output)
            return output

        bass = ctx.bass_energy
        combined = ctx.mask.combined_mask

        # Warm color grade inside body
        body_region = combined > 0
        output[body_region, 2] = np.clip(
            output[body_region, 2].astype(np.int16) + 15, 0, 255
        ).astype(np.uint8)
        output[body_region, 0] = np.clip(
            output[body_region, 0].astype(np.int16) - 10, 0, 255
        ).astype(np.uint8)

        # Distance transform from mask edge (fast, single pass)
        inverted = 1 - combined
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)

        # Build aura using distance bands
        max_dist = 40.0 * (1.0 + bass)
        aura_mask = (dist > 0) & (dist < max_dist)

        if np.any(aura_mask):
            # Normalize distance to 0-1 within aura band
            norm_dist = np.clip(dist / max_dist, 0, 1)

            # Color gradient: gold (close) → orange → magenta → purple (far)
            # Use distance to interpolate colors
            aura_layer = np.zeros((h, w, 3), dtype=np.float32)
            t = norm_dist  # 0 = near body, 1 = far

            # BGR color ramp: gold(0,220,255) → orange(0,140,255) → magenta(180,50,255) → purple(180,50,200)
            aura_layer[:, :, 0] = 180 * t              # B: 0 → 180
            aura_layer[:, :, 1] = 220 - 170 * t        # G: 220 → 50
            aura_layer[:, :, 2] = 255 - 55 * t         # R: 255 → 200

            # Bass white-hot shift
            if bass > 0.5:
                white_t = (bass - 0.5) * 2 * 0.3
                aura_layer = aura_layer * (1 - white_t) + 255 * white_t

            # Fade by distance (close = bright, far = faint)
            fade = (1.0 - norm_dist) ** 1.5
            aura_layer *= fade[:, :, np.newaxis]

            # Zero out non-aura pixels
            aura_layer[~aura_mask] = 0

            # Blur for soft glow
            blur_size = int(21 + bass * 10)
            if blur_size % 2 == 0:
                blur_size += 1
            aura_u8 = np.clip(aura_layer, 0, 255).astype(np.uint8)
            aura_glow = cv2.GaussianBlur(aura_u8, (blur_size, blur_size), 0)
            output = cv2.add(output, aura_glow)

        # Spawn particles along contours
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        spawn_count = int(3 + bass * 8)
        for contour in contours:
            if len(contour) < 5:
                continue
            for _ in range(spawn_count):
                idx = random.randint(0, len(contour) - 1)
                px, py = contour[idx][0]
                self._particles.append(_AuraParticle(float(px), float(py)))

        if len(self._particles) > 150:
            self._particles = self._particles[-150:]

        self._draw_particles(output)

        return composite_head(output, original, ctx.pose, ctx.frame.shape)

    def _draw_particles(self, frame: np.ndarray) -> None:
        """Update and draw all particles onto the frame."""
        alive = []
        for p in self._particles:
            if p.update():
                alpha = p.life / p.max_life
                color = tuple(int(c * alpha) for c in p.color)
                cv2.circle(frame, (int(p.x), int(p.y)), 2, color, -1)
                alive.append(p)
        self._particles = alive

    @property
    def name(self) -> str:
        return "Energy Aura"

    @property
    def needs_mask(self) -> bool:
        return True

    @property
    def needs_pose(self) -> bool:
        return True
