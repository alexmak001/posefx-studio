"""Fire skeleton renderer — warm gradient skeleton with rising particles."""

import random

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.effects import CONFIDENCE_MIN, SKELETON_CONNECTIONS

MAX_PARTICLES = 500


class _Particle:
    """A single fire particle that drifts upward and fades."""

    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size")

    def __init__(self, x: float, y: float, bass: float) -> None:
        self.x = x + random.uniform(-5, 5)
        self.y = y + random.uniform(-5, 5)
        self.vx = random.uniform(-2.5, 2.5)
        self.vy = random.uniform(-7.0, -2.0) * (1 + bass)
        self.life = 0.0
        self.max_life = random.uniform(0.3, 0.8)
        self.size = random.randint(2, 5)


class FireSkeletonRenderer(BaseRenderer):
    """Skeleton drawn in warm orange/red/yellow with rising fire particles.

    Particles emit from joints, drift upward, and fade out.
    Bass energy controls particle spawn rate, speed, and skeleton brightness.
    """

    def __init__(self) -> None:
        self._particles: list[_Particle] = []
        self._last_ts = 0.0

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render fire skeleton effect.

        Args:
            ctx: Current frame data with pose and mask.

        Returns:
            Composited frame with fire skeleton.
        """
        h, w = ctx.frame.shape[:2]
        output = ctx.frame.copy()

        # Black out person area
        if ctx.mask and ctx.mask.num_people > 0:
            output[ctx.mask.combined_mask > 0] = 0

        # Estimate dt
        dt = ctx.timestamp - self._last_ts if self._last_ts > 0 else 0.033
        dt = min(dt, 0.1)
        self._last_ts = ctx.timestamp

        bass = ctx.bass_energy
        thickness = int(5 + bass * 6)

        # Draw skeleton + spawn particles
        glow_layer = np.zeros((h, w, 3), dtype=np.uint8)

        if ctx.pose and ctx.pose.num_people > 0:
            spawn_per_joint = int(3 + bass * 12)

            for person_idx in range(ctx.pose.num_people):
                kpts = ctx.pose.keypoints[person_idx]
                confs = ctx.pose.confidences[person_idx]

                # Skeleton in hot orange-yellow, brighter with bass
                bright = min(255, int(220 + bass * 35))
                line_color = (0, int(bright * 0.5), bright)       # BGR orange
                core_color = (0, int(bright * 0.8), bright)       # brighter core

                for i, j in SKELETON_CONNECTIONS:
                    if confs[i] > CONFIDENCE_MIN and confs[j] > CONFIDENCE_MIN:
                        pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                        pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                        cv2.line(glow_layer, pt1, pt2, line_color, thickness + 4, cv2.LINE_AA)
                        cv2.line(output, pt1, pt2, core_color, thickness, cv2.LINE_AA)

                for k in range(17):
                    if confs[k] > CONFIDENCE_MIN:
                        x, y = float(kpts[k, 0]), float(kpts[k, 1])
                        cv2.circle(output, (int(x), int(y)), thickness + 2, (0, 220, 255), -1)
                        cv2.circle(glow_layer, (int(x), int(y)), thickness + 6, line_color, -1)
                        for _ in range(spawn_per_joint):
                            if len(self._particles) < MAX_PARTICLES:
                                self._particles.append(_Particle(x, y, bass))

        # Glow on skeleton
        glow_size = int(31 + bass * 20)
        if glow_size % 2 == 0:
            glow_size += 1
        glow = cv2.GaussianBlur(glow_layer, (glow_size, glow_size), 0)
        output = cv2.add(output, glow)

        # Update and draw particles
        alive: list[_Particle] = []
        for p in self._particles:
            p.life += dt
            if p.life >= p.max_life:
                continue
            p.x += p.vx
            p.y += p.vy
            alive.append(p)

            t = p.life / p.max_life
            # Yellow → orange → red → dark
            r = int(255 * max(0, 1 - t * 0.3))
            g = int(255 * max(0, 0.8 - t * 1.2))
            b_val = int(80 * max(0, 0.5 - t))
            alpha = 1.0 - t * t  # fade slower at start
            color = (int(b_val * alpha), int(g * alpha), int(r * alpha))
            radius = max(1, int(p.size * (1 - t * 0.5)))
            cv2.circle(output, (int(p.x), int(p.y)), radius, color, -1)

        self._particles = alive

        return output

    @property
    def name(self) -> str:
        return "Fire Skeleton"

    @property
    def needs_pose(self) -> bool:
        return True

    @property
    def needs_mask(self) -> bool:
        return True
