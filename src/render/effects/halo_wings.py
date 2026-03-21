"""Halo wings renderer — glowing halo and translucent energy wings."""

import math
import random

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.effects import CONFIDENCE_MIN


class _WingParticle:
    """A glowing dot shed from wingtips."""

    __slots__ = ("x", "y", "vy", "life", "max_life")

    def __init__(self, x: float, y: float) -> None:
        self.x = x + random.uniform(-3, 3)
        self.y = y
        self.vy = random.uniform(1.0, 3.0)
        self.max_life = random.randint(15, 25)
        self.life = self.max_life

    def update(self) -> bool:
        """Advance one frame. Returns False when dead."""
        self.y += self.vy
        self.life -= 1
        return self.life > 0


class HaloWingsRenderer(BaseRenderer):
    """Glowing halo ring above head and translucent energy wings from shoulders.

    Body and face pass through normally. Halo and wings are additive
    overlays, not replacements.
    """

    def __init__(self) -> None:
        self._particles: list[_WingParticle] = []

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render halo and wings effect.

        Args:
            ctx: Current frame data with pose.

        Returns:
            Frame with halo and wing overlays.
        """
        h, w = ctx.frame.shape[:2]
        output = ctx.frame.copy()

        if not ctx.pose or ctx.pose.num_people == 0:
            self._draw_particles(output)
            return output

        bass = ctx.bass_energy
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        for person_idx in range(ctx.pose.num_people):
            kpts = ctx.pose.keypoints[person_idx]
            confs = ctx.pose.confidences[person_idx]

            self._draw_halo(overlay, kpts, confs, bass, h, w)
            self._draw_wings(overlay, kpts, confs, bass, h, w)

        # Blur overlay for glow
        glow_size = int(15 + bass * 10)
        if glow_size % 2 == 0:
            glow_size += 1
        glow = cv2.GaussianBlur(overlay, (glow_size, glow_size), 0)

        # Additive blend
        output = cv2.add(output, glow)
        output = cv2.add(output, overlay)

        # Draw particles on top
        self._draw_particles(output)

        return output

    def _draw_halo(self, layer: np.ndarray, kpts: np.ndarray,
                   confs: np.ndarray, bass: float, h: int, w: int) -> None:
        """Draw a glowing elliptical halo above the head."""
        # Use nose (0) as primary, eyes as backup
        if confs[0] > CONFIDENCE_MIN:
            head_x, head_y = float(kpts[0, 0]), float(kpts[0, 1])
        elif confs[1] > CONFIDENCE_MIN and confs[2] > CONFIDENCE_MIN:
            head_x = float((kpts[1, 0] + kpts[2, 0]) / 2)
            head_y = float((kpts[1, 1] + kpts[2, 1]) / 2)
        else:
            return

        # Estimate head radius from eye/ear spread
        head_pts = []
        for idx in [0, 1, 2, 3, 4]:
            if confs[idx] > CONFIDENCE_MIN:
                head_pts.append(kpts[idx])
        if len(head_pts) >= 2:
            pts = np.array(head_pts)
            head_radius = float(np.max(np.sqrt(
                (pts[:, 0] - head_x) ** 2 + (pts[:, 1] - head_y) ** 2
            ))) * 1.2
        else:
            head_radius = 40.0

        head_radius = max(head_radius, 25.0)

        # Halo position: above head
        halo_y = int(head_y - head_radius * 1.5)
        halo_x = int(head_x)
        halo_w = int(head_radius * (1.0 + bass * 0.2))
        halo_h = int(head_radius * 0.35)

        # Golden color, brightens with bass
        brightness = int(180 + bass * 75)
        halo_color = (0, brightness, brightness)  # golden in BGR

        # Draw halo ellipse
        if 0 < halo_y < layer.shape[0] and 0 < halo_x < layer.shape[1]:
            cv2.ellipse(layer, (halo_x, halo_y), (halo_w, halo_h),
                        0, 0, 360, halo_color, 3, cv2.LINE_AA)
            # Slightly thicker behind for 3D tilt illusion
            cv2.ellipse(layer, (halo_x, halo_y + 2), (halo_w - 2, halo_h - 1),
                        0, 160, 380, halo_color, 2, cv2.LINE_AA)

    def _draw_wings(self, layer: np.ndarray, kpts: np.ndarray,
                    confs: np.ndarray, bass: float, h: int, w: int) -> None:
        """Draw curved energy wings from shoulders."""
        # Need both shoulders
        l_shoulder_idx, r_shoulder_idx = 5, 6
        l_elbow_idx, r_elbow_idx = 7, 8

        for side, sh_idx, elb_idx in [("left", l_shoulder_idx, l_elbow_idx),
                                       ("right", r_shoulder_idx, r_elbow_idx)]:
            if confs[sh_idx] < CONFIDENCE_MIN:
                continue

            sx, sy = float(kpts[sh_idx, 0]), float(kpts[sh_idx, 1])

            # Wing direction based on shoulder-elbow angle (or default outward)
            if confs[elb_idx] > CONFIDENCE_MIN:
                ex, ey = float(kpts[elb_idx, 0]), float(kpts[elb_idx, 1])
                arm_angle = math.atan2(ey - sy, ex - sx)
            else:
                arm_angle = math.pi / 2 if side == "left" else -math.pi / 2

            # Wing extends opposite to arm direction and upward
            wing_dir = -1 if side == "left" else 1
            spread = 1.0 + bass * 0.2

            # Generate 3-4 curved wing feather paths
            wing_color = (180, 220, 255)  # warm white-gold
            num_feathers = 4

            for f in range(num_feathers):
                # Each feather arcs outward and slightly downward
                angle_offset = (f - num_feathers / 2) * 0.25 * spread
                base_angle = -math.pi / 2 + wing_dir * 0.8 + angle_offset

                length = int((80 + f * 20) * spread)
                points = []
                for t in np.linspace(0, 1, 12):
                    # Curved path: starts at shoulder, arcs outward and down
                    cx = sx + wing_dir * t * length * math.cos(base_angle + t * 0.5)
                    cy = sy + t * length * math.sin(base_angle + t * 0.5) * 0.6
                    points.append([int(cx), int(cy)])

                if len(points) > 2:
                    pts_arr = np.array(points, dtype=np.int32)
                    # Fade: inner feathers brighter
                    alpha = 0.6 - f * 0.1
                    feather_color = tuple(int(c * alpha) for c in wing_color)
                    cv2.polylines(layer, [pts_arr], False, feather_color,
                                  2, cv2.LINE_AA)

                    # Spawn particles from wingtip on bass
                    if bass > 0.7 and f == num_feathers - 1 and len(points) > 0:
                        tip = points[-1]
                        if 0 <= tip[0] < w and 0 <= tip[1] < h:
                            for _ in range(2):
                                self._particles.append(
                                    _WingParticle(float(tip[0]), float(tip[1]))
                                )

        # Cap particles
        if len(self._particles) > 60:
            self._particles = self._particles[-60:]

    def _draw_particles(self, frame: np.ndarray) -> None:
        """Update and draw wing particles."""
        alive = []
        for p in self._particles:
            if p.update():
                alpha = p.life / p.max_life
                brightness = int(200 * alpha)
                color = (0, brightness, brightness)  # golden fade
                cv2.circle(frame, (int(p.x), int(p.y)), 2, color, -1)
                alive.append(p)
        self._particles = alive

    @property
    def name(self) -> str:
        return "Halo Wings"

    @property
    def needs_pose(self) -> bool:
        return True
