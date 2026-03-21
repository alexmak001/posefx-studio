"""Sprite puppet renderer — 2D geometric character driven by pose keypoints."""

import math

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.effects import CONFIDENCE_MIN

# COCO keypoint indices
_NOSE = 0
_L_EYE, _R_EYE = 1, 2
_L_EAR, _R_EAR = 3, 4
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16

# Limb segment definitions: (start_kpt, end_kpt, color_BGR, width_base)
_LIMB_DEFS = [
    # Torso (drawn first as background)
    (_L_SHOULDER, _L_HIP, (80, 60, 180), 24),
    (_R_SHOULDER, _R_HIP, (80, 60, 180), 24),
    (_L_SHOULDER, _R_SHOULDER, (80, 60, 180), 20),
    (_L_HIP, _R_HIP, (80, 60, 180), 20),
    # Back arms (behind torso)
    (_R_SHOULDER, _R_ELBOW, (60, 50, 150), 14),
    (_R_ELBOW, _R_WRIST, (60, 50, 150), 12),
    # Front arms
    (_L_SHOULDER, _L_ELBOW, (100, 70, 200), 14),
    (_L_ELBOW, _L_WRIST, (100, 70, 200), 12),
    # Legs
    (_L_HIP, _L_KNEE, (150, 100, 50), 16),
    (_L_KNEE, _L_ANKLE, (150, 100, 50), 14),
    (_R_HIP, _R_KNEE, (130, 80, 40), 16),
    (_R_KNEE, _R_ANKLE, (130, 80, 40), 14),
]


class SpritePuppetRenderer(BaseRenderer):
    """Full body replacement with a geometric 2D puppet character.

    Uses colored rectangles and circles positioned at COCO keypoints
    to create a simple but expressive avatar. Face IS replaced (full
    avatar mode). Falls back gracefully when keypoints are missing.
    """

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render sprite puppet effect.

        Args:
            ctx: Current frame data with pose.

        Returns:
            Frame with geometric puppet avatar.
        """
        h, w = ctx.frame.shape[:2]
        # Dark background
        output = np.full((h, w, 3), 25, dtype=np.uint8)

        if not ctx.pose or ctx.pose.num_people == 0:
            return output

        bass = ctx.bass_energy
        scale_pulse = 1.0 + bass * 0.08

        for person_idx in range(ctx.pose.num_people):
            kpts = ctx.pose.keypoints[person_idx]
            confs = ctx.pose.confidences[person_idx]

            self._draw_puppet(output, kpts, confs, bass, scale_pulse)

        return output

    def _draw_puppet(self, frame: np.ndarray, kpts: np.ndarray,
                     confs: np.ndarray, bass: float,
                     scale_pulse: float) -> None:
        """Draw one geometric puppet character."""
        h, w = frame.shape[:2]

        # Draw torso fill (quadrilateral)
        if all(confs[i] > CONFIDENCE_MIN for i in [_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP]):
            torso_pts = np.array([
                [int(kpts[_L_SHOULDER, 0]), int(kpts[_L_SHOULDER, 1])],
                [int(kpts[_R_SHOULDER, 0]), int(kpts[_R_SHOULDER, 1])],
                [int(kpts[_R_HIP, 0]), int(kpts[_R_HIP, 1])],
                [int(kpts[_L_HIP, 0]), int(kpts[_L_HIP, 1])],
            ], dtype=np.int32)
            cv2.fillPoly(frame, [torso_pts], (80, 60, 180))
            cv2.polylines(frame, [torso_pts], True, (120, 90, 220), 2, cv2.LINE_AA)

        # Draw limb segments
        for start_idx, end_idx, color, width_base in _LIMB_DEFS:
            if confs[start_idx] < CONFIDENCE_MIN or confs[end_idx] < CONFIDENCE_MIN:
                continue

            p1 = (int(kpts[start_idx, 0]), int(kpts[start_idx, 1]))
            p2 = (int(kpts[end_idx, 0]), int(kpts[end_idx, 1]))
            width = int(width_base * scale_pulse)

            # Draw as a thick rounded line for a limb look
            cv2.line(frame, p1, p2, color, width, cv2.LINE_AA)

            # Outline
            outline_color = tuple(min(255, c + 40) for c in color)
            cv2.line(frame, p1, p2, outline_color, 2, cv2.LINE_AA)

        # Draw joints as circles
        joint_color = (200, 200, 220)
        joint_radius = int(6 * scale_pulse)
        for k in range(17):
            if confs[k] > CONFIDENCE_MIN:
                pt = (int(kpts[k, 0]), int(kpts[k, 1]))
                cv2.circle(frame, pt, joint_radius, joint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, joint_radius, (100, 100, 120), 1, cv2.LINE_AA)

        # Draw head (larger circle with face features)
        self._draw_head(frame, kpts, confs, bass, scale_pulse)

        # Bass hit: particle burst from joints
        if bass > 0.7:
            for k in [_L_WRIST, _R_WRIST, _L_ANKLE, _R_ANKLE]:
                if confs[k] > CONFIDENCE_MIN:
                    pt = (int(kpts[k, 0]), int(kpts[k, 1]))
                    burst_radius = int(10 + bass * 15)
                    cv2.circle(frame, pt, burst_radius, (100, 200, 255), 2, cv2.LINE_AA)

    def _draw_head(self, frame: np.ndarray, kpts: np.ndarray,
                   confs: np.ndarray, bass: float, scale: float) -> None:
        """Draw a cartoon head with simple face features."""
        # Find head center and radius
        head_pts = []
        for idx in [_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR]:
            if confs[idx] > CONFIDENCE_MIN:
                head_pts.append(kpts[idx])

        if len(head_pts) < 2:
            return

        pts = np.array(head_pts)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        spread = float(np.max(np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)))
        radius = int(max(spread * 1.6, 30) * scale)

        # Head circle - skin tone
        head_color = (140, 180, 220)  # warm peach in BGR
        cv2.circle(frame, (int(cx), int(cy)), radius, head_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(cx), int(cy)), radius, (100, 140, 180), 2, cv2.LINE_AA)

        # Eyes
        if confs[_L_EYE] > CONFIDENCE_MIN and confs[_R_EYE] > CONFIDENCE_MIN:
            eye_radius = max(int(radius * 0.15), 3)
            for idx in [_L_EYE, _R_EYE]:
                ex, ey = int(kpts[idx, 0]), int(kpts[idx, 1])
                # White
                cv2.circle(frame, (ex, ey), eye_radius + 2, (255, 255, 255), -1)
                # Pupil
                cv2.circle(frame, (ex, ey), eye_radius, (40, 30, 20), -1)

        # Mouth (simple arc below nose)
        if confs[_NOSE] > CONFIDENCE_MIN:
            mx, my = int(kpts[_NOSE, 0]), int(kpts[_NOSE, 1]) + int(radius * 0.35)
            mouth_w = int(radius * 0.4)
            cv2.ellipse(frame, (mx, my), (mouth_w, int(mouth_w * 0.4)),
                        0, 0, 180, (60, 40, 120), 2, cv2.LINE_AA)

    @property
    def name(self) -> str:
        return "Sprite Puppet"

    @property
    def needs_pose(self) -> bool:
        return True
