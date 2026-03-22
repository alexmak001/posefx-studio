"""Sprite puppet renderer — 2D geometric character driven by pose keypoints.

Supports a custom avatar image: if ctx.avatar is set, it replaces the
cartoon head with the uploaded image, scaled and rotated to match the
detected head position and orientation.
"""

import math
import time

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
        background = ctx.frame.copy()

        if not ctx.pose or ctx.pose.num_people == 0:
            return background

        bass = ctx.bass_energy
        scale_pulse = 1.0 + bass * 0.08
        opacity = ctx.puppet_opacity

        # Draw puppet onto a separate layer
        puppet_layer = background.copy()
        for person_idx in range(ctx.pose.num_people):
            kpts = ctx.pose.keypoints[person_idx]
            confs = ctx.pose.confidences[person_idx]
            self._draw_puppet(puppet_layer, kpts, confs, bass, scale_pulse, ctx.avatar)

        # Blend puppet layer with camera feed at the configured opacity
        output = cv2.addWeighted(puppet_layer, opacity, background, 1.0 - opacity, 0)
        return output

    @staticmethod
    def _hue_to_bgr(hue: float) -> tuple[int, int, int]:
        """Convert a hue (0-180) to a bright BGR color."""
        hsv = np.uint8([[[int(hue) % 180, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def _draw_puppet(self, frame: np.ndarray, kpts: np.ndarray,
                     confs: np.ndarray, bass: float,
                     scale_pulse: float,
                     avatar: np.ndarray | None = None) -> None:
        """Draw one geometric puppet character with glowing color-cycling skeleton."""
        h, w = frame.shape[:2]

        # Color cycling — hue rotates over time, bass speeds it up
        t = time.monotonic()
        base_hue = (t * 30 + bass * 60) % 180  # slow rotation, bass pushes faster

        # Draw skeleton on a separate layer for glow effect
        glow_layer = np.zeros_like(frame)

        # Draw torso fill (quadrilateral)
        torso_hue = (base_hue + 0) % 180
        torso_color = self._hue_to_bgr(torso_hue)
        if all(confs[i] > CONFIDENCE_MIN for i in [_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP]):
            torso_pts = np.array([
                [int(kpts[_L_SHOULDER, 0]), int(kpts[_L_SHOULDER, 1])],
                [int(kpts[_R_SHOULDER, 0]), int(kpts[_R_SHOULDER, 1])],
                [int(kpts[_R_HIP, 0]), int(kpts[_R_HIP, 1])],
                [int(kpts[_L_HIP, 0]), int(kpts[_L_HIP, 1])],
            ], dtype=np.int32)
            # Semi-transparent fill on the main frame
            overlay = frame.copy()
            cv2.fillPoly(overlay, [torso_pts], torso_color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            # Bright outline on glow layer
            cv2.polylines(glow_layer, [torso_pts], True, torso_color, 3, cv2.LINE_AA)

        # Draw limb segments with color cycling
        for i, (start_idx, end_idx, _color, width_base) in enumerate(_LIMB_DEFS):
            if confs[start_idx] < CONFIDENCE_MIN or confs[end_idx] < CONFIDENCE_MIN:
                continue

            p1 = (int(kpts[start_idx, 0]), int(kpts[start_idx, 1]))
            p2 = (int(kpts[end_idx, 0]), int(kpts[end_idx, 1]))
            width = max(2, int(width_base * scale_pulse * 0.5))

            # Each limb offset in hue for rainbow effect
            limb_hue = (base_hue + i * 12) % 180
            limb_color = self._hue_to_bgr(limb_hue)

            # Draw on glow layer
            cv2.line(glow_layer, p1, p2, limb_color, width, cv2.LINE_AA)

        # Draw joints (skip head keypoints when avatar is set)
        head_keypoints = {_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR}
        joint_radius = int(5 * scale_pulse)
        for k in range(17):
            if confs[k] > CONFIDENCE_MIN:
                if avatar is not None and k in head_keypoints:
                    continue
                pt = (int(kpts[k, 0]), int(kpts[k, 1]))
                joint_hue = (base_hue + k * 10) % 180
                joint_color = self._hue_to_bgr(joint_hue)
                cv2.circle(glow_layer, pt, joint_radius, joint_color, -1, cv2.LINE_AA)

        # Apply glow: blur the skeleton layer and add it twice (blur + sharp)
        glow_radius = int(15 + bass * 20)
        glow_radius = glow_radius if glow_radius % 2 == 1 else glow_radius + 1
        blurred = cv2.GaussianBlur(glow_layer, (glow_radius, glow_radius), 0)
        # Additive blend: glow halo + sharp lines
        frame[:] = cv2.add(frame, blurred)
        frame[:] = cv2.add(frame, glow_layer)

        # Draw head — custom avatar or default cartoon (after glow so it's on top)
        if avatar is not None:
            self._draw_avatar_head(frame, kpts, confs, avatar, scale_pulse)
        else:
            self._draw_head(frame, kpts, confs, bass, scale_pulse)

        # Bass hit: particle burst from joints
        if bass > 0.7:
            for k in [_L_WRIST, _R_WRIST, _L_ANKLE, _R_ANKLE]:
                if confs[k] > CONFIDENCE_MIN:
                    pt = (int(kpts[k, 0]), int(kpts[k, 1]))
                    burst_radius = int(10 + bass * 15)
                    burst_color = self._hue_to_bgr((base_hue + 90) % 180)
                    cv2.circle(frame, pt, burst_radius, burst_color, 2, cv2.LINE_AA)

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

    def _draw_avatar_head(self, frame: np.ndarray, kpts: np.ndarray,
                          confs: np.ndarray, avatar: np.ndarray,
                          scale: float) -> None:
        """Draw the custom avatar image at the detected head position."""
        h, w = frame.shape[:2]

        # Find head center and size from keypoints
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
        radius = int(max(spread * 1.4, 30) * scale)
        diameter = radius * 2

        if diameter < 10:
            return

        # Compute head tilt angle from eyes
        # COCO: L_EYE=1 is the person's left (screen-right), R_EYE=2 is person's right (screen-left)
        # We want the angle the eye line makes with horizontal, then rotate avatar to match
        angle_deg = 0.0
        if confs[_L_EYE] > CONFIDENCE_MIN and confs[_R_EYE] > CONFIDENCE_MIN:
            dx = kpts[_L_EYE, 0] - kpts[_R_EYE, 0]
            dy = kpts[_L_EYE, 1] - kpts[_R_EYE, 1]
            angle_deg = math.degrees(math.atan2(dy, dx))

        # Resize avatar to fit head (always square-ish to avoid distortion)
        ah, aw = avatar.shape[:2]
        new_size = diameter

        if new_size < 4:
            return

        resized = cv2.resize(avatar, (new_size, new_size), interpolation=cv2.INTER_AREA)

        # Rotate to match head tilt
        rot_mat = cv2.getRotationMatrix2D((new_size / 2, new_size / 2), -angle_deg, 1.0)
        cos_a = abs(rot_mat[0, 0])
        sin_a = abs(rot_mat[0, 1])
        rot_w = int(new_size * sin_a + new_size * cos_a)
        rot_h = int(new_size * cos_a + new_size * sin_a)
        rot_mat[0, 2] += (rot_w - new_size) / 2
        rot_mat[1, 2] += (rot_h - new_size) / 2

        channels = resized.shape[2] if resized.ndim == 3 else 1
        border_val = (0, 0, 0, 0) if channels == 4 else (0, 0, 0)
        rotated = cv2.warpAffine(resized, rot_mat, (rot_w, rot_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=border_val)

        # Compute paste region on frame
        x1 = int(cx - rot_w / 2)
        y1 = int(cy - rot_h / 2)
        x2 = x1 + rot_w
        y2 = y1 + rot_h

        # Clip to frame bounds
        sx1 = max(0, -x1)
        sy1 = max(0, -y1)
        dx1 = max(0, x1)
        dy1 = max(0, y1)
        dx2 = min(w, x2)
        dy2 = min(h, y2)
        sx2 = sx1 + (dx2 - dx1)
        sy2 = sy1 + (dy2 - dy1)

        if dx2 <= dx1 or dy2 <= dy1:
            return

        roi = rotated[sy1:sy2, sx1:sx2]
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            return

        if roi.shape[2] == 4:
            # Alpha blend
            alpha = roi[:, :, 3:4].astype(np.float32) / 255.0
            bgr = roi[:, :, :3]
            frame[dy1:dy2, dx1:dx2] = (
                bgr * alpha + frame[dy1:dy2, dx1:dx2] * (1.0 - alpha)
            ).astype(np.uint8)
        else:
            # No alpha — create circular mask for clean look
            mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
            center = (roi.shape[1] // 2, roi.shape[0] // 2)
            rad = min(roi.shape[0], roi.shape[1]) // 2
            cv2.circle(mask, center, rad, 255, -1, cv2.LINE_AA)
            mask_3 = mask[:, :, np.newaxis].astype(np.float32) / 255.0
            frame[dy1:dy2, dx1:dx2] = (
                roi * mask_3 + frame[dy1:dy2, dx1:dx2] * (1.0 - mask_3)
            ).astype(np.uint8)

    @property
    def name(self) -> str:
        return "Sprite Puppet"

    @property
    def needs_pose(self) -> bool:
        return True
