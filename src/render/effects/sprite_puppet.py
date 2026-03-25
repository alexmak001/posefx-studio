"""Custom Avatar renderer — displays uploaded avatar or cartoon head at detected position.

Supports a custom avatar image: if ctx.avatar is set, it replaces the
cartoon head with the uploaded image, scaled and rotated to match the
detected head position and orientation.
"""

import math

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.effects import CONFIDENCE_MIN

# COCO keypoint indices
_NOSE = 0
_L_EYE, _R_EYE = 1, 2
_L_EAR, _R_EAR = 3, 4


class SpritePuppetRenderer(BaseRenderer):
    """Custom avatar overlay on detected heads.

    If a custom avatar is uploaded, it is placed at each detected head
    position with rotation matching eye angle. Otherwise shows a simple
    cartoon head. Opacity is controllable via puppet_opacity.
    """

    def render(self, ctx: RenderContext) -> np.ndarray:
        h, w = ctx.frame.shape[:2]
        background = ctx.frame.copy()

        if not ctx.pose or ctx.pose.num_people == 0:
            return background

        bass = ctx.bass_energy
        scale_pulse = 1.0 + bass * 0.08
        opacity = ctx.puppet_opacity

        # Draw avatar onto a separate layer
        puppet_layer = background.copy()
        for person_idx in range(ctx.pose.num_people):
            kpts = ctx.pose.keypoints[person_idx]
            confs = ctx.pose.confidences[person_idx]
            if ctx.avatar is not None:
                self._draw_avatar_head(puppet_layer, kpts, confs, ctx.avatar, scale_pulse)
            else:
                self._draw_head(puppet_layer, kpts, confs, bass, scale_pulse)

        output = cv2.addWeighted(puppet_layer, opacity, background, 1.0 - opacity, 0)
        return output

    def _draw_head(self, frame: np.ndarray, kpts: np.ndarray,
                   confs: np.ndarray, bass: float, scale: float) -> None:
        """Draw a cartoon head with simple face features."""
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

        head_color = (140, 180, 220)
        cv2.circle(frame, (int(cx), int(cy)), radius, head_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(cx), int(cy)), radius, (100, 140, 180), 2, cv2.LINE_AA)

        if confs[_L_EYE] > CONFIDENCE_MIN and confs[_R_EYE] > CONFIDENCE_MIN:
            eye_radius = max(int(radius * 0.15), 3)
            for idx in [_L_EYE, _R_EYE]:
                ex, ey = int(kpts[idx, 0]), int(kpts[idx, 1])
                cv2.circle(frame, (ex, ey), eye_radius + 2, (255, 255, 255), -1)
                cv2.circle(frame, (ex, ey), eye_radius, (40, 30, 20), -1)

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

        angle_deg = 0.0
        if confs[_L_EYE] > CONFIDENCE_MIN and confs[_R_EYE] > CONFIDENCE_MIN:
            dx = kpts[_L_EYE, 0] - kpts[_R_EYE, 0]
            dy = kpts[_L_EYE, 1] - kpts[_R_EYE, 1]
            angle_deg = math.degrees(math.atan2(dy, dx))

        new_size = diameter
        if new_size < 4:
            return

        resized = cv2.resize(avatar, (new_size, new_size), interpolation=cv2.INTER_AREA)

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

        x1 = int(cx - rot_w / 2)
        y1 = int(cy - rot_h / 2)
        x2 = x1 + rot_w
        y2 = y1 + rot_h

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
            alpha = roi[:, :, 3:4].astype(np.float32) / 255.0
            bgr = roi[:, :, :3]
            frame[dy1:dy2, dx1:dx2] = (
                bgr * alpha + frame[dy1:dy2, dx1:dx2] * (1.0 - alpha)
            ).astype(np.uint8)
        else:
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
        return "Custom Avatar"

    @property
    def needs_pose(self) -> bool:
        return True
