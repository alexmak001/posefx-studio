"""Neon wireframe renderer — glowing skeleton + halo circle around each head."""

import time

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.effects import CONFIDENCE_MIN, SKELETON_CONNECTIONS


class NeonWireframeRenderer(BaseRenderer):
    """Glowing neon skeleton and halo circle around each detected head.

    Multi-color hue-cycling with heavy bloom glow.
    Bass energy controls thickness, glow intensity, and color speed.
    """

    def render(self, ctx: RenderContext) -> np.ndarray:
        h, w = ctx.frame.shape[:2]
        # Dim entire frame for contrast
        output = (ctx.frame * 0.5).astype(np.uint8)

        if not ctx.pose or ctx.pose.num_people == 0:
            return output

        bass = ctx.bass_energy
        t = time.monotonic()

        # Bass-reactive parameters
        thickness = int(5 + bass * 10)
        joint_radius = int(8 + bass * 6)
        glow_kernel = int(31 + bass * 50)
        if glow_kernel % 2 == 0:
            glow_kernel += 1
        bloom_kernel = int(61 + bass * 80)
        if bloom_kernel % 2 == 0:
            bloom_kernel += 1

        # Hue rotates over time, bass speeds it up
        base_hue = (t * 25 + bass * 80) % 180

        # Draw on separate layers for glow passes
        core_layer = np.zeros((h, w, 3), dtype=np.uint8)
        glow_layer = np.zeros((h, w, 3), dtype=np.uint8)

        for person_idx in range(ctx.pose.num_people):
            kpts = ctx.pose.keypoints[person_idx]
            confs = ctx.pose.confidences[person_idx]

            # ---- Skeleton lines ----
            for conn_idx, (i, j) in enumerate(SKELETON_CONNECTIONS):
                if confs[i] > CONFIDENCE_MIN and confs[j] > CONFIDENCE_MIN:
                    pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                    pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))

                    limb_hue = (base_hue + conn_idx * 11) % 180
                    color = _hue_to_bgr(limb_hue)
                    glow_hue = (limb_hue + 15) % 180
                    glow_color = _hue_to_bgr(glow_hue)

                    cv2.line(glow_layer, pt1, pt2, glow_color,
                             thickness + 10, cv2.LINE_AA)
                    cv2.line(core_layer, pt1, pt2, color,
                             thickness, cv2.LINE_AA)

            # ---- Joint dots ----
            for k in range(17):
                if confs[k] > CONFIDENCE_MIN:
                    pt = (int(kpts[k, 0]), int(kpts[k, 1]))
                    joint_hue = (base_hue + k * 10) % 180
                    color = _hue_to_bgr(joint_hue)
                    glow_color = _hue_to_bgr((joint_hue + 15) % 180)

                    cv2.circle(glow_layer, pt, joint_radius + 6,
                               glow_color, -1, cv2.LINE_AA)
                    cv2.circle(core_layer, pt, joint_radius,
                               color, -1, cv2.LINE_AA)

            # ---- Head glow circle ----
            head_pts = []
            for idx in [0, 1, 2, 3, 4]:
                if confs[idx] > CONFIDENCE_MIN:
                    head_pts.append(kpts[idx])

            if len(head_pts) >= 2:
                pts = np.array(head_pts)
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                spread = float(np.max(np.sqrt(
                    (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
                )))
                radius = int(max(spread * 1.8, 35) * (1.0 + bass * 0.15))

                person_hue = (base_hue + person_idx * 40) % 180
                color = _hue_to_bgr(person_hue)
                glow_color = _hue_to_bgr((person_hue + 15) % 180)

                cv2.circle(glow_layer, (cx, cy), radius, glow_color,
                           thickness + 10, cv2.LINE_AA)
                cv2.circle(core_layer, (cx, cy), radius, color,
                           thickness, cv2.LINE_AA)

        # Multi-pass glow: tight halo + wide bloom
        glow_tight = cv2.GaussianBlur(glow_layer, (glow_kernel, glow_kernel), 0)
        glow_wide = cv2.GaussianBlur(glow_layer, (bloom_kernel, bloom_kernel), 0)

        output = cv2.add(output, glow_wide)
        output = cv2.add(output, glow_tight)
        output = cv2.add(output, core_layer)

        return output

    @property
    def name(self) -> str:
        return "Neon Wireframe"

    @property
    def needs_pose(self) -> bool:
        return True

    @property
    def needs_mask(self) -> bool:
        return False


def _hue_to_bgr(hue: float) -> tuple[int, int, int]:
    """Convert a hue (0-180) to a bright BGR color."""
    hsv = np.uint8([[[int(hue) % 180, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])
