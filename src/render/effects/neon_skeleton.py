"""Neon skeleton renderer — glowing skeleton lines on a black background."""

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.effects import CONFIDENCE_MIN, SKELETON_CONNECTIONS


class NeonSkeletonRenderer(BaseRenderer):
    """Replaces the body with glowing neon skeleton lines.

    Person mask area is blacked out, skeleton is drawn on top with
    multiple blur passes for a rich glow. Bass energy controls line
    thickness and glow intensity.
    """

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render neon skeleton effect.

        Args:
            ctx: Current frame data with pose and mask.

        Returns:
            Composited frame with neon skeleton on black background.
        """
        h, w = ctx.frame.shape[:2]
        output = ctx.frame.copy()

        # Black out person area completely
        if ctx.mask and ctx.mask.num_people > 0:
            output[ctx.mask.combined_mask > 0] = 0

        if not ctx.pose or ctx.pose.num_people == 0:
            return output

        bass = ctx.bass_energy

        # Thick base lines, bass makes them chunkier
        thickness = int(5 + bass * 8)
        joint_radius = int(7 + bass * 6)

        # Bright core color + outer glow color
        core_color = (150, 255, 150)   # bright green-white core
        glow_color = (0, 255, 80)      # saturated green glow

        # Draw skeleton on separate layers
        core_layer = np.zeros((h, w, 3), dtype=np.uint8)
        glow_layer = np.zeros((h, w, 3), dtype=np.uint8)

        for person_idx in range(ctx.pose.num_people):
            kpts = ctx.pose.keypoints[person_idx]
            confs = ctx.pose.confidences[person_idx]

            for i, j in SKELETON_CONNECTIONS:
                if confs[i] > CONFIDENCE_MIN and confs[j] > CONFIDENCE_MIN:
                    pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                    pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                    # Thick outer glow line
                    cv2.line(glow_layer, pt1, pt2, glow_color, thickness + 6, cv2.LINE_AA)
                    # Thinner bright core
                    cv2.line(core_layer, pt1, pt2, core_color, thickness, cv2.LINE_AA)

            for k in range(17):
                if confs[k] > CONFIDENCE_MIN:
                    x, y = int(kpts[k, 0]), int(kpts[k, 1])
                    cv2.circle(glow_layer, (x, y), joint_radius + 4, glow_color, -1, cv2.LINE_AA)
                    cv2.circle(core_layer, (x, y), joint_radius, core_color, -1, cv2.LINE_AA)

        # Multi-pass glow: large soft blur + medium blur + core
        glow_size_large = int(51 + bass * 40)
        if glow_size_large % 2 == 0:
            glow_size_large += 1
        glow_size_med = int(21 + bass * 20)
        if glow_size_med % 2 == 0:
            glow_size_med += 1

        glow_large = cv2.GaussianBlur(glow_layer, (glow_size_large, glow_size_large), 0)
        glow_med = cv2.GaussianBlur(glow_layer, (glow_size_med, glow_size_med), 0)

        # Stack: soft outer glow → medium glow → sharp core
        output = cv2.add(output, glow_large)
        output = cv2.add(output, glow_med)
        output = cv2.add(output, core_layer)

        return output

    @property
    def name(self) -> str:
        return "Neon Skeleton"

    @property
    def needs_pose(self) -> bool:
        return True

    @property
    def needs_mask(self) -> bool:
        return True
