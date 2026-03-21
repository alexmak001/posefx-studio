"""Neon wireframe renderer — glowing skeleton on black with real face."""

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.effects import CONFIDENCE_MIN, SKELETON_CONNECTIONS
from src.render.utils import composite_head


class NeonWireframeRenderer(BaseRenderer):
    """Replaces body with glowing neon skeleton lines on black.

    Person mask area is blacked out, bright cyan skeleton drawn on top
    with multi-pass glow. Head region shows the real camera face.
    Bass energy controls thickness, glow, and color shift.
    """

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render neon wireframe effect.

        Args:
            ctx: Current frame data with pose and mask.

        Returns:
            Composited frame with neon skeleton and real face.
        """
        h, w = ctx.frame.shape[:2]
        original = ctx.frame.copy()
        output = np.zeros_like(ctx.frame)

        # Keep background outside person mask
        if ctx.mask and ctx.mask.num_people > 0:
            mask_inv = ctx.mask.combined_mask == 0
            output[mask_inv] = ctx.frame[mask_inv]

        if not ctx.pose or ctx.pose.num_people == 0:
            return composite_head(output, original, ctx.pose, ctx.frame.shape)

        bass = ctx.bass_energy

        # Bass-reactive parameters
        thickness = int(4 + bass * 6)
        joint_radius = int(8 + bass * 4)
        glow_kernel = int(21 + bass * 30)
        if glow_kernel % 2 == 0:
            glow_kernel += 1

        # Color shifts from cyan toward white at high bass
        b_val = 255
        g_val = 255
        r_val = int(bass * 200)
        core_color = (b_val, g_val, r_val)       # cyan → white
        glow_color = (200, 200, 0)                # darker cyan for glow

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
                    cv2.line(glow_layer, pt1, pt2, glow_color,
                             thickness + 6, cv2.LINE_AA)
                    cv2.line(core_layer, pt1, pt2, core_color,
                             thickness, cv2.LINE_AA)

            for k in range(17):
                if confs[k] > CONFIDENCE_MIN:
                    pt = (int(kpts[k, 0]), int(kpts[k, 1]))
                    cv2.circle(glow_layer, pt, joint_radius + 4,
                               glow_color, -1, cv2.LINE_AA)
                    cv2.circle(core_layer, pt, joint_radius,
                               core_color, -1, cv2.LINE_AA)

        # Multi-pass glow
        glow = cv2.GaussianBlur(glow_layer, (glow_kernel, glow_kernel), 0)
        output = cv2.add(output, glow)
        output = cv2.add(output, core_layer)

        return composite_head(output, original, ctx.pose, ctx.frame.shape)

    @property
    def name(self) -> str:
        return "Neon Wireframe"

    @property
    def needs_pose(self) -> bool:
        return True

    @property
    def needs_mask(self) -> bool:
        return True
