"""Robot skeleton renderer — angular metallic joints and connectors."""

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.effects import CONFIDENCE_MIN, SKELETON_CONNECTIONS


class RobotSkeletonRenderer(BaseRenderer):
    """Metallic blue/silver skeleton with thick joints and angular connectors.

    Features glowing joint highlights and thick industrial-looking limbs.
    Bass energy controls joint pulse size and glow intensity.
    """

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render robot skeleton effect.

        Args:
            ctx: Current frame data with pose and mask.

        Returns:
            Composited frame with robot skeleton on black background.
        """
        h, w = ctx.frame.shape[:2]
        output = ctx.frame.copy()

        # Black out person area
        if ctx.mask and ctx.mask.num_people > 0:
            output[ctx.mask.combined_mask > 0] = 0

        if not ctx.pose or ctx.pose.num_people == 0:
            return output

        bass = ctx.bass_energy
        joint_radius = int(10 + bass * 10)
        line_thickness = int(6 + bass * 4)

        line_color = (200, 190, 170)    # bright silver
        joint_color = (255, 180, 50)    # bright electric blue (BGR)
        joint_core = (255, 240, 200)    # near-white center
        joint_border = (120, 100, 70)   # darker border

        glow_layer = np.zeros((h, w, 3), dtype=np.uint8)

        for person_idx in range(ctx.pose.num_people):
            kpts = ctx.pose.keypoints[person_idx]
            confs = ctx.pose.confidences[person_idx]

            # Draw thick connector lines
            for i, j in SKELETON_CONNECTIONS:
                if confs[i] > CONFIDENCE_MIN and confs[j] > CONFIDENCE_MIN:
                    pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                    pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                    # Dark border behind
                    cv2.line(output, pt1, pt2, joint_border, line_thickness + 4, cv2.LINE_AA)
                    cv2.line(output, pt1, pt2, line_color, line_thickness, cv2.LINE_AA)

            # Draw joints: border → fill → bright center → glow
            for k in range(17):
                if confs[k] > CONFIDENCE_MIN:
                    x, y = int(kpts[k, 0]), int(kpts[k, 1])
                    cv2.circle(output, (x, y), joint_radius + 4, joint_border, -1, cv2.LINE_AA)
                    cv2.circle(output, (x, y), joint_radius, joint_color, -1, cv2.LINE_AA)
                    cv2.circle(output, (x, y), joint_radius // 2, joint_core, -1, cv2.LINE_AA)
                    # Add to glow layer
                    cv2.circle(glow_layer, (x, y), joint_radius + 6, joint_color, -1)

        # Glow around joints
        glow_size = int(31 + bass * 30)
        if glow_size % 2 == 0:
            glow_size += 1
        glow = cv2.GaussianBlur(glow_layer, (glow_size, glow_size), 0)
        output = cv2.add(output, glow)

        return output

    @property
    def name(self) -> str:
        return "Robot Skeleton"

    @property
    def needs_pose(self) -> bool:
        return True

    @property
    def needs_mask(self) -> bool:
        return True
