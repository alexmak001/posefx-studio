"""Body glow renderer — glowing outline around the person silhouette."""

import time

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext


class BodyGlowRenderer(BaseRenderer):
    """Bright glowing outline around the body silhouette.

    Interior of mask is fully blacked out, with a thick multi-pass
    glow on the edge. Bass energy controls glow color cycling speed,
    thickness, and brightness.
    """

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render body glow effect.

        Args:
            ctx: Current frame data with mask.

        Returns:
            Composited frame with glowing body outline.
        """
        h, w = ctx.frame.shape[:2]
        output = ctx.frame.copy()

        if not ctx.mask or ctx.mask.num_people == 0:
            return output

        # Full black out the person interior
        output[ctx.mask.combined_mask > 0] = 0

        # Find contours
        contours, _ = cv2.findContours(
            ctx.mask.combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return output

        bass = ctx.bass_energy

        # Color cycles with time, bass accelerates it
        cycle_speed = 0.5 + bass * 4.0
        hue = (time.monotonic() * cycle_speed * 60) % 180

        # Full brightness, full saturation
        hsv_pixel = np.array([[[int(hue), 255, 255]]], dtype=np.uint8)
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
        color = tuple(int(c) for c in bgr_pixel[0, 0])

        # Bright white-ish core color
        core_hsv = np.array([[[int(hue), 80, 255]]], dtype=np.uint8)
        core_bgr = cv2.cvtColor(core_hsv, cv2.COLOR_HSV2BGR)
        core_color = tuple(int(c) for c in core_bgr[0, 0])

        # Thick contour lines on separate layers
        outer_layer = np.zeros((h, w, 3), dtype=np.uint8)
        core_layer = np.zeros((h, w, 3), dtype=np.uint8)

        outer_thickness = int(8 + bass * 12)
        core_thickness = int(3 + bass * 4)

        cv2.drawContours(outer_layer, contours, -1, color, outer_thickness, cv2.LINE_AA)
        cv2.drawContours(core_layer, contours, -1, core_color, core_thickness, cv2.LINE_AA)

        # Multi-pass glow
        glow_large = int(61 + bass * 50)
        if glow_large % 2 == 0:
            glow_large += 1
        glow_med = int(25 + bass * 20)
        if glow_med % 2 == 0:
            glow_med += 1

        blur_large = cv2.GaussianBlur(outer_layer, (glow_large, glow_large), 0)
        blur_med = cv2.GaussianBlur(outer_layer, (glow_med, glow_med), 0)

        output = cv2.add(output, blur_large)
        output = cv2.add(output, blur_med)
        output = cv2.add(output, core_layer)

        return output

    @property
    def name(self) -> str:
        return "Body Glow"

    @property
    def needs_mask(self) -> bool:
        return True
