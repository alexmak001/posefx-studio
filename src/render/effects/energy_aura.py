"""Energy aura renderer — light rainbow color fade radiating from body outline."""

import time

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.utils import composite_head


class EnergyAuraRenderer(BaseRenderer):
    """Light rainbow color fade radiating from body outline.

    Uses distance transform for fast concentric glow rings around
    the body silhouette. Colors cycle through the rainbow over time.
    """

    def render(self, ctx: RenderContext) -> np.ndarray:
        h, w = ctx.frame.shape[:2]
        original = ctx.frame.copy()
        output = ctx.frame.copy()

        if not ctx.mask or ctx.mask.num_people == 0:
            return output

        bass = ctx.bass_energy
        combined = ctx.mask.combined_mask
        t = time.monotonic()

        # Warm color grade inside body
        body_region = combined > 0
        output[body_region, 2] = np.clip(
            output[body_region, 2].astype(np.int16) + 15, 0, 255
        ).astype(np.uint8)
        output[body_region, 0] = np.clip(
            output[body_region, 0].astype(np.int16) - 10, 0, 255
        ).astype(np.uint8)

        # Distance transform from mask edge
        inverted = 1 - combined
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)

        # Build aura using distance bands
        max_dist = 40.0 * (1.0 + bass)
        aura_mask = (dist > 0) & (dist < max_dist)

        if np.any(aura_mask):
            norm_dist = np.clip(dist / max_dist, 0, 1)

            # Rainbow hue cycling over time
            base_hue = (t * 20) % 180

            # Create hue field: base_hue + offset from distance
            hue_field = (base_hue + norm_dist * 60) % 180

            # Convert hue to BGR using vectorized HSV approach
            hsv_img = np.zeros((h, w, 3), dtype=np.uint8)
            hsv_img[:, :, 0] = (hue_field % 180).astype(np.uint8)
            hsv_img[:, :, 1] = 255
            hsv_img[:, :, 2] = 255
            bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

            aura_layer = bgr_img.astype(np.float32)

            # Bass white-hot shift
            if bass > 0.5:
                white_t = (bass - 0.5) * 2 * 0.3
                aura_layer = aura_layer * (1 - white_t) + 255 * white_t

            # Fade with distance
            fade = (1.0 - norm_dist) ** 1.5
            aura_layer *= fade[:, :, np.newaxis]
            aura_layer[~aura_mask] = 0

            blur_size = int(21 + bass * 10)
            if blur_size % 2 == 0:
                blur_size += 1
            aura_u8 = np.clip(aura_layer, 0, 255).astype(np.uint8)
            aura_glow = cv2.GaussianBlur(aura_u8, (blur_size, blur_size), 0)
            output = cv2.add(output, aura_glow)

        return composite_head(output, original, ctx.pose, ctx.frame.shape)

    @property
    def name(self) -> str:
        return "Energy Aura"

    @property
    def needs_mask(self) -> bool:
        return True

    @property
    def needs_pose(self) -> bool:
        return True
