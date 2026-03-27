"""Bass Pulse renderer — color overlay that syncs with music."""

import time

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext


class BassPulseRenderer(BaseRenderer):
    """Overlays slow-cycling color washes on the camera feed, pulsing with bass.

    Low bass = subtle tint. High bass = vivid color flash.
    The hue drifts over time so colors keep changing.
    """

    def render(self, ctx: RenderContext) -> np.ndarray:
        output = ctx.frame.copy()
        bass = ctx.bass_energy
        t = time.monotonic()

        # Hue rotates slowly over time
        hue = (t * 12) % 180

        # Bass controls overlay intensity: subtle tint at rest, vivid pulse on hits
        alpha = 0.05 + bass * 0.35

        # Create a full-frame color overlay
        h, w = output.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Primary color wash
        hsv_color = np.uint8([[[int(hue) % 180, 180, 200]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        overlay[:] = bgr_color

        # Add a second complementary color in a soft gradient from the edges
        hue2 = (hue + 90) % 180
        hsv_color2 = np.uint8([[[int(hue2) % 180, 160, 180]]])
        bgr_color2 = cv2.cvtColor(hsv_color2, cv2.COLOR_HSV2BGR)[0, 0]

        # Radial gradient: stronger at edges, weaker in center
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        edge_mask = np.clip(dist / max_dist, 0, 1)

        # Blend second color at edges
        for c in range(3):
            overlay[:, :, c] = (
                overlay[:, :, c].astype(np.float32) * (1 - edge_mask)
                + float(bgr_color2[c]) * edge_mask
            ).astype(np.uint8)

        # Blend overlay onto frame
        output = cv2.addWeighted(output, 1.0 - alpha, overlay, alpha, 0)

        # On strong bass hits, add a brief bright flash at the edges
        if bass > 0.6:
            flash_alpha = (bass - 0.6) * 0.5
            flash = np.zeros_like(output)
            flash_hue = (hue + 45) % 180
            hsv_flash = np.uint8([[[int(flash_hue) % 180, 255, 255]]])
            bgr_flash = cv2.cvtColor(hsv_flash, cv2.COLOR_HSV2BGR)[0, 0]
            flash[:] = bgr_flash

            # Vignette mask — flash only at edges
            vignette = np.clip(edge_mask * 1.5 - 0.3, 0, 1).astype(np.float32)
            for c in range(3):
                flash[:, :, c] = (flash[:, :, c].astype(np.float32) * vignette).astype(np.uint8)

            output = cv2.addWeighted(output, 1.0, flash, flash_alpha, 0)

        return output

    @property
    def name(self) -> str:
        return "Bass Pulse"
