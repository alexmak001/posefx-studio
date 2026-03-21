"""Mode-based engine for managing renderers and inference."""

import logging
import threading
import time

import cv2
import numpy as np

from src.inference.base import MaskResult, PoseResult
from src.inference.pose_estimator import YOLOPoseEstimator
from src.inference.segmenter import YOLOSegmenter
from src.render.base import BaseRenderer, RenderContext
from src.render.effects.digital_rain import DigitalRainRenderer
from src.render.effects.energy_aura import EnergyAuraRenderer
from src.render.effects.glitch_body import GlitchBodyRenderer
from src.render.effects.halo_wings import HaloWingsRenderer
from src.render.effects.motion_trails import MotionTrailsRenderer
from src.render.effects.neon_wireframe import NeonWireframeRenderer
from src.render.effects.particle_dissolve import ParticleDissolveRenderer
from src.render.effects.passthrough import PassthroughRenderer
from src.render.effects.shadow_clones import ShadowClonesRenderer
from src.render.effects.sprite_puppet import SpritePuppetRenderer
from src.utils.config import AppConfig

logger = logging.getLogger(__name__)


class PartyEngine:
    """Manages the active renderer, inference models, and frame processing.

    Only runs inference models that the current renderer actually needs,
    skipping unnecessary work for better performance.

    Args:
        config: Application configuration.
        platform: Detected platform string ('mac', 'jetson', 'cpu').
    """

    def __init__(self, config: AppConfig, platform: str) -> None:
        self._config = config
        self._platform = platform
        self._lock = threading.Lock()
        self._bass_energy = 0.0

        # Load inference models
        self._pose_estimator = YOLOPoseEstimator(config.inference)
        self._segmenter = YOLOSegmenter(config.inference)

        # Register all renderers in display order
        self._renderers: list[BaseRenderer] = [
            NeonWireframeRenderer(),
            EnergyAuraRenderer(),
            MotionTrailsRenderer(),
            GlitchBodyRenderer(),
            DigitalRainRenderer(),
            ShadowClonesRenderer(),
            ParticleDissolveRenderer(),
            HaloWingsRenderer(),
            SpritePuppetRenderer(),
            PassthroughRenderer(),
        ]
        self._renderer_map: dict[str, BaseRenderer] = {
            r.name: r for r in self._renderers
        }

        # Set default renderer
        default_name = getattr(config, "effects", None)
        if default_name and hasattr(default_name, "default"):
            # Try to find matching renderer
            for r in self._renderers:
                if r.name.lower().replace(" ", "_") == default_name.default:
                    self._active_idx = self._renderers.index(r)
                    break
            else:
                self._active_idx = 0
        else:
            self._active_idx = 0

        logger.info(
            "PartyEngine initialized with %d effects, active: %s",
            len(self._renderers),
            self.active_renderer.name,
        )

    @property
    def active_renderer(self) -> BaseRenderer:
        """The currently active renderer."""
        return self._renderers[self._active_idx]

    def set_renderer(self, name: str) -> None:
        """Switch active renderer by name. Thread-safe.

        Args:
            name: Human-readable effect name.
        """
        with self._lock:
            if name in self._renderer_map:
                self._active_idx = self._renderers.index(self._renderer_map[name])
                logger.info("Switched effect to: %s", name)
            else:
                logger.warning("Unknown effect: %s", name)

    def next_renderer(self) -> str:
        """Switch to the next renderer in the list. Returns the new name."""
        with self._lock:
            self._active_idx = (self._active_idx + 1) % len(self._renderers)
            name = self.active_renderer.name
        logger.info("Switched effect to: %s", name)
        return name

    def prev_renderer(self) -> str:
        """Switch to the previous renderer in the list. Returns the new name."""
        with self._lock:
            self._active_idx = (self._active_idx - 1) % len(self._renderers)
            name = self.active_renderer.name
        logger.info("Switched effect to: %s", name)
        return name

    def get_renderer_names(self) -> list[str]:
        """List all available effect names."""
        return [r.name for r in self._renderers]

    def set_bass_energy(self, energy: float) -> None:
        """Update bass energy from audio thread.

        Args:
            energy: Bass energy level 0.0-1.0.
        """
        self._bass_energy = max(0.0, min(1.0, energy))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run inference and render for one frame.

        Only runs the inference models that the current renderer needs.
        Downscales the frame for inference if inference_scale < 1.0,
        then scales results back to full resolution for rendering.

        Args:
            frame: BGR input frame.

        Returns:
            Composited output frame.
        """
        with self._lock:
            renderer = self.active_renderer

        pose: PoseResult | None = None
        mask: MaskResult | None = None
        scale = self._config.inference.inference_scale
        needs_inference = renderer.needs_pose or renderer.needs_mask

        # Downscale for inference if needed
        if needs_inference and 0 < scale < 1.0:
            small = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame
            scale = 1.0

        if renderer.needs_pose:
            pose = self._pose_estimator.infer(small)
            if scale < 1.0 and pose.num_people > 0:
                inv = 1.0 / scale
                pose = PoseResult(
                    keypoints=pose.keypoints * inv,
                    confidences=pose.confidences,
                    boxes=pose.boxes * inv,
                    num_people=pose.num_people,
                )

        if renderer.needs_mask:
            mask = self._segmenter.infer(small)
            if scale < 1.0 and mask.num_people > 0:
                h, w = frame.shape[:2]
                full_masks = np.zeros((mask.num_people, h, w), dtype=np.uint8)
                for i in range(mask.num_people):
                    full_masks[i] = cv2.resize(
                        mask.masks[i], (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    full_masks[i] = (full_masks[i] > 0).astype(np.uint8)
                mask = MaskResult(
                    masks=full_masks,
                    combined_mask=np.any(full_masks, axis=0).astype(np.uint8),
                    num_people=mask.num_people,
                )

        ctx = RenderContext(
            frame=frame,
            pose=pose,
            mask=mask,
            bass_energy=self._bass_energy,
            timestamp=time.monotonic(),
        )

        return renderer.render(ctx)
