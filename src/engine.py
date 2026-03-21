"""Mode-based engine for managing renderers and inference."""

import logging
import threading
import time

import numpy as np

from src.inference.base import MaskResult, PoseResult
from src.inference.pose_estimator import YOLOPoseEstimator
from src.inference.segmenter import YOLOSegmenter
from src.render.base import BaseRenderer, RenderContext
from src.render.effects.body_glow import BodyGlowRenderer
from src.render.effects.fire_skeleton import FireSkeletonRenderer
from src.render.effects.neon_skeleton import NeonSkeletonRenderer
from src.render.effects.particle_fill import ParticleFillRenderer
from src.render.effects.passthrough import PassthroughRenderer
from src.render.effects.robot_skeleton import RobotSkeletonRenderer
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
            NeonSkeletonRenderer(),
            RobotSkeletonRenderer(),
            FireSkeletonRenderer(),
            BodyGlowRenderer(),
            ParticleFillRenderer(),
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

        Args:
            frame: BGR input frame.

        Returns:
            Composited output frame.
        """
        with self._lock:
            renderer = self.active_renderer

        pose: PoseResult | None = None
        mask: MaskResult | None = None

        if renderer.needs_pose:
            pose = self._pose_estimator.infer(frame)
        if renderer.needs_mask:
            mask = self._segmenter.infer(frame)

        ctx = RenderContext(
            frame=frame,
            pose=pose,
            mask=mask,
            bass_energy=self._bass_energy,
            timestamp=time.monotonic(),
        )

        return renderer.render(ctx)
