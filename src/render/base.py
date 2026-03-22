"""Base classes for the rendering / effects system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.inference.base import MaskResult, PoseResult


@dataclass
class RenderContext:
    """Everything a renderer needs to draw one frame.

    Attributes:
        frame: Original camera frame (H, W, 3) BGR.
        pose: Pose estimation result, or None if not available.
        mask: Segmentation result, or None if not available.
        bass_energy: Audio bass energy 0.0-1.0 (0.0 if no audio).
        timestamp: time.monotonic() of frame capture.
    """

    frame: np.ndarray
    pose: PoseResult | None
    mask: MaskResult | None
    bass_energy: float
    timestamp: float
    avatar: np.ndarray | None = None
    puppet_opacity: float = 0.7


class BaseRenderer(ABC):
    """Abstract base class for visual effect renderers."""

    @abstractmethod
    def render(self, ctx: RenderContext) -> np.ndarray:
        """Compose an output frame from the render context.

        Args:
            ctx: Current frame data including pose, mask, and audio.

        Returns:
            Composited frame (H, W, 3) BGR.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable effect name for the UI."""
        ...

    @property
    def needs_pose(self) -> bool:
        """Whether this renderer requires pose inference."""
        return False

    @property
    def needs_mask(self) -> bool:
        """Whether this renderer requires segmentation inference."""
        return False
