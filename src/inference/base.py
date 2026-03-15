"""Abstract base classes for inference modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class PoseResult:
    """Result from pose estimation on a single frame.

    Attributes:
        keypoints: Pixel coordinates for N people, 17 COCO keypoints. Shape (N, 17, 2).
        confidences: Per-keypoint confidence scores. Shape (N, 17).
        boxes: Bounding boxes [x1, y1, x2, y2]. Shape (N, 4).
        num_people: Number of detected people.
    """
    keypoints: np.ndarray
    confidences: np.ndarray
    boxes: np.ndarray
    num_people: int


@dataclass
class MaskResult:
    """Result from person segmentation on a single frame.

    Attributes:
        masks: Binary masks per person. Shape (N, H, W).
        combined_mask: Union of all person masks. Shape (H, W).
        num_people: Number of detected people.
    """
    masks: np.ndarray
    combined_mask: np.ndarray
    num_people: int


class BasePoseEstimator(ABC):
    """Abstract interface for pose estimation models."""

    @abstractmethod
    def infer(self, frame: np.ndarray) -> PoseResult:
        """Run pose estimation on a single frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            PoseResult with detected keypoints and bounding boxes.
        """
        ...


class BaseSegmenter(ABC):
    """Abstract interface for person segmentation models."""

    @abstractmethod
    def infer(self, frame: np.ndarray) -> MaskResult:
        """Run segmentation on a single frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            MaskResult with per-person and combined masks.
        """
        ...
