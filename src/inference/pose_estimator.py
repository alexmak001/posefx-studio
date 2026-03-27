"""YOLOv8-pose implementation of pose estimation."""

import logging

import torch
import numpy as np
from ultralytics import YOLO

from src.inference.base import BasePoseEstimator, PoseResult
from src.utils.config import InferenceConfig

logger = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    """Return the best available device, falling back to cpu if requested is unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to cpu")
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available — falling back to cpu")
        return "cpu"
    return requested


class YOLOPoseEstimator(BasePoseEstimator):
    """Pose estimator using YOLOv8-pose.

    Args:
        config: Inference configuration with model path, device, and confidence threshold.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self._device = _resolve_device(config.device)
        logger.info("Loading pose model: %s on device: %s", config.pose_model, self._device)
        self._model = YOLO(config.pose_model)
        self._conf = config.confidence_threshold

    def infer(self, frame: np.ndarray) -> PoseResult:
        """Run YOLOv8-pose on a frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            PoseResult with detected keypoints and bounding boxes.
        """
        results = self._model(
            frame,
            device=self._device,
            conf=self._conf,
            verbose=False,
        )
        result = results[0]

        if result.keypoints is None or len(result.keypoints) == 0:
            h, w = frame.shape[:2]
            return PoseResult(
                keypoints=np.empty((0, 17, 2), dtype=np.float32),
                confidences=np.empty((0, 17), dtype=np.float32),
                boxes=np.empty((0, 4), dtype=np.float32),
                num_people=0,
            )

        kpts = result.keypoints
        keypoints_xy = kpts.xy.cpu().numpy()           # (N, 17, 2)
        keypoints_conf = kpts.conf.cpu().numpy()        # (N, 17)
        boxes = result.boxes.xyxy.cpu().numpy()         # (N, 4)

        return PoseResult(
            keypoints=keypoints_xy,
            confidences=keypoints_conf,
            boxes=boxes,
            num_people=len(keypoints_xy),
        )
