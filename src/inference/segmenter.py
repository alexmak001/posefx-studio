"""YOLOv8-seg implementation of person segmentation."""

import logging

import torch
import cv2
import numpy as np
from ultralytics import YOLO

from src.inference.base import BaseSegmenter, MaskResult
from src.utils.config import InferenceConfig
from src.inference.pose_estimator import _resolve_device

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  # COCO class 0 = person


class YOLOSegmenter(BaseSegmenter):
    """Person segmenter using YOLOv8-seg.

    Args:
        config: Inference configuration with model path, device, and confidence threshold.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self._device = _resolve_device(config.device)
        logger.info("Loading seg model: %s on device: %s", config.seg_model, self._device)
        self._model = YOLO(config.seg_model)
        self._conf = config.confidence_threshold

    def infer(self, frame: np.ndarray) -> MaskResult:
        """Run YOLOv8-seg on a frame, filtering to person class only.

        Args:
            frame: BGR image as numpy array.

        Returns:
            MaskResult with per-person and combined masks.
        """
        h, w = frame.shape[:2]

        results = self._model(
            frame,
            device=self._device,
            conf=self._conf,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        result = results[0]

        if result.masks is None or len(result.masks) == 0:
            return MaskResult(
                masks=np.empty((0, h, w), dtype=np.uint8),
                combined_mask=np.zeros((h, w), dtype=np.uint8),
                num_people=0,
            )

        # Get mask data and resize to frame dimensions
        raw_masks = result.masks.data.cpu().numpy()  # (N, mask_h, mask_w)
        n = raw_masks.shape[0]
        masks = np.zeros((n, h, w), dtype=np.uint8)

        for i in range(n):
            resized = cv2.resize(
                raw_masks[i], (w, h), interpolation=cv2.INTER_LINEAR
            )
            masks[i] = (resized > 0.5).astype(np.uint8)

        combined_mask = np.any(masks, axis=0).astype(np.uint8)

        return MaskResult(
            masks=masks,
            combined_mask=combined_mask,
            num_people=n,
        )
