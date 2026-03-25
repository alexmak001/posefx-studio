"""Shared rendering utilities for effects."""

import numpy as np
import cv2

from src.inference.base import PoseResult


def get_head_mask(pose: PoseResult, frame_shape: tuple, scale: float = 1.4) -> np.ndarray:
    """Create a circular mask around the head region using pose keypoints.

    Uses nose (0), left_eye (1), right_eye (2), left_ear (3), right_ear (4)
    to estimate head center and radius. Returns a binary mask where
    1 = head region, 0 = elsewhere.

    Args:
        pose: Pose estimation result.
        frame_shape: (H, W, ...) shape of the frame.
        scale: How much larger than the detected head to protect.

    Returns:
        Binary uint8 mask with head regions marked as 1.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if pose is None or pose.num_people == 0:
        return mask

    head_indices = [0, 1, 2, 3, 4]  # nose, eyes, ears

    for person_idx in range(pose.num_people):
        kpts = pose.keypoints[person_idx]
        confs = pose.confidences[person_idx]

        visible = []
        for idx in head_indices:
            if confs[idx] > 0.3:
                visible.append(kpts[idx])

        if len(visible) < 2:
            continue

        points = np.array(visible)
        center_x = float(np.mean(points[:, 0]))
        center_y = float(np.mean(points[:, 1]))

        dists = np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2)
        radius = float(np.max(dists)) * scale
        radius = max(radius, 30.0)

        cv2.circle(mask, (int(center_x), int(center_y)), int(radius), 1, -1)

    return mask


def composite_head(output: np.ndarray, original: np.ndarray,
                   pose: PoseResult | None, frame_shape: tuple,
                   head_scale: float = 1.4) -> np.ndarray:
    """Blend the original face back into an effect frame with soft feathered edges.

    Args:
        output: The effect-rendered frame.
        original: The original camera frame.
        pose: Pose result (may be None).
        frame_shape: Frame shape tuple.
        head_scale: Scale factor for head mask.

    Returns:
        Frame with real face composited back in with smooth blending.
    """
    if pose is None or pose.num_people == 0:
        return output

    head = get_head_mask(pose, frame_shape, scale=head_scale)
    if not np.any(head):
        return output

    # Feather the mask edges with Gaussian blur for smooth blending
    head_255 = (head * 255).astype(np.uint8)
    feather_size = 31
    soft_mask = cv2.GaussianBlur(head_255, (feather_size, feather_size), 0)
    alpha = soft_mask.astype(np.float32) / 255.0
    alpha_3ch = alpha[:, :, np.newaxis]

    # Smooth blend: face at full opacity in center, fades into effect at edges
    result = (original.astype(np.float32) * alpha_3ch +
              output.astype(np.float32) * (1.0 - alpha_3ch))
    return result.astype(np.uint8)
