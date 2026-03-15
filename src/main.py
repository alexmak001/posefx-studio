"""Main entry point for posefx-studio.

Usage: python -m src.main --config config/demo.yaml
"""

import argparse
import logging

import cv2
import numpy as np

from src.inference.base import MaskResult, PoseResult
from src.inference.pose_estimator import YOLOPoseEstimator
from src.inference.segmenter import YOLOSegmenter
from src.io.preview import PreviewWindow
from src.io.webcam import WebcamCapture
from src.utils.config import load_config
from src.utils.timing import FPSCounter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# COCO skeleton connections (pairs of keypoint indices)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                   # shoulders
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12),                         # torso
    (11, 12),                                 # hips
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]

KEYPOINT_COLOR = (0, 255, 255)   # cyan
SKELETON_COLOR = (0, 255, 0)     # green
KEYPOINT_RADIUS = 4
SKELETON_THICKNESS = 2
CONFIDENCE_MIN = 0.3
MASK_COLOR = (255, 100, 50)  # blue-ish overlay
MASK_ALPHA = 0.4


def draw_mask(frame: np.ndarray, mask_result: MaskResult) -> None:
    """Draw semi-transparent person mask overlay on a frame.

    Args:
        frame: BGR image to draw on (modified in place).
        mask_result: Segmentation result with combined mask.
    """
    if mask_result.num_people == 0:
        return

    overlay = frame.copy()
    overlay[mask_result.combined_mask > 0] = MASK_COLOR
    cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, dst=frame)


def draw_skeleton(frame: np.ndarray, pose: PoseResult, show_confidence: bool) -> None:
    """Draw skeleton overlay on a frame.

    Args:
        frame: BGR image to draw on (modified in place).
        pose: Pose estimation result.
        show_confidence: Whether to display confidence values near keypoints.
    """
    for person_idx in range(pose.num_people):
        kpts = pose.keypoints[person_idx]     # (17, 2)
        confs = pose.confidences[person_idx]  # (17,)

        # Draw skeleton lines
        for i, j in SKELETON_CONNECTIONS:
            if confs[i] > CONFIDENCE_MIN and confs[j] > CONFIDENCE_MIN:
                pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                cv2.line(frame, pt1, pt2, SKELETON_COLOR, SKELETON_THICKNESS)

        # Draw keypoints
        for k in range(17):
            if confs[k] > CONFIDENCE_MIN:
                x, y = int(kpts[k, 0]), int(kpts[k, 1])
                cv2.circle(frame, (x, y), KEYPOINT_RADIUS, KEYPOINT_COLOR, -1)
                if show_confidence:
                    cv2.putText(
                        frame,
                        f"{confs[k]:.2f}",
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 255, 255),
                        1,
                    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="posefx-studio: real-time body tracking pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


def main() -> None:
    """Run the main pipeline loop."""
    args = parse_args()
    config = load_config(args.config)

    camera = WebcamCapture(config.camera)
    preview = PreviewWindow()
    fps_counter = FPSCounter()

    pose_estimator = YOLOPoseEstimator(config.inference)
    segmenter = YOLOSegmenter(config.inference)

    logger.info("Starting pipeline — press 'q' or ESC to quit")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                logger.error("Failed to read frame from camera")
                break

            fps_counter.tick()

            # Segmentation mask (drawn first, underneath skeleton)
            if config.debug.show_mask:
                mask_result = segmenter.infer(frame)
                draw_mask(frame, mask_result)

            # Pose estimation + skeleton overlay
            if config.debug.show_skeleton:
                pose = pose_estimator.infer(frame)
                draw_skeleton(frame, pose, config.debug.show_keypoint_confidence)

            # FPS overlay
            if config.debug.show_fps:
                cv2.putText(
                    frame,
                    f"FPS: {fps_counter.fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

            preview.show(frame)
            if preview.should_quit():
                break
    finally:
        camera.release()
        preview.destroy()
        logger.info("Pipeline stopped")


if __name__ == "__main__":
    main()
