"""YAML config loader with typed access."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera capture settings."""
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class InferenceConfig:
    """Model inference settings."""
    pose_model: str = "yolov8n-pose.pt"
    seg_model: str = "yolov8n-seg.pt"
    device: str = "mps"
    confidence_threshold: float = 0.5


@dataclass
class DebugConfig:
    """Debug overlay settings."""
    show_skeleton: bool = True
    show_mask: bool = True
    show_fps: bool = True
    show_keypoint_confidence: bool = True


@dataclass
class AppConfig:
    """Top-level application config."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load application config from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    logger.info("Loaded config from %s", path)

    return AppConfig(
        camera=CameraConfig(**raw.get("camera", {})),
        inference=InferenceConfig(**raw.get("inference", {})),
        debug=DebugConfig(**raw.get("debug", {})),
    )
