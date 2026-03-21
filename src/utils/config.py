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
class EffectsConfig:
    """Effect rendering settings."""
    default: str = "neon_skeleton"


@dataclass
class AudioConfig:
    """Audio capture and bass extraction settings."""
    enabled: bool = True
    device_index: int | None = None
    sample_rate: int = 44100
    chunk_size: int = 1024
    bass_low_hz: int = 20
    bass_high_hz: int = 200
    smoothing_attack: float = 0.3
    smoothing_decay: float = 0.05
    sensitivity: float = 3.0
    noise_gate: float = 500.0


@dataclass
class AppConfig:
    """Top-level application config."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    effects: EffectsConfig = field(default_factory=EffectsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)


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
        effects=EffectsConfig(**raw.get("effects", {})),
        audio=AudioConfig(**raw.get("audio", {})),
    )
