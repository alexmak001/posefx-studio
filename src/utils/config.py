"""YAML config loader with typed access."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)
CaptureMode = Literal["raw", "edited", "both"]
VALID_CAPTURE_MODES = {"raw", "edited", "both"}


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
    inference_scale: float = 0.5


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
class CaptureConfig:
    """Photo and video output settings."""
    photo_dir: str = "data/photos"
    recording_dir: str = "data/recordings"
    edited_subdir: str = "edited"
    raw_subdir: str = "raw"
    photo_mode: CaptureMode = "both"
    record_mode: CaptureMode = "both"
    photo_quality: int = 95
    auto_capture_interval: int = 15
    countdown_seconds: int = 3

    def __post_init__(self) -> None:
        """Validate capture mode settings loaded from config."""
        if self.photo_mode not in VALID_CAPTURE_MODES:
            raise ValueError(
                f"Invalid capture.photo_mode: {self.photo_mode!r}. "
                f"Expected one of {sorted(VALID_CAPTURE_MODES)}"
            )
        if self.record_mode not in VALID_CAPTURE_MODES:
            raise ValueError(
                f"Invalid capture.record_mode: {self.record_mode!r}. "
                f"Expected one of {sorted(VALID_CAPTURE_MODES)}"
            )


@dataclass
class WebConfig:
    """Web server settings."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    stream_quality: int = 60
    stream_max_fps: int = 15
    qr_visible_on_startup: bool = True


@dataclass
class YouTubeConfig:
    """YouTube DJ settings."""
    enabled: bool = True
    max_search_results: int = 10
    max_duration_seconds: int = 600


@dataclass
class AIConfig:
    """AI Lab settings (Replicate API)."""
    enabled: bool = True
    results_dir: str = "data/ai_results"
    uploads_dir: str = "data/uploads"
    max_upload_size_mb: int = 50
    replicate_model_face_swap: str = "lucataco/faceswap"
    replicate_model_video_swap: str = "xiankgx/face-swap"
    replicate_model_edit: str = "black-forest-labs/flux-kontext-pro"
    replicate_model_video_gen: str = "minimax/video-01-live"


@dataclass
class AppConfig:
    """Top-level application config."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    effects: EffectsConfig = field(default_factory=EffectsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    web: WebConfig = field(default_factory=WebConfig)
    youtube: YouTubeConfig = field(default_factory=YouTubeConfig)
    ai: AIConfig = field(default_factory=AIConfig)


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
        capture=CaptureConfig(**raw.get("capture", {})),
        web=WebConfig(**raw.get("web", {})),
        youtube=YouTubeConfig(**raw.get("youtube", {})),
        ai=AIConfig(**raw.get("ai", {})),
    )
