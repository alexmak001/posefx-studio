"""Tests for config loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import CaptureConfig, load_config


def test_load_config_from_file():
    """Test loading a valid config file."""
    config_data = {
        "camera": {"device_id": 1, "width": 640, "height": 480, "fps": 60},
        "inference": {
            "pose_model": "yolov8s-pose.pt",
            "seg_model": "yolov8s-seg.pt",
            "device": "cuda",
            "confidence_threshold": 0.7,
        },
        "debug": {
            "show_skeleton": False,
            "show_mask": False,
            "show_fps": True,
            "show_keypoint_confidence": False,
        },
        "capture": {
            "photo_dir": "data/custom-photos",
            "recording_dir": "data/custom-recordings",
            "edited_subdir": "fx",
            "raw_subdir": "clean",
            "photo_mode": "raw",
            "record_mode": "edited",
            "photo_quality": 90,
            "auto_capture_interval": 10,
            "countdown_seconds": 5,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        tmp_path = f.name

    config = load_config(tmp_path)
    Path(tmp_path).unlink()

    assert config.camera.device_id == 1
    assert config.camera.width == 640
    assert config.inference.device == "cuda"
    assert config.inference.confidence_threshold == 0.7
    assert config.debug.show_skeleton is False
    assert config.capture.photo_dir == "data/custom-photos"
    assert config.capture.recording_dir == "data/custom-recordings"
    assert config.capture.edited_subdir == "fx"
    assert config.capture.raw_subdir == "clean"
    assert config.capture.photo_mode == "raw"
    assert config.capture.record_mode == "edited"
    assert config.capture.photo_quality == 90


def test_load_config_defaults():
    """Test that missing sections get default values."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({}, f)
        tmp_path = f.name

    config = load_config(tmp_path)
    Path(tmp_path).unlink()

    assert config.camera.width == 1280
    assert config.inference.device == "mps"
    assert config.debug.show_fps is True
    assert config.capture == CaptureConfig()


def test_load_config_file_not_found():
    """Test that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_load_config_invalid_capture_mode():
    """Invalid capture mode values should fail fast."""
    config_data = {
        "capture": {
            "photo_mode": "invalid",
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        tmp_path = f.name

    with pytest.raises(ValueError):
        load_config(tmp_path)

    Path(tmp_path).unlink()
