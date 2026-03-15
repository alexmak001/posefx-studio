"""Tests for config loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import AppConfig, CameraConfig, DebugConfig, InferenceConfig, load_config


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


def test_load_config_file_not_found():
    """Test that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")
