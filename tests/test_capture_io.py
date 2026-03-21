"""Tests for photo and video capture helpers."""

from pathlib import Path

import cv2
import numpy as np

from src.io.photo_capture import PhotoCapture
from src.io.recorder import VideoRecorder


def test_photo_capture_saves_jpeg(monkeypatch, tmp_path):
    """PhotoCapture writes a JPEG to the configured directory."""
    written = {}

    def fake_imwrite(path, frame, params):
        written["path"] = path
        written["shape"] = frame.shape
        written["params"] = params
        return True

    monkeypatch.setattr("cv2.imwrite", fake_imwrite)
    capture = PhotoCapture(quality=88)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    saved_path = capture.capture(frame, tmp_path / "photos", stem="photo_test")

    assert saved_path.name == "photo_test.jpg"
    assert saved_path.suffix == ".jpg"
    assert Path(written["path"]) == saved_path
    assert written["shape"] == frame.shape
    assert written["params"] == [cv2.IMWRITE_JPEG_QUALITY, 88]


def test_video_recorder_start_add_frame_stop(monkeypatch, tmp_path):
    """VideoRecorder opens a writer, records frames, and returns the saved path."""

    class FakeWriter:
        def __init__(self, path, fourcc, fps, resolution):
            self.path = path
            self.fourcc = fourcc
            self.fps = fps
            self.resolution = resolution
            self.frames = []
            self.released = False

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            self.released = True

    writers = []

    def fake_writer(path, fourcc, fps, resolution):
        writer = FakeWriter(path, fourcc, fps, resolution)
        writers.append(writer)
        return writer

    monkeypatch.setattr("cv2.VideoWriter", fake_writer)
    monkeypatch.setattr("cv2.VideoWriter_fourcc", lambda *args: 1234)

    recorder = VideoRecorder()
    output_dir = tmp_path / "recordings"
    path = recorder.start(output_dir, fps=24.0, resolution=(16, 12), stem="recording_test")
    recorder.add_frame(np.zeros((12, 16, 3), dtype=np.uint8))
    saved_path = recorder.stop()

    assert path == saved_path
    assert saved_path is not None
    assert saved_path.parent == output_dir.resolve()
    assert saved_path.name == "recording_test.mp4"
    assert saved_path.suffix == ".mp4"
    assert len(writers) == 1
    assert writers[0].fps == 24.0
    assert writers[0].resolution == (16, 12)
    assert len(writers[0].frames) == 1
    assert writers[0].released is True
