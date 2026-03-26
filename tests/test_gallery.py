"""Tests for gallery metadata scanning."""

from pathlib import Path

from src.web.server import _ensure_browser_safe_video, _gallery_cache_path, _is_browser_safe_video, scan_gallery_items


def test_scan_gallery_items_includes_type_and_variant(tmp_path):
    """Gallery scan should classify photos/recordings and raw/edited variants."""
    photo_edited = tmp_path / "photos" / "edited"
    photo_raw = tmp_path / "photos" / "raw"
    recording_edited = tmp_path / "recordings" / "edited"
    recording_raw = tmp_path / "recordings" / "raw"

    for directory in (photo_edited, photo_raw, recording_edited, recording_raw):
        directory.mkdir(parents=True, exist_ok=True)

    (photo_edited / "edited_1.jpg").write_bytes(b"jpg")
    (photo_raw / "raw_1.jpg").write_bytes(b"jpg")
    (recording_edited / "edited_1.mp4").write_bytes(b"mp4")
    (recording_raw / "raw_1.mp4").write_bytes(b"mp4")

    items = scan_gallery_items(
        photo_dir=tmp_path / "photos",
        recording_dir=tmp_path / "recordings",
    )

    by_name = {item["name"]: item for item in items}

    assert by_name["edited_1.jpg"]["type"] == "photo"
    assert by_name["edited_1.jpg"]["variant"] == "edited"
    assert by_name["raw_1.jpg"]["type"] == "photo"
    assert by_name["raw_1.jpg"]["variant"] == "raw"
    assert by_name["edited_1.mp4"]["type"] == "recording"
    assert by_name["edited_1.mp4"]["variant"] == "edited"
    assert by_name["raw_1.mp4"]["type"] == "recording"
    assert by_name["raw_1.mp4"]["variant"] == "raw"


def test_scan_gallery_items_skips_hidden_and_unknown_extensions(tmp_path):
    """Gallery scan should ignore hidden files and unsupported extensions."""
    photo_dir = tmp_path / "photos" / "edited"
    photo_dir.mkdir(parents=True, exist_ok=True)

    (photo_dir / ".hidden.jpg").write_bytes(b"jpg")
    (photo_dir / "notes.txt").write_text("nope")
    (photo_dir / "visible.jpg").write_bytes(b"jpg")

    items = scan_gallery_items(
        photo_dir=tmp_path / "photos",
        recording_dir=tmp_path / "recordings",
    )

    names = [item["name"] for item in items]

    assert names == ["visible.jpg"]


def test_is_browser_safe_video_accepts_h264_yuv420p(monkeypatch, tmp_path):
    """Gallery playback should accept H.264/yuv420p assets as-is."""
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")

    monkeypatch.setattr(
        "src.web.server._probe_video_stream",
        lambda path: {"codec_name": "h264", "pix_fmt": "yuv420p"},
    )

    assert _is_browser_safe_video(video_path) is True


def test_ensure_browser_safe_video_transcodes_incompatible_assets(monkeypatch, tmp_path):
    """Gallery playback should cache a transcoded MP4 for incompatible videos."""
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")

    monkeypatch.setattr("src.web.server._HAS_FFMPEG", True)
    monkeypatch.setattr("src.web.server._is_browser_safe_video", lambda path: False)

    commands = []

    def fake_run(cmd, check, timeout):
        commands.append((cmd, check, timeout))
        cached_path = _gallery_cache_path(video_path, ".mp4")
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path.write_bytes(b"transcoded")

    monkeypatch.setattr("subprocess.run", fake_run)

    playable_path = _ensure_browser_safe_video(video_path)

    assert playable_path.suffix == ".mp4"
    assert playable_path != video_path
    assert playable_path.read_bytes() == b"transcoded"
    assert commands
