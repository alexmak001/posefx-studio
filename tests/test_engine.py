"""Tests for engine helpers."""

from pathlib import Path

from src.engine import PartyEngine


def test_mux_audio_reencodes_to_browser_safe_mp4(monkeypatch, tmp_path):
    """Muxing should re-encode video to H.264 for mobile browser playback."""
    commands = []
    video_path = tmp_path / "clip.mp4"
    audio_path = tmp_path / "clip.wav"
    tmp_out = tmp_path / "clip.tmp.mp4"
    video_path.write_bytes(b"original-video")
    audio_path.write_bytes(b"audio")

    def fake_run(cmd, check, timeout):
        commands.append((cmd, check, timeout))
        tmp_out.write_bytes(b"muxed-video")

    monkeypatch.setattr("subprocess.run", fake_run)

    PartyEngine._mux_audio(video_path, audio_path)

    assert len(commands) == 1
    cmd, check, timeout = commands[0]
    assert check is True
    assert timeout == 60
    assert cmd[:6] == [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
    ]
    assert "-c:v" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert "-pix_fmt" in cmd
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert "-movflags" in cmd
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"
    assert "-c:a" in cmd
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert video_path.read_bytes() == b"muxed-video"
    assert not tmp_out.exists()
