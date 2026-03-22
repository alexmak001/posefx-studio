"""Mode-based engine for managing renderers and inference."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.inference.base import MaskResult, PoseResult
from src.inference.pose_estimator import YOLOPoseEstimator
from src.inference.segmenter import YOLOSegmenter
from src.io.photo_capture import PhotoCapture
from src.io.recorder import VideoRecorder
from src.render.base import BaseRenderer, RenderContext
from src.render.effects.digital_rain import DigitalRainRenderer
from src.render.effects.energy_aura import EnergyAuraRenderer
from src.render.effects.glitch_body import GlitchBodyRenderer
from src.render.effects.halo_wings import HaloWingsRenderer
from src.render.effects.motion_trails import MotionTrailsRenderer
from src.render.effects.neon_wireframe import NeonWireframeRenderer
from src.render.effects.particle_dissolve import ParticleDissolveRenderer
from src.render.effects.passthrough import PassthroughRenderer
from src.render.effects.shadow_clones import ShadowClonesRenderer
from src.render.effects.sprite_puppet import SpritePuppetRenderer
from src.utils.config import AppConfig

if TYPE_CHECKING:
    from src.audio.capture import AudioCapture

logger = logging.getLogger(__name__)
FLASH_DURATION_SECONDS = 0.2
_HAS_FFMPEG = shutil.which("ffmpeg") is not None


@dataclass(frozen=True)
class CaptureState:
    """Snapshot of capture-related runtime state for preview and control."""

    is_recording: bool
    auto_capture_enabled: bool
    countdown_value: int | None
    flash_active: bool
    person_present: bool
    last_photo_path: str | None
    last_raw_photo_path: str | None
    last_recording_path: str | None
    last_raw_recording_path: str | None


class PartyEngine:
    """Manages the active renderer, inference models, and frame processing.

    Only runs inference models that the current renderer actually needs,
    skipping unnecessary work for better performance.

    Args:
        config: Application configuration.
        platform: Detected platform string ('mac', 'jetson', 'cpu').
    """

    def __init__(self, config: AppConfig, platform: str,
                 audio_capture: AudioCapture | None = None) -> None:
        self._config = config
        self._platform = platform
        self._lock = threading.Lock()
        self._bass_energy = 0.0
        self._audio_capture = audio_capture
        self._photo_capture = PhotoCapture(config.capture.photo_quality)
        self._edited_video_recorder = VideoRecorder()
        self._raw_video_recorder = VideoRecorder()
        self._photo_dir = Path(config.capture.photo_dir)
        self._recording_dir = Path(config.capture.recording_dir)
        self._edited_photo_dir = self._photo_dir / config.capture.edited_subdir
        self._raw_photo_dir = self._photo_dir / config.capture.raw_subdir
        self._edited_recording_dir = self._recording_dir / config.capture.edited_subdir
        self._raw_recording_dir = self._recording_dir / config.capture.raw_subdir
        self._photo_mode = config.capture.photo_mode
        self._record_mode = config.capture.record_mode
        self._countdown_deadline: float | None = None
        self._flash_until = 0.0
        self._auto_capture_enabled = False
        self._auto_capture_due_at: float | None = None
        self._person_present = False
        self._last_photo_path: Path | None = None
        self._last_raw_photo_path: Path | None = None
        self._last_recording_path: Path | None = None
        self._last_raw_recording_path: Path | None = None
        self._latest_frame: np.ndarray | None = None
        self._fps = 0.0

        # Load inference models
        self._pose_estimator = YOLOPoseEstimator(config.inference)
        self._segmenter = YOLOSegmenter(config.inference)

        # Register all renderers in display order
        self._renderers: list[BaseRenderer] = [
            NeonWireframeRenderer(),
            EnergyAuraRenderer(),
            MotionTrailsRenderer(),
            GlitchBodyRenderer(),
            DigitalRainRenderer(),
            ShadowClonesRenderer(),
            ParticleDissolveRenderer(),
            HaloWingsRenderer(),
            SpritePuppetRenderer(),
            PassthroughRenderer(),
        ]
        self._renderer_map: dict[str, BaseRenderer] = {
            r.name: r for r in self._renderers
        }

        # Set default renderer
        default_name = getattr(config, "effects", None)
        if default_name and hasattr(default_name, "default"):
            # Try to find matching renderer
            for r in self._renderers:
                if r.name.lower().replace(" ", "_") == default_name.default:
                    self._active_idx = self._renderers.index(r)
                    break
            else:
                self._active_idx = 0
        else:
            self._active_idx = 0

        logger.info(
            "PartyEngine initialized with %d effects, active: %s",
            len(self._renderers),
            self.active_renderer.name,
        )

    @property
    def active_renderer(self) -> BaseRenderer:
        """The currently active renderer."""
        return self._renderers[self._active_idx]

    def set_renderer(self, name: str) -> None:
        """Switch active renderer by name. Thread-safe.

        Args:
            name: Human-readable effect name.
        """
        with self._lock:
            if name in self._renderer_map:
                self._active_idx = self._renderers.index(self._renderer_map[name])
                logger.info("Switched effect to: %s", name)
            else:
                logger.warning("Unknown effect: %s", name)

    def next_renderer(self) -> str:
        """Switch to the next renderer in the list. Returns the new name."""
        with self._lock:
            self._active_idx = (self._active_idx + 1) % len(self._renderers)
            name = self.active_renderer.name
        logger.info("Switched effect to: %s", name)
        return name

    def prev_renderer(self) -> str:
        """Switch to the previous renderer in the list. Returns the new name."""
        with self._lock:
            self._active_idx = (self._active_idx - 1) % len(self._renderers)
            name = self.active_renderer.name
        logger.info("Switched effect to: %s", name)
        return name

    def get_renderer_names(self) -> list[str]:
        """List all available effect names."""
        return [r.name for r in self._renderers]

    def trigger_photo(self, seconds: int | None = None) -> bool:
        """Start a photo countdown.

        Args:
            seconds: Optional countdown override.

        Returns:
            True if a new countdown was armed, False if one was already active.
        """
        countdown = max(1, int(seconds if seconds is not None else self._config.capture.countdown_seconds))
        with self._lock:
            if self._countdown_deadline is not None:
                logger.info("Photo countdown already active")
                return False
            self._countdown_deadline = time.monotonic() + countdown
        logger.info("Photo countdown started (%d seconds)", countdown)
        return True

    def start_recording(self, fps: float, resolution: tuple[int, int]) -> Path:
        """Start recording composited output frames.

        Args:
            fps: Output frames per second.
            resolution: Frame size as (width, height).

        Returns:
            Path to the recording file being written.
        """
        stem = f"recording_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time_ns() % 1_000_000_000):09d}"
        with self._lock:
            edited_path: Path | None = None
            raw_path: Path | None = None

            try:
                if self._record_mode in {"edited", "both"}:
                    edited_path = self._edited_video_recorder.start(
                        self._edited_recording_dir,
                        fps,
                        resolution,
                        stem=stem,
                    )
                if self._record_mode in {"raw", "both"}:
                    raw_path = self._raw_video_recorder.start(
                        self._raw_recording_dir,
                        fps,
                        resolution,
                        stem=stem,
                    )
            except Exception:
                self._edited_video_recorder.stop()
                self._raw_video_recorder.stop()
                raise

            self._last_recording_path = edited_path
            self._last_raw_recording_path = raw_path
            self._recording_stem = stem

        # Start audio recording alongside video
        if self._audio_capture is not None:
            self._audio_capture.start_recording()

        return edited_path or raw_path

    def stop_recording(self) -> Path | None:
        """Stop the active recording, mux audio if available."""
        with self._lock:
            edited_path = self._edited_video_recorder.stop()
            raw_path = self._raw_video_recorder.stop()
            if edited_path is not None:
                self._last_recording_path = edited_path
            if raw_path is not None:
                self._last_raw_recording_path = raw_path

        # Stop audio recording and mux into the video files
        if self._audio_capture is not None:
            wav_dir = self._recording_dir / "tmp"
            wav_path = wav_dir / f"{getattr(self, '_recording_stem', 'audio')}.wav"
            saved_wav = self._audio_capture.stop_recording(wav_path)
            if saved_wav and _HAS_FFMPEG:
                for video_path in (edited_path, raw_path):
                    if video_path is not None:
                        self._mux_audio(video_path, saved_wav)
                # Clean up temp WAV
                try:
                    saved_wav.unlink(missing_ok=True)
                    if wav_dir.exists() and not any(wav_dir.iterdir()):
                        wav_dir.rmdir()
                except Exception:
                    pass

        return edited_path or raw_path

    @staticmethod
    def _mux_audio(video_path: Path, audio_path: Path) -> None:
        """Mux audio into an existing video file using ffmpeg.

        Replaces the video file in-place.

        Args:
            video_path: Path to the MP4 video.
            audio_path: Path to the WAV audio.
        """
        tmp_out = video_path.with_suffix(".tmp.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-loglevel", "warning",
            str(tmp_out),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=60)
            tmp_out.replace(video_path)
            logger.info("Muxed audio into %s", video_path)
        except Exception:
            logger.warning("Failed to mux audio into %s", video_path, exc_info=True)
            try:
                tmp_out.unlink(missing_ok=True)
            except Exception:
                pass

    def toggle_auto_capture(self) -> bool:
        """Toggle periodic auto-capture mode.

        Returns:
            The new auto-capture state.
        """
        with self._lock:
            self._auto_capture_enabled = not self._auto_capture_enabled
            self._auto_capture_due_at = None
            enabled = self._auto_capture_enabled
        logger.info("Auto-capture: %s", "ON" if enabled else "OFF")
        return enabled

    def get_capture_state(self) -> CaptureState:
        """Return a snapshot of capture status for UI overlays."""
        now = time.monotonic()
        with self._lock:
            countdown_value = None
            if self._countdown_deadline is not None:
                countdown_value = max(1, math.ceil(self._countdown_deadline - now))

            return CaptureState(
                is_recording=(
                    self._edited_video_recorder.is_recording
                    or self._raw_video_recorder.is_recording
                ),
                auto_capture_enabled=self._auto_capture_enabled,
                countdown_value=countdown_value,
                flash_active=now < self._flash_until,
                person_present=self._person_present,
                last_photo_path=str(self._last_photo_path) if self._last_photo_path else None,
                last_raw_photo_path=str(self._last_raw_photo_path) if self._last_raw_photo_path else None,
                last_recording_path=str(self._last_recording_path) if self._last_recording_path else None,
                last_raw_recording_path=(
                    str(self._last_raw_recording_path)
                    if self._last_raw_recording_path
                    else None
                ),
            )

    def set_bass_energy(self, energy: float) -> None:
        """Update bass energy from audio thread.

        Args:
            energy: Bass energy level 0.0-1.0.
        """
        self._bass_energy = max(0.0, min(1.0, energy))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run inference and render for one frame.

        Only runs the inference models that the current renderer needs.
        Downscales the frame for inference if inference_scale < 1.0,
        then scales results back to full resolution for rendering.

        Args:
            frame: BGR input frame.

        Returns:
            Composited output frame.
        """
        with self._lock:
            renderer = self.active_renderer
            auto_capture_enabled = self._auto_capture_enabled

        pose: PoseResult | None = None
        mask: MaskResult | None = None
        scale = self._config.inference.inference_scale
        needs_presence_pose = auto_capture_enabled and not renderer.needs_pose and not renderer.needs_mask
        needs_pose = renderer.needs_pose or needs_presence_pose
        needs_mask = renderer.needs_mask
        needs_inference = needs_pose or needs_mask

        # Downscale for inference if needed
        if needs_inference and 0 < scale < 1.0:
            small = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame
            scale = 1.0

        if needs_pose:
            pose = self._pose_estimator.infer(small)
            if scale < 1.0 and pose.num_people > 0:
                inv = 1.0 / scale
                pose = PoseResult(
                    keypoints=pose.keypoints * inv,
                    confidences=pose.confidences,
                    boxes=pose.boxes * inv,
                    num_people=pose.num_people,
                )

        if needs_mask:
            mask = self._segmenter.infer(small)
            if scale < 1.0 and mask.num_people > 0:
                h, w = frame.shape[:2]
                full_masks = np.zeros((mask.num_people, h, w), dtype=np.uint8)
                for i in range(mask.num_people):
                    full_masks[i] = cv2.resize(
                        mask.masks[i], (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    full_masks[i] = (full_masks[i] > 0).astype(np.uint8)
                mask = MaskResult(
                    masks=full_masks,
                    combined_mask=np.any(full_masks, axis=0).astype(np.uint8),
                    num_people=mask.num_people,
                )

        ctx = RenderContext(
            frame=frame,
            pose=pose,
            mask=mask,
            bass_energy=self._bass_energy,
            timestamp=time.monotonic(),
        )

        output = renderer.render(ctx)
        timestamp = ctx.timestamp
        if pose is not None:
            person_present = pose.num_people > 0
        elif mask is not None:
            person_present = mask.num_people > 0
        else:
            person_present = False

        # Determine recording state and capture logic under the lock,
        # but perform I/O (video write, photo encode) outside the lock
        # to avoid blocking other threads (audio, future web server).
        record_edited = False
        record_raw = False

        with self._lock:
            self._person_present = person_present

            record_edited = self._edited_video_recorder.is_recording
            record_raw = self._raw_video_recorder.is_recording

            should_capture_manual = (
                self._countdown_deadline is not None and timestamp >= self._countdown_deadline
            )
            if should_capture_manual:
                self._countdown_deadline = None

            should_capture_auto = False
            if not should_capture_manual and self._auto_capture_enabled:
                if person_present:
                    if self._auto_capture_due_at is None:
                        self._auto_capture_due_at = (
                            timestamp + self._config.capture.auto_capture_interval
                        )
                    elif (
                        self._countdown_deadline is None
                        and timestamp >= self._auto_capture_due_at
                    ):
                        should_capture_auto = True
                        self._auto_capture_due_at = (
                            timestamp + self._config.capture.auto_capture_interval
                        )
                else:
                    self._auto_capture_due_at = None
            else:
                self._auto_capture_due_at = None

        # Video recording I/O outside the lock
        if record_edited:
            self._edited_video_recorder.add_frame(output)
        if record_raw:
            self._raw_video_recorder.add_frame(frame)

        if should_capture_manual or should_capture_auto:
            stem = f"photo_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time_ns() % 1_000_000_000):09d}"
            try:
                edited_photo_path: Path | None = None
                raw_photo_path: Path | None = None
                if self._photo_mode in {"edited", "both"}:
                    edited_photo_path = self._photo_capture.capture(
                        output,
                        self._edited_photo_dir,
                        stem=stem,
                    )
                if self._photo_mode in {"raw", "both"}:
                    raw_photo_path = self._photo_capture.capture(
                        frame,
                        self._raw_photo_dir,
                        stem=stem,
                    )
            except Exception:
                logger.exception("Photo capture failed")
            else:
                with self._lock:
                    self._last_photo_path = edited_photo_path
                    self._last_raw_photo_path = raw_photo_path
                    self._flash_until = timestamp + FLASH_DURATION_SECONDS

        return output

    def set_latest_frame(self, frame: np.ndarray, fps: float) -> None:
        """Store the latest composited frame for MJPEG streaming.

        Called from the main loop after process_frame + HUD overlays.

        Args:
            frame: The composited BGR frame (with HUD).
            fps: Current FPS reading.
        """
        self._latest_frame = frame
        self._fps = fps

    def get_latest_jpeg(self, quality: int = 60) -> bytes | None:
        """Encode the latest frame as JPEG for MJPEG streaming.

        Args:
            quality: JPEG quality 0-100.

        Returns:
            JPEG bytes, or None if no frame is available yet.
        """
        frame = self._latest_frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return buf.tobytes()

    def get_state(self) -> dict:
        """Return current engine state for WebSocket broadcast.

        Returns:
            Dict with effect, fps, recording, bass, auto_capture, person info.
        """
        capture = self.get_capture_state()
        return {
            "effect": self.active_renderer.name,
            "effects": self.get_renderer_names(),
            "fps": round(self._fps, 1),
            "bass_energy": round(self._bass_energy, 3),
            "is_recording": capture.is_recording,
            "auto_capture": capture.auto_capture_enabled,
            "countdown": capture.countdown_value,
            "person_present": capture.person_present,
        }

    def close(self) -> Path | None:
        """Release capture resources before shutdown.

        Returns:
            Finalized recording path if a recording was active, else None.
        """
        return self.stop_recording()
