"""Mode-based engine for managing renderers and inference."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import math
import shutil
import subprocess
import sys
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
from src.render.effects.bass_pulse import BassPulseRenderer
from src.render.effects.digital_rain import DigitalRainRenderer
from src.render.effects.energy_aura import EnergyAuraRenderer
from src.render.effects.glitch_body import GlitchBodyRenderer
from src.render.effects.halo_wings import HaloWingsRenderer
from src.render.effects.motion_trails import MotionTrailsRenderer
from src.render.effects.neon_wireframe import NeonWireframeRenderer
from src.render.effects.particle_dissolve import ParticleDissolveRenderer
from src.render.effects.passthrough import PassthroughRenderer
from src.render.effects.snowfall_custom import SnowfallCustomRenderer
from src.render.effects.sprite_puppet import SpritePuppetRenderer
from src.utils.config import AppConfig
from src.utils.qr import build_qr_png

if TYPE_CHECKING:
    from src.audio.capture import AudioCapture

logger = logging.getLogger(__name__)
FLASH_DURATION_SECONDS = 0.2
_HAS_FFMPEG = shutil.which("ffmpeg") is not None
QR_LABEL_TEXT = "SCAN TO PARTY"
QR_MARGIN = 10
QR_PADDING = 6
QR_LABEL_HEIGHT = 16
QR_MIN_SIZE = 52
QR_MAX_SIZE = 90
QR_PANEL_COLOR = (8, 8, 8)
QR_PANEL_ALPHA = 0.82
QR_LABEL_COLOR = (220, 220, 220)


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
        self._avatar_image: np.ndarray | None = None
        self._avatar_dir = Path("data/avatars")
        self._puppet_opacity = 1.0
        self._hub_url: str | None = None
        self._qr_visible = config.web.qr_visible_on_startup
        self._qr_png_bytes: bytes | None = None
        self._qr_image: np.ndarray | None = None
        self._snowfall_scale = 1.0
        self._snowfall_custom_scale = 1.0
        self._snowfall_density = 1.0
        self._snowfall_custom_density = 1.0
        self._brightness = 1.0
        self._tv_source = "camera"  # "camera" | "youtube" | "media"
        self._show_hud = True
        self._bass_overlay = False
        self._bass_overlay_renderer = BassPulseRenderer()

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
            SnowfallCustomRenderer(),
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

    def reload_effects(self) -> list[str]:
        """Hot-reload all effect renderer modules and reinstantiate them.

        Reimports every renderer module from disk, creates fresh instances,
        and swaps them in. Camera, models, and state are preserved.

        Returns:
            List of reloaded effect names.
        """
        # Modules to reload (in import order — utils first, then effects)
        module_names = [
            "src.render.effects",
            "src.render.utils",
            "src.render.base",
            "src.render.effects.neon_wireframe",
            "src.render.effects.energy_aura",
            "src.render.effects.motion_trails",
            "src.render.effects.glitch_body",
            "src.render.effects.digital_rain",
            "src.render.effects.snowfall_custom",
            "src.render.effects.particle_dissolve",
            "src.render.effects.halo_wings",
            "src.render.effects.sprite_puppet",
            "src.render.effects.passthrough",
        ]

        with self._lock:
            current_name = self.active_renderer.name

            # Reload all modules
            for mod_name in module_names:
                if mod_name in sys.modules:
                    try:
                        importlib.reload(sys.modules[mod_name])
                    except Exception:
                        logger.exception("Failed to reload %s", mod_name)
                        raise

            # Re-import classes from freshly reloaded modules
            from src.render.effects.neon_wireframe import NeonWireframeRenderer as NWR
            from src.render.effects.energy_aura import EnergyAuraRenderer as EAR
            from src.render.effects.motion_trails import MotionTrailsRenderer as MTR
            from src.render.effects.glitch_body import GlitchBodyRenderer as GBR
            from src.render.effects.digital_rain import DigitalRainRenderer as DRR
            from src.render.effects.snowfall_custom import SnowfallCustomRenderer as SCR
            from src.render.effects.particle_dissolve import ParticleDissolveRenderer as PDR
            from src.render.effects.halo_wings import HaloWingsRenderer as HWR
            from src.render.effects.sprite_puppet import SpritePuppetRenderer as SPR
            from src.render.effects.bass_pulse import BassPulseRenderer as BPR
            from src.render.effects.passthrough import PassthroughRenderer as PTR

            self._renderers = [
                NWR(), EAR(), MTR(), GBR(), DRR(), SCR(), PDR(), HWR(), SPR(), PTR(),
            ]
            self._bass_overlay_renderer = BPR()
            self._renderer_map = {r.name: r for r in self._renderers}

            # Re-apply saved state to fresh renderer instances
            for r in self._renderers:
                if hasattr(r, 'snowfall_scale'):
                    r.snowfall_scale = self._snowfall_scale
                if hasattr(r, 'snowfall_density'):
                    r.snowfall_density = self._snowfall_density
                if hasattr(r, 'snowfall_custom_scale'):
                    r.snowfall_custom_scale = self._snowfall_custom_scale
                if hasattr(r, 'snowfall_custom_density'):
                    r.snowfall_custom_density = self._snowfall_custom_density

            # Try to stay on the same effect by name
            if current_name in self._renderer_map:
                self._active_idx = self._renderers.index(self._renderer_map[current_name])
            else:
                self._active_idx = 0

        names = self.get_renderer_names()
        logger.info("Hot-reloaded %d effects: %s", len(names), ", ".join(names))
        return names

    def set_avatar(self, image_data: bytes) -> Path:
        """Set a custom avatar image for the Sprite Puppet effect.

        Automatically removes the background (white, checkerboard, or
        solid color) and saves as BGRA PNG with transparency.

        Args:
            image_data: Raw image bytes (JPEG/PNG).

        Returns:
            Path to the saved avatar file.
        """
        self._avatar_dir.mkdir(parents=True, exist_ok=True)
        path = self._avatar_dir / "current.png"

        arr = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Could not decode image")

        img = self._prepare_avatar(img)

        cv2.imwrite(str(path), img)
        self._avatar_image = img
        logger.info("Custom avatar set: %s (%dx%d)", path, img.shape[1], img.shape[0])
        return path

    @staticmethod
    def _prepare_avatar(img: np.ndarray) -> np.ndarray:
        """Prepare an avatar image for the Sprite Puppet effect.

        If the image already has meaningful alpha, use it as-is.
        If a face is detected, crop to the face with an elliptical alpha mask.
        If no face is detected (cartoon/meme), use the whole image as-is.

        Args:
            img: Input image (BGR or BGRA).

        Returns:
            Processed BGRA image ready for head overlay.
        """
        # If image already has meaningful alpha, keep it
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            if np.mean(alpha < 250) > 0.2:
                return img
            bgr = img[:, :, :3]
        else:
            bgr = img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Try to detect a face
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            # No face found — cartoon/meme, use whole image as-is (fully opaque)
            alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
            return np.dstack([bgr, alpha])

        # Take the largest face
        areas = [w * h for (x, y, w, h) in faces]
        fx, fy, fw, fh = faces[areas.index(max(areas))]

        # Expand the crop region to include forehead/chin/sides (40% padding)
        pad = 0.4
        cx = fx + fw // 2
        cy = fy + fh // 2
        half_w = int(fw * (1 + pad) / 2)
        half_h = int(fh * (1 + pad) / 2)
        # Make it square-ish for better head mapping
        half = max(half_w, half_h)

        img_h, img_w = bgr.shape[:2]
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(img_w, cx + half)
        y2 = min(img_h, cy + half)

        cropped = bgr[y1:y2, x1:x2]
        ch, cw = cropped.shape[:2]

        # Create an elliptical alpha mask (soft edges)
        alpha = np.zeros((ch, cw), dtype=np.uint8)
        center = (cw // 2, ch // 2)
        axes = (cw // 2, ch // 2)
        cv2.ellipse(alpha, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
        # Feather the edges
        alpha = cv2.GaussianBlur(alpha, (15, 15), 0)

        return np.dstack([cropped, alpha])

    def clear_avatar(self) -> None:
        """Remove the custom avatar."""
        self._avatar_image = None
        path = self._avatar_dir / "current.png"
        if path.exists():
            path.unlink()
        logger.info("Custom avatar cleared")

    @property
    def avatar_image(self) -> np.ndarray | None:
        """The current custom avatar image, or None."""
        return self._avatar_image

    @property
    def puppet_opacity(self) -> float:
        """Sprite puppet skeleton opacity 0.0-1.0."""
        return self._puppet_opacity

    def set_puppet_opacity(self, value: float) -> None:
        """Set the puppet skeleton opacity.

        Args:
            value: Opacity 0.0 (invisible) to 1.0 (fully opaque).
        """
        self._puppet_opacity = max(0.0, min(1.0, float(value)))

    @property
    def snowfall_scale(self) -> float:
        """Snowfall sprite size scale 0.3-3.0."""
        return self._snowfall_scale

    def set_snowfall_scale(self, value: float) -> None:
        """Set the snowfall sprite size scale."""
        self._snowfall_scale = max(0.3, min(3.0, float(value)))
        for r in self._renderers:
            if hasattr(r, 'snowfall_scale'):
                r.snowfall_scale = self._snowfall_scale

    @property
    def snowfall_custom_scale(self) -> float:
        """Snowfall Custom sprite size scale 0.3-3.0."""
        return self._snowfall_custom_scale

    def set_snowfall_custom_scale(self, value: float) -> None:
        """Set the Snowfall Custom sprite size scale."""
        self._snowfall_custom_scale = max(0.3, min(3.0, float(value)))
        for r in self._renderers:
            if hasattr(r, 'snowfall_custom_scale'):
                r.snowfall_custom_scale = self._snowfall_custom_scale

    @property
    def snowfall_density(self) -> float:
        """Snowfall sprite density 0.1-3.0."""
        return self._snowfall_density

    def set_snowfall_density(self, value: float) -> None:
        """Set the Snowfall sprite density."""
        self._snowfall_density = max(0.1, min(3.0, float(value)))
        for r in self._renderers:
            if hasattr(r, 'snowfall_density'):
                r.snowfall_density = self._snowfall_density

    @property
    def snowfall_custom_density(self) -> float:
        """Snowfall Custom sprite density 0.1-3.0."""
        return self._snowfall_custom_density

    def set_snowfall_custom_density(self, value: float) -> None:
        """Set the Snowfall Custom sprite density."""
        self._snowfall_custom_density = max(0.1, min(3.0, float(value)))
        for r in self._renderers:
            if hasattr(r, 'snowfall_custom_density'):
                r.snowfall_custom_density = self._snowfall_custom_density

    @property
    def brightness(self) -> float:
        """Global brightness multiplier 0.2-2.0."""
        return self._brightness

    def set_brightness(self, value: float) -> None:
        """Set global brightness multiplier."""
        self._brightness = max(0.2, min(2.0, float(value)))

    @property
    def tv_source(self) -> str:
        """Current TV source: 'camera', 'youtube', or 'media'."""
        return self._tv_source

    def set_tv_source(self, source: str) -> None:
        """Switch TV source."""
        if source in ("camera", "youtube", "media"):
            self._tv_source = source
            logger.info("TV source switched to: %s", source)

    @property
    def show_hud(self) -> bool:
        return self._show_hud

    def toggle_hud(self) -> bool:
        self._show_hud = not self._show_hud
        return self._show_hud

    @property
    def bass_overlay(self) -> bool:
        return self._bass_overlay

    def toggle_bass_overlay(self) -> bool:
        self._bass_overlay = not self._bass_overlay
        return self._bass_overlay

    @property
    def noise_gate(self) -> float:
        return self._config.audio.noise_gate

    def set_noise_gate(self, value: float) -> None:
        self._config.audio.noise_gate = max(0.0, value)

    @property
    def hub_url(self) -> str | None:
        """Current phone-access URL for the local web app."""
        return self._hub_url

    @property
    def qr_visible(self) -> bool:
        """Whether the QR overlay should be shown on the preview output."""
        return self._qr_visible

    def set_hub_url(self, url: str) -> None:
        """Set the local hub URL and regenerate the cached QR image.

        Args:
            url: Local network URL guests should scan.

        Raises:
            RuntimeError: If QR generation dependencies are unavailable.
        """
        qr_png = build_qr_png(url)
        decoded = cv2.imdecode(
            np.frombuffer(qr_png, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if decoded is None:
            raise RuntimeError("Failed to decode generated QR PNG")

        with self._lock:
            self._hub_url = url
            self._qr_png_bytes = qr_png
            self._qr_image = decoded

    def toggle_qr_visibility(self) -> bool:
        """Toggle the preview QR overlay and return the new state."""
        with self._lock:
            self._qr_visible = not self._qr_visible
            return self._qr_visible

    def get_qr_png(self) -> bytes | None:
        """Return the cached QR image bytes for the phone UI."""
        with self._lock:
            return self._qr_png_bytes

    def overlay_qr(self, frame: np.ndarray) -> np.ndarray:
        """Overlay the cached QR code in the lower-right corner.

        Args:
            frame: Preview frame that will be shown and streamed.

        Returns:
            The same frame with the QR panel applied when enabled.
        """
        with self._lock:
            if not self._qr_visible or self._qr_image is None:
                return frame
            qr_image = self._qr_image.copy()

        frame_h, frame_w = frame.shape[:2]
        side = int(min(QR_MAX_SIZE, max(QR_MIN_SIZE, min(frame_h, frame_w) * 0.12)))
        qr_resized = cv2.resize(
            qr_image,
            (side, side),
            interpolation=cv2.INTER_NEAREST,
        )

        panel_w = side + (QR_PADDING * 2)
        panel_h = side + QR_LABEL_HEIGHT + (QR_PADDING * 2)
        panel_x = max(QR_MARGIN, frame_w - panel_w - QR_MARGIN)
        panel_y = max(QR_MARGIN, frame_h - panel_h - QR_MARGIN)
        panel_x2 = min(frame_w, panel_x + panel_w)
        panel_y2 = min(frame_h, panel_y + panel_h)

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x2, panel_y2),
            QR_PANEL_COLOR,
            thickness=-1,
        )
        frame[:] = cv2.addWeighted(overlay, QR_PANEL_ALPHA, frame, 1.0 - QR_PANEL_ALPHA, 0.0)

        text_size = cv2.getTextSize(QR_LABEL_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        label_x = panel_x + (panel_w - text_size[0]) // 2
        label_y = panel_y + 12
        cv2.putText(
            frame,
            QR_LABEL_TEXT,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            QR_LABEL_COLOR,
            1,
            cv2.LINE_AA,
        )

        qr_x = panel_x + QR_PADDING
        qr_y = panel_y + QR_LABEL_HEIGHT + QR_PADDING
        frame[qr_y:qr_y + side, qr_x:qr_x + side] = qr_resized
        return frame

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
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
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
            avatar=self._avatar_image,
            puppet_opacity=self._puppet_opacity,
        )

        output = renderer.render(ctx)

        # Apply bass color overlay if toggled on
        if self._bass_overlay:
            bass_ctx = RenderContext(
                frame=output,
                pose=None,
                mask=None,
                bass_energy=self._bass_energy,
                timestamp=ctx.timestamp,
            )
            output = self._bass_overlay_renderer.render(bass_ctx)

        # Apply global brightness
        if self._brightness != 1.0:
            output = cv2.convertScaleAbs(output, alpha=self._brightness, beta=0)

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
            "puppet_opacity": self._puppet_opacity,
            "qr_visible": self._qr_visible,
            "hub_url": self._hub_url,
            "snowfall_scale": self._snowfall_scale,
            "snowfall_custom_scale": self._snowfall_custom_scale,
            "snowfall_density": self._snowfall_density,
            "snowfall_custom_density": self._snowfall_custom_density,
            "brightness": self._brightness,
            "tv_source": self._tv_source,
            "show_hud": self._show_hud,
            "bass_overlay": self._bass_overlay,
            "noise_gate": self._config.audio.noise_gate,
        }

    def close(self) -> Path | None:
        """Release capture resources before shutdown.

        Returns:
            Finalized recording path if a recording was active, else None.
        """
        return self.stop_recording()
