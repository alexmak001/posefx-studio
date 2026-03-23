"""Main entry point for posefx-studio.

Usage:
    python -m src.main --config config/demo.yaml
    python -m src.main --config config/demo.yaml --input video.mp4
"""

import argparse
import logging
import socket

import cv2

from src.audio.capture import AudioCapture
from src.engine import PartyEngine
from src.io.preview import PreviewWindow
from src.io.video_input import VideoFileInput
from src.io.webcam import WebcamCapture
from src.utils.config import load_config
from src.utils.platform import detect_platform
from src.utils.timing import FPSCounter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

KEY_NEXT = ord("n")
KEY_PREV = ord("p")
KEY_BASS_METER = ord("b")
KEY_SNAP = ord("s")
KEY_RECORD = ord("r")
KEY_AUTO_CAPTURE = ord("a")
KEY_RELOAD = ord("l")
COUNTDOWN_FONT_SCALE = 4.0
COUNTDOWN_THICKNESS = 8
STATUS_TEXT_COLOR = (255, 255, 255)
RECORDING_COLOR = (0, 0, 255)
AUTO_COLOR = (0, 215, 255)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="posefx-studio: real-time body tracking pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--input", default=None, help="Path to video file (uses webcam if not provided)")
    parser.add_argument("--audio", default=None, help="Path to WAV file for audio input (uses mic if not provided)")
    return parser.parse_args()


def main() -> None:
    """Run the main pipeline loop."""
    args = parse_args()
    config = load_config(args.config)

    platform = detect_platform()

    # Select input source
    if args.input:
        source = VideoFileInput(args.input)
        logger.info("Using video file input: %s", args.input)
    else:
        source = WebcamCapture(config.camera)
        logger.info("Using webcam input")

    preview = PreviewWindow()
    fps_counter = FPSCounter()

    # Start audio capture if enabled (before engine so we can pass it in)
    audio: AudioCapture | None = None
    if config.audio.enabled:
        audio = AudioCapture(config.audio, file_path=args.audio)
        audio.start()

    engine = PartyEngine(config, platform, audio_capture=audio)

    # Start web server if enabled
    if config.web.enabled:
        from src.web.server import start_server
        start_server(engine, config)
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            local_ip = "localhost"
        logger.info(
            "Party hub ready at http://%s:%d", local_ip, config.web.port
        )

    show_bass_meter = False

    logger.info(
        "Starting pipeline — 'n'/'p' cycle effects, 's' photo, 'r' record, 'a' auto, 'b' bass, 'l' reload effects, 'q'/ESC quit"
    )
    logger.info("Available effects: %s", ", ".join(engine.get_renderer_names()))

    try:
        while True:
            ok, frame = source.read()
            if not ok:
                logger.error("Failed to read frame from source")
                break

            fps_counter.tick()

            # Feed bass energy into the engine
            if audio is not None:
                engine.set_bass_energy(audio.bass_energy)

            # Process frame through the engine (inference + effect rendering)
            output = engine.process_frame(frame)
            capture_state = engine.get_capture_state()
            preview_frame = output.copy()

            # HUD overlays
            if config.debug.show_fps:
                cv2.putText(
                    preview_frame,
                    f"FPS: {fps_counter.fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

            # Show current effect name (top-right)
            effect_name = engine.active_renderer.name
            text_size = cv2.getTextSize(
                effect_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )[0]
            text_x = preview_frame.shape[1] - text_size[0] - 10
            cv2.putText(
                preview_frame,
                effect_name,
                (text_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Bass meter overlay (bottom-left bar + numeric value)
            if show_bass_meter and audio is not None:
                bass = audio.bass_energy
                bar_w = 200
                bar_h = 20
                bar_x, bar_y = 10, preview_frame.shape[0] - 40
                # Background
                cv2.rectangle(
                    preview_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                    (40, 40, 40), -1,
                )
                # Fill — green to red gradient
                fill_w = int(bar_w * bass)
                if fill_w > 0:
                    g = int(255 * (1 - bass))
                    r = int(255 * bass)
                    cv2.rectangle(
                        preview_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                        (0, g, r), -1,
                    )
                # Border
                cv2.rectangle(
                    preview_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                    (200, 200, 200), 1,
                )
                # Numeric value
                cv2.putText(
                    preview_frame,
                    f"BASS: {bass:.2f}",
                    (bar_x + bar_w + 10, bar_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )

            if capture_state.flash_active:
                flash_overlay = preview_frame.copy()
                flash_overlay[:] = 255
                preview_frame = cv2.addWeighted(flash_overlay, 0.45, preview_frame, 0.55, 0.0)
                snap_text = "SNAP!"
                snap_size = cv2.getTextSize(
                    snap_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4
                )[0]
                snap_x = (preview_frame.shape[1] - snap_size[0]) // 2
                snap_y = (preview_frame.shape[0] + snap_size[1]) // 2
                cv2.putText(
                    preview_frame,
                    snap_text,
                    (snap_x, snap_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (255, 255, 255),
                    4,
                )

            if capture_state.countdown_value is not None:
                countdown_text = str(capture_state.countdown_value)
                countdown_size = cv2.getTextSize(
                    countdown_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    COUNTDOWN_FONT_SCALE,
                    COUNTDOWN_THICKNESS,
                )[0]
                countdown_x = (preview_frame.shape[1] - countdown_size[0]) // 2
                countdown_y = (preview_frame.shape[0] + countdown_size[1]) // 2
                cv2.putText(
                    preview_frame,
                    countdown_text,
                    (countdown_x, countdown_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    COUNTDOWN_FONT_SCALE,
                    STATUS_TEXT_COLOR,
                    COUNTDOWN_THICKNESS,
                )

            if capture_state.is_recording:
                cv2.circle(preview_frame, (28, 68), 10, RECORDING_COLOR, -1)
                cv2.putText(
                    preview_frame,
                    "REC",
                    (46, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    STATUS_TEXT_COLOR,
                    2,
                )

            if capture_state.auto_capture_enabled:
                auto_text = "AUTO"
                auto_size = cv2.getTextSize(auto_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                auto_x = preview_frame.shape[1] - auto_size[0] - 10
                auto_y = 64
                cv2.putText(
                    preview_frame,
                    auto_text,
                    (auto_x, auto_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    AUTO_COLOR,
                    2,
                )

            # Feed latest frame to web MJPEG stream
            engine.set_latest_frame(preview_frame, fps_counter.fps)

            preview.show(preview_frame)

            # Handle keyboard input
            key = preview.last_key
            if key == KEY_NEXT:
                engine.next_renderer()
            elif key == KEY_PREV:
                engine.prev_renderer()
            elif key == KEY_SNAP:
                engine.trigger_photo()
            elif key == KEY_RECORD:
                capture_state = engine.get_capture_state()
                if capture_state.is_recording:
                    saved_path = engine.stop_recording()
                    if saved_path is not None:
                        logger.info("Recording saved to %s", saved_path)
                else:
                    resolution = (frame.shape[1], frame.shape[0])
                    try:
                        record_fps = fps_counter.fps if fps_counter.fps >= 5 else config.camera.fps
                        path = engine.start_recording(
                            record_fps,
                            resolution,
                        )
                    except Exception:
                        logger.exception("Failed to start recording")
                    else:
                        logger.info("Recording started: %s", path)
            elif key == KEY_AUTO_CAPTURE:
                engine.toggle_auto_capture()
            elif key == KEY_RELOAD:
                try:
                    names = engine.reload_effects()
                    logger.info("Effects reloaded: %s", ", ".join(names))
                except Exception:
                    logger.exception("Failed to reload effects")
            elif key == KEY_BASS_METER:
                show_bass_meter = not show_bass_meter
                logger.info("Bass meter: %s", "ON" if show_bass_meter else "OFF")

            if preview.should_quit():
                break
    finally:
        if audio is not None:
            audio.stop()
        final_recording = engine.close()
        if final_recording is not None:
            logger.info("Recording saved to %s", final_recording)
        source.release()
        preview.destroy()
        logger.info("Pipeline stopped")


if __name__ == "__main__":
    main()
