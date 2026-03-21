"""Main entry point for posefx-studio.

Usage:
    python -m src.main --config config/demo.yaml
    python -m src.main --config config/demo.yaml --input video.mp4
"""

import argparse
import logging

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
    engine = PartyEngine(config, platform)

    # Start audio capture if enabled
    audio: AudioCapture | None = None
    if config.audio.enabled:
        audio = AudioCapture(config.audio, file_path=args.audio)
        audio.start()

    show_bass_meter = False

    logger.info(
        "Starting pipeline — 'n'/'p' cycle effects, 'b' toggle bass meter, 'q'/ESC quit"
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

            # HUD overlays
            if config.debug.show_fps:
                cv2.putText(
                    output,
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
            text_x = output.shape[1] - text_size[0] - 10
            cv2.putText(
                output,
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
                bar_x, bar_y = 10, output.shape[0] - 40
                # Background
                cv2.rectangle(
                    output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                    (40, 40, 40), -1,
                )
                # Fill — green to red gradient
                fill_w = int(bar_w * bass)
                if fill_w > 0:
                    g = int(255 * (1 - bass))
                    r = int(255 * bass)
                    cv2.rectangle(
                        output, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                        (0, g, r), -1,
                    )
                # Border
                cv2.rectangle(
                    output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                    (200, 200, 200), 1,
                )
                # Numeric value
                cv2.putText(
                    output,
                    f"BASS: {bass:.2f}",
                    (bar_x + bar_w + 10, bar_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )

            preview.show(output)

            # Handle keyboard input
            key = preview.last_key
            if key == KEY_NEXT:
                engine.next_renderer()
            elif key == KEY_PREV:
                engine.prev_renderer()
            elif key == KEY_BASS_METER:
                show_bass_meter = not show_bass_meter
                logger.info("Bass meter: %s", "ON" if show_bass_meter else "OFF")

            if preview.should_quit():
                break
    finally:
        if audio is not None:
            audio.stop()
        source.release()
        preview.destroy()
        logger.info("Pipeline stopped")


if __name__ == "__main__":
    main()
