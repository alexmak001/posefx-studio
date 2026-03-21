# posefx-studio

Real-time body-tracking visual effects for parties. Guests step in front of a camera, see themselves transformed (neon skeleton, glowing silhouette, particle body) on a TV, and control effects from their phones.

Runs on Apple Silicon MacBook for development, targets Jetson Orin Nano 8GB for deployment.

## Quick start

```bash
uv sync
uv run python -m src.main --config config/demo.yaml
```

Press `q` or `ESC` to quit.

## Usage

```bash
# Webcam mode (default)
uv run python -m src.main --config config/demo.yaml

# Video file input (for testing without a webcam)
uv run python -m src.main --config config/demo.yaml --input path/to/video.mp4

# With audio file for bass reactivity
uv run python -m src.main --config config/demo.yaml --input video.mp4 --audio music.wav
```

## Keyboard controls

| Key     | Action                        |
|---------|-------------------------------|
| `n`     | Next effect                   |
| `p`     | Previous effect               |
| `b`     | Toggle bass energy meter      |
| `q`/ESC | Quit                          |

## Effects

| Effect          | Description                                              |
|-----------------|----------------------------------------------------------|
| Neon Skeleton   | Glowing green skeleton on black background               |
| Robot Skeleton  | Metallic blue/silver angular skeleton with joint circles |
| Fire Skeleton   | Orange/red skeleton with upward-drifting fire particles  |
| Body Glow       | Bright outline glow around body silhouette               |
| Particle Fill   | Swirling particle dots filling the body mask             |
| Passthrough     | Raw camera feed (no effect)                              |

All effects respond to bass energy from audio input — lines get thicker, particles speed up, and glow intensifies with the music.

## Configuration

All settings are in `config/demo.yaml`:

- **camera** — device ID, resolution, FPS
- **inference** — model paths, device (auto/cpu/mps/cuda), confidence threshold
- **audio** — enable/disable, device, FFT bass band, smoothing, sensitivity, noise gate
- **effects** — default effect
- **debug** — toggle FPS, skeleton, mask overlays

## Architecture

```
webcam/video → [frame] → pose estimator  → PoseResult ─┐
                        → segmenter       → MaskResult ─┤→ PartyEngine → active effect renderer → preview window
                        audio capture     → bass_energy ─┘
```

The pipeline only runs inference models that the active renderer needs — e.g., passthrough skips both pose and segmentation for maximum FPS.

## Development

```bash
uv sync                    # install dependencies
uv run pytest tests/       # run tests
```

### Requirements

- Python 3.11+
- `uv` for dependency management
- Webcam (or video file for testing)
- Optional: microphone or WAV file for bass-reactive effects
