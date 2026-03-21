# posefx-studio — Project Intelligence

## What this project is

A real-time party entertainment station. Webcam + TV + body tracking + visual effects.
Guests step in front of a camera, see themselves transformed (neon skeleton, glowing
silhouette, particle body), control effects from their phones, and take photos.

Runs on Apple Silicon MacBook for dev, deploys to Jetson Orin Nano 8GB for the party.

## Architecture

Pipeline: webcam → pose/seg inference → effect renderer → composited frame → TV + web stream

See `.architecture.md` for the full module map and implementation status.
See `BUILDPLAN.md` for the step-by-step build plan.

## Critical rules (ALL coding agents must follow these)

### Pipeline
- Pipeline stages: input → inference → tracking → logic → render → output
- Each stage is a separate module under src/
- Stages communicate through dataclasses (PoseResult, MaskResult, RenderContext)
- NEVER access raw model outputs outside the inference layer
- Model-specific code lives ONLY behind abstract base classes in src/inference/
- Config is loaded once at startup and passed explicitly — no global state

### Code style
- Python 3.11+
- Type hints on all public function signatures
- Docstrings on all classes and public methods (Google style)
- No wildcard imports
- No print() for debugging — use Python logging module
- Constants in UPPER_SNAKE_CASE at module top

### Dependencies
- Use `uv` for package management (already configured)
- Add deps to pyproject.toml, then `uv sync`
- Key deps: opencv-python, ultralytics, numpy, pyyaml, fastapi, pyaudio

### Testing
- Tests go in tests/
- Test logic modules (gestures, zones, config) — not rendering or I/O
- Use pytest
- Run: `uv run pytest tests/`

### Config
- All tunable values live in config/demo.yaml
- src/utils/config.py provides typed access via AppConfig dataclass
- NEVER hardcode device IDs, paths, thresholds, resolutions, or model names

### Performance
- Target: ≥15 FPS at 720p on Apple Silicon
- Profile before optimizing
- Inference is the bottleneck — don't optimize rendering prematurely
- Only run inference models the current renderer needs (check needs_pose / needs_mask)

### Hardware portability
- inference/base.py defines the abstract interfaces
- Mac uses PyTorch + MPS, Jetson uses TensorRT engines
- src/utils/platform.py auto-detects: mac / jetson / cpu
- Jetson port = writing new concrete inference classes, not changing the pipeline
- OpenCV is the I/O layer on both platforms

### Web app
- FastAPI server runs in background thread
- Single-page vanilla HTML/JS/CSS frontend — no framework, no build step
- MJPEG stream for live preview
- WebSocket for real-time state sync
- All engine methods called from web must be thread-safe

### File organization
- src/io/ — webcam, video input, preview, recording, photo capture
- src/inference/ — base classes, YOLO wrappers, future TensorRT wrappers
- src/render/ — base renderer, effects/ subfolder with one file per effect
- src/audio/ — mic capture, FFT, bass extraction
- src/web/ — FastAPI server, HTML template
- src/utils/ — config, timing, platform detection, geometry
- src/engine.py — PartyEngine (mode switching, frame processing, thread-safe controls)
- config/ — YAML config files
- data/ — runtime output (photos, recordings)
- tools/ — benchmark and utility scripts

### Key interfaces
```python
# src/inference/base.py
PoseResult:  keypoints (N,17,2), confidences (N,17), boxes (N,4), num_people
MaskResult:  masks (N,H,W), combined_mask (H,W), num_people

# src/render/base.py
RenderContext: frame, pose, mask, bass_energy, timestamp
BaseRenderer:  render(ctx) → frame, name, needs_pose, needs_mask
```

### Build and run
```bash
uv sync                                                            # install deps
uv run python -m src.main --config config/demo.yaml                # webcam mode
uv run python -m src.main --config config/demo.yaml --input v.mp4  # test video
uv run python -m src.main --config config/demo.yaml --fullscreen   # party mode
uv run pytest tests/                                               # run tests
```