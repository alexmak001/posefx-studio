# posefx-studio — Project Intelligence

## What this project is

A real-time party entertainment station. Webcam + TV + body tracking + visual effects.
Guests step in front of a camera, see themselves transformed (neon skeleton, glowing
silhouette, particle body), control effects from their phones, and take photos.

Runs on Apple Silicon MacBook for dev, deploys to Jetson Orin Nano 8GB for the party.

## Architecture

Pipeline: webcam/video → PartyEngine (inference at half-res → active effect renderer) → composited frame → preview window

Audio capture runs in a background thread, feeds bass_energy into the engine for bass-reactive effects.

See `.architecture.md` for the full module map and implementation status.
See `BUILDPLAN.md` for the step-by-step build plan.

## What is implemented (DO NOT rewrite — extend)

| Module | What it does |
|--------|-------------|
| `src/utils/config.py` | YAML config → typed `AppConfig` dataclass (camera, inference, debug, effects, audio) |
| `src/utils/timing.py` | `FPSCounter` with rolling average |
| `src/utils/platform.py` | Auto-detect mac/jetson/cpu at startup |
| `src/io/webcam.py` | `WebcamCapture` — OpenCV camera wrapper |
| `src/io/video_input.py` | `VideoFileInput` — MP4 input with loop, same interface as webcam |
| `src/io/preview.py` | `PreviewWindow` — OpenCV display, quit detection, keyboard input |
| `src/inference/base.py` | `BasePoseEstimator`, `BaseSegmenter`, `PoseResult`, `MaskResult` |
| `src/inference/pose_estimator.py` | `YOLOPoseEstimator` — YOLOv8n-pose, COCO 17-keypoint |
| `src/inference/segmenter.py` | `YOLOSegmenter` — YOLOv8n-seg, person-only filter |
| `src/audio/capture.py` | `AudioCapture` — mic or WAV file, FFT bass extraction, background thread |
| `src/render/base.py` | `BaseRenderer` ABC, `RenderContext` dataclass |
| `src/render/utils.py` | `get_head_mask()`, `composite_head()` — shared head exclusion for effects |
| `src/render/effects/` | 10 effect renderers (see below) |
| `src/engine.py` | `PartyEngine` — manages renderers, runs selective inference at configurable scale, thread-safe |
| `src/main.py` | Entry point — CLI args, main loop, HUD overlays, keyboard controls |
| `config/demo.yaml` | All config: camera, inference, effects, audio, debug |

### The 10 effects (in `src/render/effects/`)

1. **neon_wireframe.py** — Glowing cyan skeleton on black (pose+mask)
2. **energy_aura.py** — Warm concentric glow rings + particles (pose+mask)
3. **motion_trails.py** — Rainbow afterimage streaks (pose+mask)
4. **glitch_body.py** — Horizontal band distortion + RGB split (pose+mask)
5. **digital_rain.py** — Matrix-style falling characters in body (pose+mask)
6. **shadow_clones.py** — Time-delayed colored silhouette copies (pose+mask)
7. **particle_dissolve.py** — Scattering colored dots filling body (pose+mask)
8. **halo_wings.py** — Golden halo + energy wings overlay (pose only)
9. **sprite_puppet.py** — Geometric 2D puppet avatar (pose only)
10. **passthrough.py** — Clean camera + bass vignette (no inference)

Effects 1-8 preserve real face via `composite_head()`. All respond to `bass_energy`.

## What is NOT yet implemented (next steps in BUILDPLAN.md)

- **Step 4**: Photo capture with countdown + video recording + auto-capture mode
- **Step 5**: Web app (FastAPI + MJPEG stream + phone UI for effect control/gallery)
- **Step 6**: Fullscreen mode, crossfade transitions, startup reliability polish

## Critical rules (ALL coding agents must follow these)

### Pipeline
- Pipeline stages: input → inference → render → output
- Stages communicate through dataclasses (PoseResult, MaskResult, RenderContext)
- NEVER access raw model outputs outside the inference layer
- Model-specific code lives ONLY behind abstract base classes in src/inference/
- Config is loaded once at startup and passed explicitly — no global state
- PartyEngine only runs inference models the active renderer declares it needs

### Performance
- Target: ≥10 FPS at 720p on Apple Silicon (15 is aspirational)
- Inference runs at half resolution (inference_scale: 0.5), results scaled back up
- CPU device is preferred over MPS for YOLOv8 nano models on Apple Silicon
- Profile before optimizing — inference is the bottleneck
- Keep render-only time under 70ms per frame

### Effects
- Every body effect (1-8) must use `composite_head()` to preserve the real face
- New effects go in `src/render/effects/` as separate files, one class per file
- Each effect must declare `needs_pose` and `needs_mask` properties
- Register new effects in `src/engine.py` `_renderers` list
- Effects must handle the case where pose/mask has 0 people gracefully

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
- Key deps: opencv-python, ultralytics, numpy, pyyaml, fastapi, pyaudio, scipy

### Config
- All tunable values live in config/demo.yaml
- src/utils/config.py provides typed access via AppConfig dataclass
- NEVER hardcode device IDs, paths, thresholds, resolutions, or model names
- When adding new config sections, add a new dataclass in config.py and wire it into AppConfig

### Hardware portability
- inference/base.py defines the abstract interfaces
- Mac uses PyTorch + CPU (faster than MPS for nano models), Jetson uses TensorRT
- src/utils/platform.py auto-detects: mac / jetson / cpu
- Jetson port = writing new concrete inference classes, not changing the pipeline

### Web app (when implementing Step 5)
- FastAPI server runs in background thread
- Single-page vanilla HTML/JS/CSS frontend — no framework, no build step
- MJPEG stream for live preview
- WebSocket for real-time state sync
- All engine methods called from web must be thread-safe

### File organization
- src/io/ — webcam, video input, preview, recording, photo capture
- src/inference/ — base classes, YOLO wrappers, future TensorRT wrappers
- src/render/ — base renderer, utils, effects/ subfolder with one file per effect
- src/audio/ — mic capture, FFT, bass extraction
- src/web/ — FastAPI server, HTML template (not yet created)
- src/utils/ — config, timing, platform detection, geometry
- src/engine.py — PartyEngine (mode switching, frame processing, thread-safe controls)
- config/ — YAML config files
- data/ — runtime output (photos, recordings)

### Key interfaces
```python
# src/inference/base.py
PoseResult:  keypoints (N,17,2), confidences (N,17), boxes (N,4), num_people
MaskResult:  masks (N,H,W), combined_mask (H,W), num_people

# src/render/base.py
RenderContext: frame, pose, mask, bass_energy, timestamp
BaseRenderer:  render(ctx) → frame, name, needs_pose, needs_mask

# src/render/utils.py
get_head_mask(pose, frame_shape, scale=1.8) → binary mask (H,W)
composite_head(output, original, pose, frame_shape) → frame with real face

# src/engine.py
PartyEngine: process_frame(frame) → composited frame
             set_renderer(name), next_renderer(), prev_renderer()
             set_bass_energy(energy)
             get_renderer_names() → list[str]
```

### Build and run
```bash
uv sync                                                            # install deps
uv run python -m src.main --config config/demo.yaml                # webcam mode
uv run python -m src.main --config config/demo.yaml --input v.mp4  # test video
uv run python -m src.main --config config/demo.yaml --audio m.wav  # with audio file
uv run pytest tests/                                               # run tests
```
