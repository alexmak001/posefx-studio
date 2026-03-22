# posefx-studio — Project Intelligence

## What this project is

A party entertainment hub. Four features controlled from guests' phones:

1. **Live FX** — Webcam + TV + body tracking + visual effects. Guests step in front of a camera, see themselves transformed with effects that pulse to bass music.
2. **YouTube DJ** — Search and play YouTube videos on the TV from your phone.
3. **AI Video Lab** — Upload a selfie to face-swap onto images/videos, or describe an edit in text and AI does it. Powered by Replicate API.
4. **Gallery** — Browse and download all photos, recordings, and AI-generated results.

Everything is controlled from a phone web app at `http://[machine-ip]:8000`. No app install needed.

Runs on Apple Silicon MacBook for dev, deploys to Jetson Orin Nano 8GB for the party.

## Architecture

The phone web app is the central hub. It has 4 tabs, each backed by a different subsystem:

```
Phone browser → FastAPI server (port 8000)
  ├── Tab: Live FX    → PartyEngine (local inference + render + MJPEG stream)
  ├── Tab: YouTube DJ → yt-dlp (search + stream to TV)
  ├── Tab: AI Lab     → Replicate API (cloud processing, async jobs)
  └── Tab: Gallery    → Static file serving (photos, recordings, AI results)
```

TV output switches between: live camera effects, YouTube playback, or idle screen.

See `.architecture.md` for the full module map and implementation status.
See `BUILDPLAN.md` for the step-by-step build plan.

## Critical rules (ALL coding agents must follow these)

### Pipeline (Live FX)
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
- Key deps: opencv-python, ultralytics, numpy, pyyaml, fastapi, uvicorn, pyaudio, yt-dlp, replicate

### Testing
- Tests go in tests/
- Test logic modules — not rendering or I/O
- Use pytest
- Run: `uv run pytest tests/`

### Config
- All tunable values live in config/demo.yaml
- src/utils/config.py provides typed access via AppConfig dataclass
- NEVER hardcode device IDs, paths, thresholds, resolutions, or API keys
- API keys go in environment variables (REPLICATE_API_TOKEN), never in config files

### Performance
- Target: ≥15 FPS at 720p on Apple Silicon for live effects
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
- 4 tabs: Live FX, YouTube DJ, AI Lab, Gallery
- MJPEG stream for live preview
- WebSocket for real-time state sync
- All engine methods called from web must be thread-safe
- Dark theme (#111 background), large tap targets, mobile-first

### API integrations
- Replicate API for face swap and AI edits — async job pattern
- yt-dlp for YouTube search and streaming — no API key needed
- All API calls go through src/services/ modules, never directly from web handlers

### File organization
- src/io/ — webcam, video input, preview, recording, photo capture
- src/inference/ — base classes, YOLO wrappers, future TensorRT wrappers
- src/render/ — base renderer, effects/ subfolder with one file per effect
- src/audio/ — mic capture, FFT, bass extraction
- src/web/ — FastAPI server, HTML templates
- src/services/ — YouTube (yt-dlp), Replicate API, job queue
- src/utils/ — config, timing, platform detection, geometry
- src/engine.py — PartyEngine (mode switching, frame processing, thread-safe controls)
- config/ — YAML config files
- data/ — runtime output (photos, recordings, ai_results)
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

### Environment variables
```bash
export REPLICATE_API_TOKEN="r8_..."   # required for AI Lab features
```