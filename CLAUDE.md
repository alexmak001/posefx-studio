# posefx-studio — Project Rules

## Implementation Reference
- See `.architecture.md` for full module map, data flow contracts, and implementation status.

## Architecture
- Pipeline stages: input → inference → tracking → logic → render → output
- Each stage is a separate module under src/
- Stages communicate through dataclasses (PoseResult, MaskResult, etc.), never raw model outputs
- Model-specific code lives ONLY in concrete implementations behind abstract base classes
- Config is loaded once at startup and passed explicitly — no global state

## Code style
- Type hints on all public function signatures
- Docstrings on all classes and public methods (Google style)
- No wildcard imports
- No print() for debugging — use Python logging module
- Constants in UPPER_SNAKE_CASE at module top

## Testing
- Tests go in tests/
- Test logic modules (gestures, zones, config) — not rendering or I/O
- Use pytest

## Config
- All tunable values live in config/demo.yaml
- src/utils/config.py provides typed access
- Never hardcode device IDs, paths, thresholds, or resolutions

## Performance
- Target: ≥15 FPS at 720p on Apple Silicon
- Profile before optimizing
- Inference is the bottleneck — don't optimize rendering prematurely

## Future portability
- inference/base.py defines the interfaces
- Jetson port means writing new concrete classes, not changing the pipeline
- Keep OpenCV as the I/O layer — it works on both platforms
