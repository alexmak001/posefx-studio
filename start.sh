#!/bin/bash
# Start posefx-studio with fullscreen preview on the Jetson display

cd "$(dirname "$0")"

DISPLAY=:0 uv run python -m src.main --config config/jetson.yaml
