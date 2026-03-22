"""FastAPI web server for phone-based party control."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

if TYPE_CHECKING:
    from src.engine import PartyEngine
    from src.utils.config import AppConfig

logger = logging.getLogger(__name__)

MJPEG_BOUNDARY = b"--frame"
MJPEG_CONTENT_TYPE = "multipart/x-mixed-replace; boundary=frame"
WS_BROADCAST_INTERVAL = 0.25  # seconds between WebSocket state pushes

# Directories to scan for gallery files
GALLERY_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi"}

app = FastAPI(title="posefx-studio", docs_url=None, redoc_url=None)

# These get set by start_server() before the app handles any requests
_engine: PartyEngine | None = None
_config: AppConfig | None = None
_ws_clients: set[WebSocket] = set()


def _get_engine() -> PartyEngine:
    assert _engine is not None, "Engine not initialized"
    return _engine


def _get_config() -> AppConfig:
    assert _config is not None, "Config not initialized"
    return _config


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page phone UI."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=template_path.read_text())


# ---------------------------------------------------------------------------
# MJPEG stream
# ---------------------------------------------------------------------------

async def _mjpeg_generator():
    """Yield MJPEG frames at the configured max FPS."""
    config = _get_config()
    engine = _get_engine()
    quality = config.web.stream_quality
    interval = 1.0 / max(1, config.web.stream_max_fps)

    while True:
        jpeg = engine.get_latest_jpeg(quality)
        if jpeg is not None:
            yield (
                MJPEG_BOUNDARY
                + b"\r\nContent-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                + jpeg
                + b"\r\n"
            )
        await asyncio.sleep(interval)


@app.get("/stream")
async def stream() -> StreamingResponse:
    """MJPEG stream of the composited output."""
    return StreamingResponse(_mjpeg_generator(), media_type=MJPEG_CONTENT_TYPE)


# ---------------------------------------------------------------------------
# Live FX endpoints
# ---------------------------------------------------------------------------

@app.get("/api/effects")
async def get_effects() -> JSONResponse:
    """List all available effect names."""
    return JSONResponse({"effects": _get_engine().get_renderer_names()})


@app.post("/api/effects/{name}")
async def set_effect(name: str) -> JSONResponse:
    """Switch the active effect."""
    engine = _get_engine()
    if name not in {r for r in engine.get_renderer_names()}:
        return JSONResponse({"error": f"Unknown effect: {name}"}, status_code=404)
    engine.set_renderer(name)
    return JSONResponse({"effect": name})


@app.get("/api/effects/current")
async def get_current_effect() -> JSONResponse:
    """Get the current effect name."""
    return JSONResponse({"effect": _get_engine().active_renderer.name})


@app.post("/api/photo")
async def trigger_photo() -> JSONResponse:
    """Trigger a photo capture with countdown."""
    armed = _get_engine().trigger_photo()
    return JSONResponse({"armed": armed})


@app.post("/api/record/start")
async def start_recording() -> JSONResponse:
    """Start video recording."""
    engine = _get_engine()
    config = _get_config()
    capture_state = engine.get_capture_state()
    if capture_state.is_recording:
        return JSONResponse({"error": "Already recording"}, status_code=409)
    try:
        # Use actual measured FPS, not config value, to avoid speedup on playback
        actual_fps = engine.get_state()["fps"]
        record_fps = actual_fps if actual_fps >= 5 else config.camera.fps
        path = engine.start_recording(
            fps=record_fps,
            resolution=(config.camera.width, config.camera.height),
        )
        return JSONResponse({"recording": str(path)})
    except Exception as exc:
        logger.exception("Failed to start recording")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/record/stop")
async def stop_recording() -> JSONResponse:
    """Stop video recording."""
    path = _get_engine().stop_recording()
    return JSONResponse({"saved": str(path) if path else None})


@app.post("/api/auto-capture/toggle")
async def toggle_auto_capture() -> JSONResponse:
    """Toggle auto-capture mode."""
    enabled = _get_engine().toggle_auto_capture()
    return JSONResponse({"auto_capture": enabled})


# ---------------------------------------------------------------------------
# Gallery endpoints
# ---------------------------------------------------------------------------

def _scan_gallery() -> list[dict]:
    """Scan data directories for gallery files."""
    config = _get_config()
    items: list[dict] = []
    dirs_to_scan = [
        ("photo", Path(config.capture.photo_dir)),
        ("recording", Path(config.capture.recording_dir)),
    ]

    for file_type, base_dir in dirs_to_scan:
        if not base_dir.exists():
            continue
        for path in sorted(base_dir.rglob("*"), reverse=True):
            if not path.is_file() or path.suffix.lower() not in GALLERY_EXTENSIONS:
                continue
            if path.name.startswith("."):
                continue
            stat = path.stat()
            items.append({
                "filename": str(path),
                "name": path.name,
                "type": file_type,
                "subdir": path.parent.name,
                "size": stat.st_size,
                "created": stat.st_mtime,
            })

    items.sort(key=lambda x: x["created"], reverse=True)
    return items


@app.get("/api/gallery")
async def get_gallery() -> JSONResponse:
    """List all gallery files."""
    return JSONResponse({"items": _scan_gallery()})


@app.get("/api/gallery/file")
async def get_gallery_file(path: str) -> FileResponse:
    """Serve a gallery file by its absolute path.

    Args:
        path: Absolute path to the file (from gallery listing).
    """
    file_path = Path(path)
    config = _get_config()
    allowed_roots = [
        Path(config.capture.photo_dir).resolve(),
        Path(config.capture.recording_dir).resolve(),
    ]

    resolved = file_path.resolve()
    if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
        return JSONResponse({"error": "Access denied"}, status_code=403)

    if not resolved.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)

    return FileResponse(str(resolved))


@app.delete("/api/gallery/file")
async def delete_gallery_file(path: str) -> JSONResponse:
    """Delete a gallery file."""
    file_path = Path(path)
    config = _get_config()
    allowed_roots = [
        Path(config.capture.photo_dir).resolve(),
        Path(config.capture.recording_dir).resolve(),
    ]

    resolved = file_path.resolve()
    if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
        return JSONResponse({"error": "Access denied"}, status_code=403)

    if not resolved.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)

    resolved.unlink()
    logger.info("Deleted gallery file: %s", resolved)
    return JSONResponse({"deleted": str(resolved)})


# ---------------------------------------------------------------------------
# YouTube stubs (Step 6)
# ---------------------------------------------------------------------------

@app.get("/api/youtube/search")
async def youtube_search(q: str = "") -> JSONResponse:
    """Search YouTube (stub)."""
    return JSONResponse({"results": [], "message": "YouTube search not yet implemented"})


@app.post("/api/youtube/play")
async def youtube_play() -> JSONResponse:
    """Play a YouTube video (stub)."""
    return JSONResponse({"error": "Not implemented"}, status_code=501)


@app.post("/api/youtube/stop")
async def youtube_stop() -> JSONResponse:
    """Stop YouTube playback (stub)."""
    return JSONResponse({"error": "Not implemented"}, status_code=501)


@app.get("/api/youtube/status")
async def youtube_status() -> JSONResponse:
    """YouTube playback status (stub)."""
    return JSONResponse({"status": "idle"})


# ---------------------------------------------------------------------------
# AI Lab stubs (Step 7)
# ---------------------------------------------------------------------------

@app.post("/api/ai/face-swap")
async def ai_face_swap() -> JSONResponse:
    """Face swap (stub)."""
    return JSONResponse({"error": "Not implemented"}, status_code=501)


@app.post("/api/ai/face-swap-video")
async def ai_face_swap_video() -> JSONResponse:
    """Video face swap (stub)."""
    return JSONResponse({"error": "Not implemented"}, status_code=501)


@app.post("/api/ai/edit")
async def ai_edit() -> JSONResponse:
    """AI edit (stub)."""
    return JSONResponse({"error": "Not implemented"}, status_code=501)


@app.get("/api/ai/jobs")
async def ai_jobs() -> JSONResponse:
    """List AI jobs (stub)."""
    return JSONResponse({"jobs": []})


@app.get("/api/ai/jobs/{job_id}")
async def ai_job(job_id: str) -> JSONResponse:
    """Get AI job status (stub)."""
    return JSONResponse({"error": "Not found"}, status_code=404)


# ---------------------------------------------------------------------------
# WebSocket for real-time state
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Push engine state to connected clients."""
    await ws.accept()
    _ws_clients.add(ws)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))
    try:
        while True:
            engine = _get_engine()
            state = engine.get_state()
            await ws.send_text(json.dumps(state))
            await asyncio.sleep(WS_BROADCAST_INTERVAL)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _ws_clients.discard(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(_ws_clients))


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def start_server(engine: PartyEngine, config: AppConfig) -> threading.Thread:
    """Start the FastAPI server in a background daemon thread.

    Args:
        engine: The PartyEngine instance to control.
        config: Application config with web settings.

    Returns:
        The background thread running uvicorn.
    """
    global _engine, _config
    _engine = engine
    _config = config

    def _run() -> None:
        uvicorn.run(
            app,
            host=config.web.host,
            port=config.web.port,
            log_level="warning",
        )

    thread = threading.Thread(target=_run, daemon=True, name="web-server")
    thread.start()
    logger.info("Web server started on http://%s:%d", config.web.host, config.web.port)
    return thread
