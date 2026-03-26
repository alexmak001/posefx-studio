"""FastAPI web server for phone-based party control."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from src.utils.network import detect_local_ip

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

# YouTube services (set in start_server when youtube is enabled)
_youtube_service = None  # YouTubeService | None
_tv_player = None  # TVPlayer | None


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


@app.post("/api/effects/set")
async def set_effect_body(request: Request) -> JSONResponse:
    """Switch the active effect (body-based, handles special chars)."""
    body = await request.json()
    name = body.get("name", "")
    engine = _get_engine()
    if name not in engine.get_renderer_names():
        return JSONResponse({"error": f"Unknown effect: {name}"}, status_code=404)
    engine.set_renderer(name)
    return JSONResponse({"effect": name})


@app.post("/api/effects/reload")
async def reload_effects() -> JSONResponse:
    """Hot-reload all effect renderer modules from disk."""
    try:
        names = _get_engine().reload_effects()
        return JSONResponse({"effects": names, "reloaded": True})
    except Exception as exc:
        logger.exception("Failed to reload effects")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/effects/{name:path}")
async def set_effect(name: str) -> JSONResponse:
    """Switch the active effect (path-based, legacy)."""
    engine = _get_engine()
    if name not in engine.get_renderer_names():
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


@app.get("/api/qr")
async def get_qr_image() -> Response:
    """Return the current QR code image for phone access."""
    qr_png = _get_engine().get_qr_png()
    if qr_png is None:
        return JSONResponse({"error": "QR not available"}, status_code=503)
    return Response(content=qr_png, media_type="image/png")


@app.post("/api/qr/toggle")
async def toggle_qr() -> JSONResponse:
    """Toggle the preview QR overlay."""
    engine = _get_engine()
    visible = engine.toggle_qr_visibility()
    return JSONResponse({"qr_visible": visible, "hub_url": engine.hub_url})


@app.post("/api/qr/refresh")
async def refresh_qr() -> JSONResponse:
    """Regenerate the QR using the current detected LAN IP."""
    engine = _get_engine()
    config = _get_config()
    hub_url = f"http://{detect_local_ip()}:{config.web.port}"
    try:
        engine.set_hub_url(hub_url)
    except RuntimeError as exc:
        logger.exception("Failed to refresh QR")
        return JSONResponse({"error": str(exc)}, status_code=503)
    return JSONResponse({"qr_visible": engine.qr_visible, "hub_url": engine.hub_url})


# ---------------------------------------------------------------------------
# Avatar endpoints
# ---------------------------------------------------------------------------

@app.post("/api/avatar")
async def upload_avatar(file: UploadFile) -> JSONResponse:
    """Upload a custom avatar image for the Sprite Puppet effect."""
    engine = _get_engine()
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:  # 10MB limit
        return JSONResponse({"error": "File too large (max 10MB)"}, status_code=413)
    try:
        path = engine.set_avatar(data)
        return JSONResponse({"avatar": str(path)})
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.delete("/api/avatar")
async def clear_avatar() -> JSONResponse:
    """Remove the custom avatar."""
    _get_engine().clear_avatar()
    return JSONResponse({"avatar": None})


@app.post("/api/snowfall-scale")
async def set_snowfall_scale(request: Request) -> JSONResponse:
    """Set the snowfall sprite size scale."""
    body = await request.json()
    value = body.get("value")
    if value is None:
        return JSONResponse({"error": "Missing 'value'"}, status_code=400)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return JSONResponse({"error": "Invalid value"}, status_code=400)
    _get_engine().set_snowfall_scale(value)
    return JSONResponse({"snowfall_scale": _get_engine().snowfall_scale})


@app.post("/api/snowfall-density")
async def set_snowfall_density(request: Request) -> JSONResponse:
    """Set the snowfall sprite density."""
    body = await request.json()
    value = body.get("value")
    if value is None:
        return JSONResponse({"error": "Missing 'value'"}, status_code=400)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return JSONResponse({"error": "Invalid value"}, status_code=400)
    _get_engine().set_snowfall_density(value)
    return JSONResponse({"snowfall_density": _get_engine().snowfall_density})


@app.post("/api/snowfall-custom-scale")
async def set_snowfall_custom_scale(request: Request) -> JSONResponse:
    """Set the Snowfall Custom sprite size scale."""
    body = await request.json()
    value = body.get("value")
    if value is None:
        return JSONResponse({"error": "Missing 'value'"}, status_code=400)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return JSONResponse({"error": "Invalid value"}, status_code=400)
    _get_engine().set_snowfall_custom_scale(value)
    return JSONResponse({"snowfall_custom_scale": _get_engine().snowfall_custom_scale})


@app.post("/api/snowfall-custom-density")
async def set_snowfall_custom_density(request: Request) -> JSONResponse:
    """Set the Snowfall Custom sprite density."""
    body = await request.json()
    value = body.get("value")
    if value is None:
        return JSONResponse({"error": "Missing 'value'"}, status_code=400)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return JSONResponse({"error": "Invalid value"}, status_code=400)
    _get_engine().set_snowfall_custom_density(value)
    return JSONResponse({"snowfall_custom_density": _get_engine().snowfall_custom_density})


@app.post("/api/brightness")
async def set_brightness(request: Request) -> JSONResponse:
    """Set the global brightness multiplier."""
    body = await request.json()
    value = body.get("value")
    if value is None:
        return JSONResponse({"error": "Missing 'value'"}, status_code=400)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return JSONResponse({"error": "Invalid value"}, status_code=400)
    _get_engine().set_brightness(value)
    return JSONResponse({"brightness": _get_engine().brightness})


@app.post("/api/puppet-opacity")
async def set_puppet_opacity(request: Request) -> JSONResponse:
    """Set the Custom Avatar opacity."""
    body = await request.json()
    value = body.get("value")
    if value is None:
        return JSONResponse({"error": "Missing 'value'"}, status_code=400)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return JSONResponse({"error": "Invalid value"}, status_code=400)
    _get_engine().set_puppet_opacity(value)
    return JSONResponse({"puppet_opacity": _get_engine().puppet_opacity})


@app.get("/api/avatar")
async def get_avatar_status() -> JSONResponse:
    """Check if a custom avatar is set."""
    has_avatar = _get_engine().avatar_image is not None
    return JSONResponse({"has_avatar": has_avatar})


# ---------------------------------------------------------------------------
# Snowfall custom image endpoints
# ---------------------------------------------------------------------------

_SNOWFALL_DIR = Path("data/snowfall")


def _remove_background(img: np.ndarray) -> np.ndarray:
    """Remove background from an image using saturation-based approach.

    Works well for images with white/light backgrounds.
    Returns BGRA image with transparent background.
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        if np.mean(alpha < 250) > 0.2:
            return img
        bgr = img[:, :, :3]
    else:
        bgr = img

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # High saturation = foreground, low saturation + high value = white bg
    white_diff = np.clip((val - 200) * 2.0, 0, 255)
    alpha = np.clip(sat * 2.5 + white_diff * 0.5, 0, 255)

    # Also consider dark areas as foreground
    dark_mask = val < 80
    alpha[dark_mask] = np.clip(alpha[dark_mask] + (80 - val[dark_mask]) * 3, 0, 255)

    # Slight blur to smooth edges
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    return np.dstack([bgr, alpha])


@app.get("/api/snowfall/images")
async def get_snowfall_images() -> JSONResponse:
    """List current custom snowfall images."""
    images = []
    if _SNOWFALL_DIR.exists():
        for p in sorted(_SNOWFALL_DIR.iterdir()):
            if p.suffix.lower() == ".png" and not p.name.startswith("."):
                images.append({"name": p.name, "path": str(p)})
    return JSONResponse({"images": images, "max": 2})


@app.post("/api/snowfall/upload")
async def upload_snowfall_image(file: UploadFile) -> JSONResponse:
    """Upload a custom snowfall sprite image (max 2)."""
    _SNOWFALL_DIR.mkdir(parents=True, exist_ok=True)

    # Check count
    existing = [
        p for p in _SNOWFALL_DIR.iterdir()
        if p.suffix.lower() == ".png" and not p.name.startswith(".")
    ]
    if len(existing) >= 2:
        return JSONResponse(
            {"error": "Maximum 2 custom images. Delete one first."},
            status_code=409,
        )

    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        return JSONResponse({"error": "File too large (max 10MB)"}, status_code=413)

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    # Remove background and save as RGBA PNG
    processed = _remove_background(img)

    # Generate filename
    import time as _time
    stem = f"sprite_{int(_time.time())}"
    path = _SNOWFALL_DIR / f"{stem}.png"
    cv2.imwrite(str(path), processed)

    # Trigger renderer reload
    engine = _get_engine()
    for r in engine._renderers:
        if hasattr(r, 'reload_custom_images'):
            r.reload_custom_images()

    logger.info("Snowfall image uploaded: %s (%dx%d)", path, processed.shape[1], processed.shape[0])
    return JSONResponse({"name": path.name, "path": str(path)})


@app.delete("/api/snowfall/images/{name}")
async def delete_snowfall_image(name: str) -> JSONResponse:
    """Delete a custom snowfall image."""
    path = _SNOWFALL_DIR / name
    if not path.exists() or not str(path.resolve()).startswith(str(_SNOWFALL_DIR.resolve())):
        return JSONResponse({"error": "Not found"}, status_code=404)
    path.unlink()

    engine = _get_engine()
    for r in engine._renderers:
        if hasattr(r, 'reload_custom_images'):
            r.reload_custom_images()

    logger.info("Snowfall image deleted: %s", name)
    return JSONResponse({"deleted": name})


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
# YouTube DJ (Step 6)
# ---------------------------------------------------------------------------

@app.get("/api/youtube/search")
async def youtube_search(q: str = "") -> JSONResponse:
    """Search YouTube for videos."""
    if not _youtube_service:
        return JSONResponse({"results": [], "error": "YouTube not enabled"})
    if not q.strip():
        return JSONResponse({"results": []})
    try:
        results = await asyncio.to_thread(_youtube_service.search, q.strip())
        return JSONResponse({"results": results})
    except Exception as exc:
        logger.exception("YouTube search failed")
        return JSONResponse({"results": [], "error": str(exc)}, status_code=500)


@app.post("/api/youtube/play")
async def youtube_play(request: Request) -> JSONResponse:
    """Play a YouTube video on the TV."""
    if not _youtube_service or not _tv_player:
        return JSONResponse({"error": "YouTube not enabled"}, status_code=501)
    body = await request.json()
    url = body.get("url", "")
    title = body.get("title", "")
    if not url:
        return JSONResponse({"error": "Missing 'url'"}, status_code=400)
    stream_url = await asyncio.to_thread(_youtube_service.get_stream_url, url)
    if not stream_url:
        return JSONResponse({"error": "Could not extract stream URL"}, status_code=502)
    ok = _tv_player.play(stream_url, title=title)
    if not ok:
        return JSONResponse({"error": "Playback failed (mpv not installed?)"}, status_code=503)
    return JSONResponse({"status": "playing", "title": title})


@app.post("/api/youtube/stop")
async def youtube_stop() -> JSONResponse:
    """Stop YouTube playback."""
    if _tv_player:
        _tv_player.stop()
    return JSONResponse({"status": "idle"})


@app.get("/api/youtube/status")
async def youtube_status() -> JSONResponse:
    """YouTube playback status."""
    if _tv_player:
        return JSONResponse(_tv_player.get_status())
    return JSONResponse({"status": "idle", "title": ""})


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
            if _tv_player:
                state["youtube"] = _tv_player.get_status()
            else:
                state["youtube"] = {"status": "idle", "title": ""}
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
    global _engine, _config, _youtube_service, _tv_player
    _engine = engine
    _config = config

    if config.youtube.enabled:
        from src.services.youtube import YouTubeService
        from src.services.tv_player import TVPlayer
        _youtube_service = YouTubeService(
            max_results=config.youtube.max_search_results,
            max_duration=config.youtube.max_duration_seconds,
        )
        _tv_player = TVPlayer()

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
