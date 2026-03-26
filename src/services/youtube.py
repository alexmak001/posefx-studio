"""YouTube search and stream URL extraction via yt-dlp."""

from __future__ import annotations

import logging
import threading
from typing import Any

import yt_dlp

logger = logging.getLogger(__name__)

_YDL_OPTS_SEARCH: dict[str, Any] = {
    "quiet": True,
    "no_warnings": True,
    "extract_flat": "in_playlist",
    "skip_download": True,
    "ignoreerrors": True,
}

_YDL_OPTS_STREAM: dict[str, Any] = {
    "quiet": True,
    "no_warnings": True,
    "format": "best[height<=1080][ext=mp4]/best[height<=720]/best",
    "skip_download": True,
}


def _fmt_duration(seconds: float | int | None) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    if not seconds:
        return "?"
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}:{seconds % 60:02d}"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


class YouTubeService:
    """Search and stream YouTube videos using yt-dlp."""

    def __init__(self, max_results: int = 10, max_duration: int = 600) -> None:
        self._max_results = max_results
        self._max_duration = max_duration
        self._lock = threading.Lock()

    def search(self, query: str) -> list[dict[str, Any]]:
        """Search YouTube and return results."""
        search_url = f"ytsearch{self._max_results}:{query}"
        with self._lock:
            with yt_dlp.YoutubeDL(_YDL_OPTS_SEARCH) as ydl:
                info = ydl.extract_info(search_url, download=False)

        results: list[dict[str, Any]] = []
        for entry in (info or {}).get("entries", []) or []:
            if entry is None:
                continue
            duration = entry.get("duration") or 0
            if isinstance(duration, float):
                duration = int(duration)
            if self._max_duration and duration > self._max_duration:
                continue
            vid_id = entry.get("id", "")
            thumb = entry.get("thumbnail") or entry.get("thumbnails", [{}])[0].get("url", "") if entry.get("thumbnails") else entry.get("thumbnail", "")
            if not thumb and vid_id:
                thumb = f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"
            results.append({
                "id": vid_id,
                "title": entry.get("title", "Unknown"),
                "thumbnail": thumb,
                "duration": duration,
                "duration_str": _fmt_duration(duration),
                "url": entry.get("webpage_url")
                    or f"https://www.youtube.com/watch?v={vid_id}",
            })
        return results

    def get_stream_url(self, video_url: str) -> str | None:
        """Extract the direct stream URL for playback."""
        try:
            with self._lock:
                with yt_dlp.YoutubeDL(_YDL_OPTS_STREAM) as ydl:
                    info = ydl.extract_info(video_url, download=False)
            return info.get("url") if info else None
        except Exception:
            logger.exception("Failed to extract stream URL for %s", video_url)
            return None
