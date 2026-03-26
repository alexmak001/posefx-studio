"""TV video playback via mpv subprocess."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading

logger = logging.getLogger(__name__)

_HAS_MPV = shutil.which("mpv") is not None


class TVPlayer:
    """Plays video on the TV output using mpv."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._status = "idle"
        self._current_title = ""

        if _HAS_MPV:
            logger.info("TVPlayer: mpv found")
        else:
            logger.warning("TVPlayer: mpv not found — install mpv for YouTube playback")

    def play(self, stream_url: str, title: str = "") -> bool:
        """Start playing a video URL on the TV."""
        with self._lock:
            self._stop_locked()
            self._status = "loading"
            self._current_title = title

        if not _HAS_MPV:
            logger.warning("TVPlayer: mpv not installed, cannot play")
            with self._lock:
                self._status = "idle"
            return False

        try:
            proc = subprocess.Popen(
                [
                    "mpv",
                    "--fullscreen",
                    "--no-terminal",
                    "--force-window=immediate",
                    "--keep-open=no",
                    "--ontop",
                    stream_url,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._lock:
                self._process = proc
                self._status = "playing"

            threading.Thread(target=self._monitor, args=(proc,), daemon=True).start()
            logger.info("TVPlayer: playing '%s'", title)
            return True
        except Exception:
            logger.exception("TVPlayer: failed to start mpv")
            with self._lock:
                self._status = "idle"
            return False

    def _monitor(self, proc: subprocess.Popen) -> None:
        """Wait for mpv to exit and reset status."""
        proc.wait()
        with self._lock:
            if self._process is proc:
                self._process = None
                self._status = "idle"
                self._current_title = ""
                logger.info("TVPlayer: playback ended")

    def stop(self) -> None:
        """Stop any active playback."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        self._status = "idle"
        self._current_title = ""

    def get_status(self) -> dict:
        """Return current playback status."""
        with self._lock:
            return {
                "status": self._status,
                "title": self._current_title,
            }

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._status == "playing"
