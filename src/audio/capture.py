"""Audio capture with real-time bass energy extraction and recording."""

import logging
import struct
import threading
import time
import wave
from pathlib import Path

import numpy as np

from src.utils.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures audio from a device or file and extracts bass energy.

    Runs in a background thread. Performs FFT on each audio chunk
    and extracts the magnitude of the bass frequency band.

    Supports two modes:
    - Microphone input (default): captures from system audio device.
    - File input: reads a WAV file and processes it at realtime speed.

    Args:
        config: Audio configuration.
        file_path: Optional path to a WAV file. If provided, uses file
            instead of microphone.
    """

    def __init__(self, config: AudioConfig, file_path: str | Path | None = None) -> None:
        self._config = config
        self._file_path = Path(file_path) if file_path else None
        self._bass_energy = 0.0
        self._stream = None
        self._pa = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._recording = False
        self._recording_chunks: list[bytes] = []
        self._recording_lock = threading.Lock()

    @property
    def bass_energy(self) -> float:
        """Current bass energy level, normalized 0.0-1.0."""
        return self._bass_energy

    def start(self) -> None:
        """Start capturing audio in a background thread."""
        if self._file_path:
            self._start_file()
        else:
            self._start_mic()

    def _start_file(self) -> None:
        """Start reading audio from a WAV file."""
        try:
            from scipy.io import wavfile
        except ImportError:
            logger.warning("scipy not installed — audio file input disabled")
            return

        if not self._file_path.exists():
            logger.warning("Audio file not found: %s", self._file_path)
            return

        try:
            sample_rate, data = wavfile.read(str(self._file_path))
            # Convert to mono float32
            if data.ndim > 1:
                data = data.mean(axis=1)
            self._file_data = data.astype(np.float32)
            self._file_sample_rate = sample_rate
            logger.info(
                "Audio file loaded: %s (%d Hz, %.1f sec)",
                self._file_path.name,
                sample_rate,
                len(self._file_data) / sample_rate,
            )

            self._running = True
            self._thread = threading.Thread(target=self._file_loop, daemon=True)
            self._thread.start()
            logger.info("Audio file playback started")

        except Exception:
            logger.warning("Could not read audio file — audio disabled", exc_info=True)

    def _start_mic(self) -> None:
        """Start capturing from microphone."""
        try:
            import pyaudio
        except ImportError:
            logger.warning("pyaudio not installed — audio disabled")
            return

        try:
            self._pa = pyaudio.PyAudio()

            device_index = self._config.device_index
            if device_index is not None:
                device_info = self._pa.get_device_info_by_index(device_index)
            else:
                device_info = self._pa.get_default_input_device_info()
                device_index = int(device_info["index"])

            logger.info(
                "Opening audio device: %s (index %d)",
                device_info.get("name", "unknown"),
                device_index,
            )

            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._config.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self._config.chunk_size,
            )

            self._running = True
            self._thread = threading.Thread(target=self._mic_loop, daemon=True)
            self._thread.start()
            logger.info("Audio capture started")

        except Exception:
            logger.warning("Could not open audio device — audio disabled", exc_info=True)
            self._cleanup_pa()

    def stop(self) -> None:
        """Stop capturing audio and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._cleanup_pa()
        logger.info("Audio capture stopped")

    def start_recording(self) -> None:
        """Start buffering raw audio chunks for later WAV export."""
        with self._recording_lock:
            self._recording_chunks = []
            self._recording = True
        logger.info("Audio recording started")

    def stop_recording(self, output_path: str | Path) -> Path | None:
        """Stop recording and write buffered audio to a WAV file.

        Args:
            output_path: Path for the output WAV file.

        Returns:
            Path to the saved WAV, or None if no audio was captured.
        """
        with self._recording_lock:
            self._recording = False
            chunks = self._recording_chunks
            self._recording_chunks = []

        if not chunks:
            logger.info("No audio chunks captured")
            return None

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sample_rate = self._config.sample_rate
        try:
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(b"".join(chunks))
            logger.info("Audio recording saved: %s (%d chunks)", path, len(chunks))
            return path
        except Exception:
            logger.exception("Failed to write audio WAV")
            return None

    def _cleanup_pa(self) -> None:
        """Release PyAudio resources."""
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def _process_samples(self, samples: np.ndarray, sample_rate: int,
                         state: dict) -> None:
        """Analyze a chunk of audio samples and update bass_energy.

        Args:
            samples: Float32 audio samples for this chunk.
            sample_rate: Sample rate of the audio.
            state: Mutable dict holding rolling_avg and smoothed values.
        """
        cfg = self._config
        chunk_size = len(samples)

        # FFT and extract bass magnitude
        freqs = np.fft.rfftfreq(chunk_size, d=1.0 / sample_rate)
        bass_mask = (freqs >= cfg.bass_low_hz) & (freqs <= cfg.bass_high_hz)
        spectrum = np.abs(np.fft.rfft(samples))
        bass_magnitude = float(np.mean(spectrum[bass_mask])) if np.any(bass_mask) else 0.0

        # Noise gate: ignore anything below the threshold
        if bass_magnitude < cfg.noise_gate:
            bass_magnitude = 0.0

        # Rolling average tracks the baseline
        rolling_avg = state["rolling_avg"]
        rolling_avg = rolling_avg * 0.93 + bass_magnitude * 0.07
        state["rolling_avg"] = rolling_avg

        # Energy = ratio above baseline
        if rolling_avg > cfg.noise_gate:
            raw = max(0.0, (bass_magnitude - rolling_avg * 0.7) / max(rolling_avg, 1.0))
        else:
            raw = 0.0

        # Apply sensitivity and clamp
        raw = min(raw * cfg.sensitivity, 1.0)

        # Smoothing: fast attack, fast decay
        smoothed = state["smoothed"]
        if raw > smoothed:
            smoothed = cfg.smoothing_attack * raw + (1 - cfg.smoothing_attack) * smoothed
        else:
            smoothed = smoothed * 0.8
        state["smoothed"] = smoothed

        self._bass_energy = smoothed

    def _mic_loop(self) -> None:
        """Background thread: read mic chunks, extract bass energy."""
        state = {"rolling_avg": 0.0, "smoothed": 0.0}

        while self._running and self._stream is not None:
            try:
                data = self._stream.read(self._config.chunk_size, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                self._process_samples(samples, self._config.sample_rate, state)

                # Buffer raw audio if recording
                if self._recording:
                    with self._recording_lock:
                        if self._recording:
                            self._recording_chunks.append(data)
            except Exception:
                logger.debug("Audio read error", exc_info=True)
                continue

    def _file_loop(self) -> None:
        """Background thread: read file chunks at realtime speed."""
        state = {"rolling_avg": 0.0, "smoothed": 0.0}
        chunk_size = self._config.chunk_size
        sample_rate = self._file_sample_rate
        data = self._file_data
        chunk_duration = chunk_size / sample_rate
        pos = 0

        while self._running:
            start_t = time.monotonic()

            # Get next chunk, loop if at end
            end = pos + chunk_size
            if end > len(data):
                pos = 0
                end = chunk_size
                logger.debug("Audio file looped")

            samples = data[pos:end]
            pos = end

            self._process_samples(samples, sample_rate, state)

            # Sleep to maintain realtime pace
            elapsed = time.monotonic() - start_t
            sleep_time = chunk_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
