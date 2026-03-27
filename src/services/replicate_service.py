"""AI job service powered by Replicate HTTP API.

Uses httpx directly to avoid SDK compatibility issues with Python 3.14.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)

REPLICATE_API = "https://api.replicate.com/v1"
POLL_INTERVAL = 2.0
TERMINAL_STATES = {"succeeded", "failed", "canceled"}


class JobStatus(Enum):
    """Status of an AI job."""
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class JobType(Enum):
    """Type of AI job."""
    FACE_SWAP = "face_swap"
    FACE_SWAP_VIDEO = "face_swap_video"
    AI_EDIT = "ai_edit"
    AI_VIDEO = "ai_video"


@dataclass
class AIJob:
    """Tracks a single AI processing job."""
    job_id: str
    job_type: JobType
    status: JobStatus
    created_at: float
    result_path: str | None = None
    error: str | None = None
    replicate_id: str | None = None
    prompt: str | None = None
    _upload_paths: list[str] = field(default_factory=list)


def _file_to_data_uri(path: str) -> str:
    """Convert a local file to a base64 data URI for the Replicate API."""
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class ReplicateService:
    """Manages AI jobs via Replicate HTTP API."""

    def __init__(
        self,
        config,
        on_status_change: Callable[[AIJob], None] | None = None,
    ) -> None:
        self._config = config
        self._on_status_change = on_status_change
        self._jobs: dict[str, AIJob] = {}
        self._lock = threading.Lock()

        self._token = os.environ.get("REPLICATE_API_TOKEN", "")
        if self._token:
            logger.info("ReplicateService: API token configured")
        else:
            logger.warning(
                "ReplicateService: REPLICATE_API_TOKEN not set — AI Lab will be unavailable"
            )

        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
        Path(config.uploads_dir).mkdir(parents=True, exist_ok=True)

    @property
    def is_configured(self) -> bool:
        """Whether the Replicate API token is set."""
        return bool(self._token)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Public job submission methods
    # ------------------------------------------------------------------

    def face_swap_image(self, source_path: str, target_path: str) -> str:
        """Submit an image face swap job. Returns job_id."""
        self._require_token()
        job = self._create_job(JobType.FACE_SWAP, [source_path, target_path])

        def run() -> None:
            try:
                prediction_id = self._create_prediction(
                    self._config.replicate_model_face_swap,
                    {
                        "swap_image": _file_to_data_uri(source_path),
                        "target_image": _file_to_data_uri(target_path),
                    },
                )
                self._update_job(job.job_id, status=JobStatus.PROCESSING, replicate_id=prediction_id)
                self._poll_and_download(job.job_id, prediction_id)
            except Exception as exc:
                logger.exception("Face swap job %s failed", job.job_id)
                self._update_job(job.job_id, status=JobStatus.FAILED, error=str(exc))

        threading.Thread(target=run, daemon=True).start()
        return job.job_id

    def face_swap_video(self, source_path: str, target_video_path: str) -> str:
        """Submit a video face swap job. Returns job_id."""
        self._require_token()
        job = self._create_job(JobType.FACE_SWAP_VIDEO, [source_path, target_video_path])

        def run() -> None:
            try:
                prediction_id = self._create_prediction(
                    self._config.replicate_model_video_swap,
                    {
                        "source_image": _file_to_data_uri(source_path),
                        "target_image": _file_to_data_uri(target_video_path),
                    },
                )
                self._update_job(job.job_id, status=JobStatus.PROCESSING, replicate_id=prediction_id)
                self._poll_and_download(job.job_id, prediction_id)
            except Exception as exc:
                logger.exception("Video face swap job %s failed", job.job_id)
                self._update_job(job.job_id, status=JobStatus.FAILED, error=str(exc))

        threading.Thread(target=run, daemon=True).start()
        return job.job_id

    def edit_image(self, image_path: str, prompt: str) -> str:
        """Submit an AI image edit job. Returns job_id."""
        self._require_token()
        job = self._create_job(JobType.AI_EDIT, [image_path], prompt=prompt)

        def run() -> None:
            try:
                prediction_id = self._create_prediction(
                    self._config.replicate_model_edit,
                    {
                        "input_image": _file_to_data_uri(image_path),
                        "prompt": prompt,
                        "aspect_ratio": "match_input_image",
                    },
                )
                self._update_job(job.job_id, status=JobStatus.PROCESSING, replicate_id=prediction_id)
                self._poll_and_download(job.job_id, prediction_id)
            except Exception as exc:
                logger.exception("AI edit job %s failed", job.job_id)
                self._update_job(job.job_id, status=JobStatus.FAILED, error=str(exc))

        threading.Thread(target=run, daemon=True).start()
        return job.job_id

    def generate_video(self, face_path: str, prompt: str) -> str:
        """Submit an AI video generation job (two-step: image gen → video gen). Returns job_id."""
        self._require_token()
        job = self._create_job(JobType.AI_VIDEO, [face_path], prompt=prompt)

        def run() -> None:
            try:
                # Step 1: Generate a still image with the face in the described scene
                logger.info("AI video job %s — step 1: generating reference image", job.job_id)
                img_pred_id = self._create_prediction(
                    self._config.replicate_model_edit,
                    {
                        "input_image": _file_to_data_uri(face_path),
                        "prompt": prompt,
                    },
                )
                img_result = self._wait_for_prediction(img_pred_id)
                if img_result is None or img_result.get("status") != "succeeded":
                    error = img_result.get("error", "Image generation step failed") if img_result else "Lost"
                    self._update_job(job.job_id, status=JobStatus.FAILED, error=str(error))
                    return

                image_url = self._extract_output_url(img_result)
                if not image_url:
                    self._update_job(job.job_id, status=JobStatus.FAILED, error="No image output from step 1")
                    return

                # Step 2: Animate the image into a video
                logger.info("AI video job %s — step 2: animating to video", job.job_id)
                vid_pred_id = self._create_prediction(
                    self._config.replicate_model_video_gen,
                    {
                        "prompt": prompt,
                        "first_frame_image": image_url,
                    },
                )
                self._update_job(job.job_id, status=JobStatus.PROCESSING, replicate_id=vid_pred_id)
                self._poll_and_download(job.job_id, vid_pred_id)
            except Exception as exc:
                logger.exception("AI video job %s failed", job.job_id)
                self._update_job(job.job_id, status=JobStatus.FAILED, error=str(exc))

        threading.Thread(target=run, daemon=True).start()
        return job.job_id

    def get_job(self, job_id: str) -> AIJob | None:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_all_jobs(self) -> list[AIJob]:
        """Get all jobs, newest first."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    # ------------------------------------------------------------------
    # Replicate HTTP API helpers
    # ------------------------------------------------------------------

    def _create_prediction(self, model: str, input_data: dict[str, Any]) -> str:
        """Create a Replicate prediction via HTTP API. Returns prediction ID."""
        # Model format: "owner/name" → use models endpoint
        url = f"{REPLICATE_API}/models/{model}/predictions"
        resp = httpx.post(
            url,
            headers=self._headers(),
            json={"input": input_data},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        prediction_id = data["id"]
        logger.info("Created Replicate prediction %s for model %s", prediction_id, model)
        return prediction_id

    def _get_prediction(self, prediction_id: str) -> dict:
        """Get prediction status from Replicate API."""
        resp = httpx.get(
            f"{REPLICATE_API}/predictions/{prediction_id}",
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _wait_for_prediction(self, prediction_id: str) -> dict | None:
        """Block until a Replicate prediction reaches a terminal state."""
        while True:
            try:
                pred = self._get_prediction(prediction_id)
            except Exception:
                logger.exception("Failed to poll prediction %s", prediction_id)
                return None
            if pred.get("status") in TERMINAL_STATES:
                return pred
            time.sleep(POLL_INTERVAL)

    @staticmethod
    def _extract_output_url(prediction: dict) -> str | None:
        """Get the first usable URL from a prediction output."""
        output = prediction.get("output")
        if isinstance(output, str):
            return output
        if isinstance(output, list) and output:
            return str(output[0])
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_token(self) -> None:
        if not self._token:
            raise RuntimeError("REPLICATE_API_TOKEN is not set")

    def _create_job(
        self,
        job_type: JobType,
        upload_paths: list[str],
        prompt: str | None = None,
    ) -> AIJob:
        job_id = uuid.uuid4().hex[:12]
        job = AIJob(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            created_at=time.time(),
            prompt=prompt,
            _upload_paths=list(upload_paths),
        )
        with self._lock:
            self._jobs[job_id] = job
        self._notify(job)
        logger.info("Created %s job %s", job_type.value, job_id)
        return job

    def _update_job(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)
        self._notify(job)
        if job.status in (JobStatus.DONE, JobStatus.FAILED):
            self._cleanup_uploads(job)

    def _notify(self, job: AIJob) -> None:
        if self._on_status_change:
            try:
                self._on_status_change(job)
            except Exception:
                logger.exception("on_status_change callback failed for job %s", job.job_id)

    def _cleanup_uploads(self, job: AIJob) -> None:
        for p in job._upload_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

    def _poll_and_download(self, job_id: str, prediction_id: str) -> None:
        """Poll a Replicate prediction until terminal, then download result."""
        result = self._wait_for_prediction(prediction_id)
        if result is None or result.get("status") == "failed":
            error_msg = result.get("error", "Prediction failed") if result else "Prediction lost"
            self._update_job(job_id, status=JobStatus.FAILED, error=str(error_msg))
            return
        if result.get("status") == "canceled":
            self._update_job(job_id, status=JobStatus.FAILED, error="Job was canceled")
            return

        output_url = self._extract_output_url(result)
        if not output_url:
            self._update_job(job_id, status=JobStatus.FAILED, error="No output URL in result")
            return

        result_path = self._download_file(output_url, job_id)
        if result_path:
            self._update_job(job_id, status=JobStatus.DONE, result_path=result_path)
            logger.info("Job %s done → %s", job_id, result_path)
        else:
            self._update_job(job_id, status=JobStatus.FAILED, error="Failed to download result")

    def _download_file(self, url: str, job_id: str) -> str | None:
        """Download a file from URL and save to results directory."""
        try:
            ext = ".png"
            url_lower = url.lower().split("?")[0]
            for candidate in (".mp4", ".webm", ".mov", ".gif", ".jpg", ".jpeg", ".png", ".webp"):
                if url_lower.endswith(candidate):
                    ext = candidate
                    break

            out_path = Path(self._config.results_dir) / f"{job_id}{ext}"
            with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        f.write(chunk)
            return str(out_path)
        except Exception:
            logger.exception("Failed to download result for job %s", job_id)
            return None

    def job_to_dict(self, job: AIJob) -> dict:
        """Serialize a job to a JSON-safe dict."""
        return {
            "job_id": job.job_id,
            "job_type": job.job_type.value,
            "status": job.status.value,
            "created_at": job.created_at,
            "result_url": f"/api/ai/results/{Path(job.result_path).name}" if job.result_path else None,
            "error": job.error,
            "prompt": job.prompt,
        }
