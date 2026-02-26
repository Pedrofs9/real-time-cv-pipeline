from celery import Task, signals
from core.celery_config import celery_app
from pipeline.video import get_redis_client
from core.config import settings
from workers.worker_init import get_processor, load_model, unload_model
from pipeline.renderer import load_detections_from_redis, render_annotated_video
import logging
import os
import time
from pathlib import Path
import json
import traceback
from core.metrics import (
    VIDEO_JOBS_COMPLETED, VIDEO_PROCESSING_DURATION, VIDEO_FRAMES_PER_JOB,
    MODE_WATCH, MODE_DOWNLOAD, GPU_MEMORY_USED_MB
)
from prometheus_client import start_http_server, Gauge
import torch
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

# ── Worker-side metrics (scraped from worker:8001) ────────────────────────────
WORKER_CPU_PERCENT = Gauge(
    'cv_worker_cpu_percent',
    'Worker process CPU usage percent'
)
WORKER_MEMORY_MB = Gauge(
    'cv_worker_memory_mb',
    'Worker process RSS memory in MB'
)
WORKER_GPU_UTIL_PERCENT = Gauge(
    'cv_worker_gpu_util_percent',
    'GPU utilization percent (requires nvidia-ml-py). 0 if unavailable.'
)

# ── pynvml init — optional, degrades gracefully on CPU-only hosts ─────────────
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
    nvmlInit()
    _nvml_handle = nvmlDeviceGetHandleByIndex(0)
    _nvml_available = True
    logger.info("✅ pynvml loaded — GPU utilization tracking enabled")
except Exception as e:
    _nvml_handle = None
    _nvml_available = False
    logger.info(f"pynvml not available — GPU util will report 0 ({e})")


def _safe_meta_update(job_id: str, mapping: dict, expire: bool = True):
    """Update job metadata — non-critical, never raises."""
    try:
        redis_client = get_redis_client()
        redis_client.hset(f"job:{job_id}:meta", mapping=mapping)
        if expire:
            redis_client.expire(f"job:{job_id}:meta", settings.results_ttl_seconds)
    except Exception as e:
        logger.warning(f"Redis meta update failed for {job_id} (non-critical): {e}")


@signals.worker_init.connect
def init_worker(**kwargs):
    import threading
    logger.info("Worker initializing ...")

    # Model load is fatal — worker must not start without it
    try:
        load_model()
    except Exception as e:
        logger.error(f"FATAL: Model failed to load: {e}", exc_info=True)
        raise

    # Metrics server — one server on 8001, all worker metrics served here
    try:
        start_http_server(settings.metrics_port)
        logger.info(f"✅ Metrics server started on port {settings.metrics_port}")
    except OSError as e:
        logger.warning(f"Metrics port {settings.metrics_port} already in use: {e}")

    # Prime psutil cpu_percent — first call always returns 0.0, must be called
    # once before the loop so subsequent calls return meaningful values
    _process = psutil.Process()
    _process.cpu_percent()

    def _update_metrics():
        """
        Background thread — updates all worker-side Prometheus gauges every 10s.
        Covers: GPU memory, GPU utilization, worker CPU, worker RAM.
        """
        process = psutil.Process()
        while True:
            try:
                # GPU memory (torch)
                GPU_MEMORY_USED_MB.set(
                    torch.cuda.memory_allocated() / (1024 * 1024)
                    if torch.cuda.is_available() else 0
                )

                # GPU utilization (pynvml) — 0 if not available
                if _nvml_available and _nvml_handle is not None:
                    util = nvmlDeviceGetUtilizationRates(_nvml_handle)
                    WORKER_GPU_UTIL_PERCENT.set(util.gpu)
                else:
                    WORKER_GPU_UTIL_PERCENT.set(0)

                # Worker process CPU + RAM
                WORKER_CPU_PERCENT.set(process.cpu_percent())
                WORKER_MEMORY_MB.set(process.memory_info().rss / (1024 * 1024))

            except Exception as e:
                logger.debug(f"Metrics update error (non-critical): {e}")

            time.sleep(10)

    threading.Thread(target=_update_metrics, daemon=True).start()
    logger.info("✅ Worker initialization complete")


@signals.worker_shutdown.connect
def shutdown_worker(**kwargs):
    logger.info("Worker shutting down...")
    unload_model()


class VideoProcessingTask(Task):
    """Custom task with error handling and result storage."""

    def _safe_redis_set(self, key, ttl, value):
        try:
            redis_client = get_redis_client()
            redis_client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.warning(f"Failed to store result in Redis (non-critical): {e}")

    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {task_id} completed successfully")
        result_to_store = {
            **(retval if isinstance(retval, dict) else {"result": retval}),
            "celery_task_id": task_id,
            "celery_status": "SUCCESS",
        }
        self._safe_redis_set(f"job:{task_id}", settings.results_ttl_seconds, result_to_store)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")
        error_result = {
            "job_id": task_id,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "celery_task_id": task_id,
            "celery_status": "FAILURE"
        }
        self._safe_redis_set(f"job:{task_id}", settings.results_ttl_seconds, error_result)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning(f"Task {task_id} retrying ({self.request.retries}/3): {exc}")


# ── Watch mode task ────────────────────────────────────────────────────────────

@celery_app.task(
    base=VideoProcessingTask,
    bind=True,
    name="process_video",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    reject_on_worker_lost=True,
    task_track_started=True
)
def process_video_task(self, video_path: str, filename: str, job_id: str = None, sample_rate: int = None):
    task_id = job_id or self.request.id
    effective_sample_rate = sample_rate if sample_rate is not None else settings.watch_frame_sample_rate

    try:
        get_redis_client().incr("cv:queue:active_jobs")
    except Exception:
        pass

    _safe_meta_update(task_id, {
        "status": "processing",
        "filename": filename,
        "started_at": str(time.time()),
        "frame_sample_rate": str(effective_sample_rate),
    })

    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        processor = get_processor()
        result = processor.process_video_file(video_path, task_id, sample_rate=effective_sample_rate, mode=MODE_WATCH)
        result["celery_task_id"] = task_id
        result["celery_retries"] = self.request.retries

        logger.info(f"Task {task_id} complete: {result['status']}")

        status = result.get("status", "unknown")
        VIDEO_JOBS_COMPLETED.labels(mode=MODE_WATCH, status=status).inc()
        VIDEO_FRAMES_PER_JOB.labels(mode=MODE_WATCH).observe(result.get("frames_processed", 0))
        if result.get("timing_ms", {}).get("total_processing_ms"):
            VIDEO_PROCESSING_DURATION.labels(mode=MODE_WATCH).observe(
                result["timing_ms"]["total_processing_ms"] / 1000
            )
        return result

    except Exception as exc:
        logger.error(f"Task {task_id} failed: {exc}", exc_info=True)
        _safe_meta_update(task_id, {"status": "failed", "error": str(exc)})
        if self.request.retries < self.max_retries:
            if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
                self.retry(exc=exc)
        raise

    finally:
        try:
            get_redis_client().decr("cv:queue:active_jobs")
        except Exception:
            pass
        # Source upload is intentionally NOT deleted in watch mode.
        # The /v1/video/{job_id} endpoint serves the raw file to the browser
        # while the WebSocket stream is active. Deleting it here would cause
        # 404s for users still watching.
        # TODO: Implement TTL-based cleanup for watch uploads in a future version,
        # triggered after the WebSocket connection closes cleanly.


# ── Download mode task ─────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="process_and_render_download",
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_and_render_download_task(self, video_path: str, filename: str, job_id: str, sample_rate: int = None):
    """
    Download mode: inference at configurable sample_rate (default from settings),
    then render annotated video. Progress: inference = 0–70%, rendering = 70–100%.
    """
    effective_sample_rate = sample_rate if sample_rate is not None else settings.download_frame_sample_rate

    try:
        get_redis_client().incr("cv:queue:active_jobs")
    except Exception:
        pass

    _safe_meta_update(job_id, {
        "status": "processing",
        "filename": filename,
        "started_at": str(time.time()),
        "frame_sample_rate": str(effective_sample_rate),
        "mode": "download",
        "download_status": "inference",
        "download_progress": "0",
    })

    try:
        redis_client = get_redis_client()

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        def inference_progress_callback(frames_done: int, total_frames: int):
            if total_frames > 0:
                pct = (frames_done / total_frames) * 70
                redis_client.hset(f"job:{job_id}:meta", mapping={
                    "download_progress": f"{pct:.1f}",
                    "download_status": "inference",
                })

        processor = get_processor()
        inference_result = processor.process_video_file(
            video_path, job_id,
            sample_rate=effective_sample_rate,
            mode=MODE_DOWNLOAD,
            progress_callback=inference_progress_callback,
        )
        if inference_result.get("status") == "failed":
            raise RuntimeError(f"Inference failed: {inference_result.get('error')}")

        total_detection_frames = inference_result["frames_processed"]
        logger.info(f"[DownloadTask {job_id}] Phase 1 done: {total_detection_frames} frames")

        redis_client.hset(f"job:{job_id}:meta", "download_status", "rendering")

        ts_to_detection, sorted_ts = load_detections_from_redis(
            redis_client, job_id, total_detection_frames
        )

        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{job_id}_annotated.mp4")

        render_annotated_video(
            source_video_path=video_path,
            output_path=output_path,
            ts_to_detection=ts_to_detection,
            sorted_ts=sorted_ts,
            job_id=job_id,
            redis_client=redis_client,
            progress_start=70.0,
            progress_end=100.0,
        )

        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(f"[DownloadTask {job_id}] Done: {output_path} ({file_size_mb:.1f}MB)")

        redis_client.hset(f"job:{job_id}:meta", mapping={
            "status": "completed",
            "download_status": "ready",
            "download_progress": "100",
            "download_video_path": output_path,
            "download_generated_at": str(time.time()),
        })
        redis_client.expire(f"job:{job_id}:meta", settings.results_ttl_seconds)

        # Delete source upload now that the annotated output is fully rendered.
        # Safe to delete here — render_annotated_video has already finished reading it.
        if settings.delete_videos_after_processing and os.path.exists(video_path):
            try:
                os.unlink(video_path)
                logger.info(f"Deleted source upload after download render: {video_path}")
            except Exception as e:
                logger.warning(f"Failed to delete source upload: {e}")

        # Record total inference duration so job_duration_s query has data
        if inference_result.get("timing_ms", {}).get("total_processing_ms"):
            VIDEO_PROCESSING_DURATION.labels(mode=MODE_DOWNLOAD).observe(
                inference_result["timing_ms"]["total_processing_ms"] / 1000
            )

        return {"status": "completed", "download_status": "ready", "job_id": job_id}

    except Exception as exc:
        logger.error(f"[DownloadTask {job_id}] Failed: {exc}", exc_info=True)
        try:
            get_redis_client().hset(f"job:{job_id}:meta", mapping={
                "status": "failed",
                "download_status": "failed",
                "download_error": str(exc),
            })
        except Exception:
            pass
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        raise

    finally:
        try:
            get_redis_client().decr("cv:queue:active_jobs")
        except Exception:
            pass


# ── Post-watch render task ─────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="generate_download_video",
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
    reject_on_worker_lost=True,
)
def generate_download_video_task(self, job_id: str):
    """
    Generate annotated video from cached watch-mode detections (no new inference).
    Used when user watched first, then requests download.
    """
    logger.info(f"[RenderTask {job_id}] Generating download from cached detections")

    try:
        redis_client = get_redis_client()
        meta = redis_client.hgetall(f"job:{job_id}:meta")

        if not meta:
            raise ValueError(f"Job {job_id} not found")
        if meta.get("status") != "completed":
            raise ValueError(f"Job not completed: {meta.get('status')}")

        total_detection_frames = int(meta.get("frames_processed", 0))
        if total_detection_frames == 0:
            raise ValueError("No processed frames found")

        source_video_path = Path(settings.upload_dir) / f"{job_id}.mp4"
        if not source_video_path.exists():
            raise FileNotFoundError(f"Source video not found: {source_video_path}")

        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{job_id}_annotated.mp4")

        redis_client.hset(f"job:{job_id}:meta", mapping={
            "download_status": "rendering",
            "download_progress": "0",
        })

        ts_to_detection, sorted_ts = load_detections_from_redis(
            redis_client, job_id, total_detection_frames
        )

        render_annotated_video(
            source_video_path=str(source_video_path),
            output_path=output_path,
            ts_to_detection=ts_to_detection,
            sorted_ts=sorted_ts,
            job_id=job_id,
            redis_client=redis_client,
            progress_start=0.0,
            progress_end=100.0,
        )

        redis_client.hset(f"job:{job_id}:meta", mapping={
            "download_status": "ready",
            "download_progress": "100",
            "download_video_path": output_path,
            "download_generated_at": str(time.time()),
        })
        redis_client.expire(f"job:{job_id}:meta", settings.results_ttl_seconds)

        # Source upload no longer needed — annotated output is in outputs/.
        # Watch mode already finished using this file before this task was queued.
        if settings.delete_videos_after_processing and source_video_path.exists():
            try:
                source_video_path.unlink()
                logger.info(f"Deleted source upload after render: {source_video_path}")
            except Exception as e:
                logger.warning(f"Failed to delete source upload: {e}")

        logger.info(f"[RenderTask {job_id}] Done: {output_path}")
        return {"status": "ready", "job_id": job_id}

    except Exception as exc:
        logger.error(f"[RenderTask {job_id}] Failed: {exc}", exc_info=True)
        try:
            get_redis_client().hset(f"job:{job_id}:meta", mapping={
                "download_status": "failed",
                "download_error": str(exc),
            })
        except Exception:
            pass
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        raise