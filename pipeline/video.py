"""
pipeline/video.py

Change vs original:
  - process_frame_sync now calls detection_pipeline.process_frame_direct_sync(frame, mode)
    instead of accepting bytes. Callers pass the numpy frame directly.
  - process_video_file no longer calls cv2.imencode / buffer.tobytes() per frame.
    The encode→decode round-trip (~17ms CPU per frame) is gone.
  - All other logic (Redis, cancellation, progress, error tracking) is unchanged.
"""

import cv2
import os
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
import logging
import threading

import redis
import numpy as np

from core.config import settings
from pipeline.detection import DetectionPipeline
from core.metrics import MODE_WATCH, MODE_DOWNLOAD
from pipeline.traffic import analyze_traffic

logger = logging.getLogger(__name__)

_redis_pool = None


def get_redis_client() -> redis.Redis:
    """Returns a Redis client using a shared connection pool, creating it if necessary."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=5,
            max_connections=10,
            retry_on_timeout=True,
        )
    return redis.Redis(connection_pool=_redis_pool)


class VideoProcessor:

    def __init__(self, detection_pipeline: DetectionPipeline) -> None:
        self.detection_pipeline = detection_pipeline
        self.active_jobs: set = set()
        self.jobs_lock = threading.Lock()
        logger.info(f"VideoProcessor initialized (upload_dir={settings.upload_dir})")

    # ── Redis helpers  ─────────────────────────────────────────────

    def _safe_redis_operation(self, operation, fallback=None, *args, **kwargs) -> Any:
        """Executes a Redis operation and returns a fallback value on any exception."""
        try:
            redis_client = get_redis_client()
            return operation(redis_client, *args, **kwargs)
        except Exception as e:
            logger.debug(f"Redis operation failed (non-critical): {e}")
            return fallback

    def _store_intermediate_result(self, job_id: str, frame_result: Dict, frame_number: int):
        """Persists a single frame's detection result to Redis and increments the frame counter."""
        def redis_operation(client):
            pipe = client.pipeline(transaction=False)
            pipe.setex(
                f"job:{job_id}:frame:{frame_number}",
                settings.results_ttl_seconds,
                json.dumps(frame_result),
            )
            pipe.hincrby(f"job:{job_id}:meta", "frames_processed", 1)
            pipe.hset(f"job:{job_id}:meta", "last_frame", frame_number)
            pipe.expire(f"job:{job_id}:meta", settings.results_ttl_seconds)
            pipe.execute()
        self._safe_redis_operation(redis_operation)

    def _store_error_frame(self, job_id: str, frame_number: int, error: str):
        """Records a frame-level processing error in Redis for later inspection."""
        def redis_operation(client):
            error_key = f"job:{job_id}:errors"
            client.hset(error_key, frame_number, json.dumps({
                "frame": frame_number, "error": error, "timestamp": time.time()
            }))
            client.expire(error_key, settings.results_ttl_seconds)
        self._safe_redis_operation(redis_operation)

    def _check_cancelled(self, job_id: str) -> bool:
        """Returns True if a cancellation request has been set for this job."""
        def redis_operation(client):
            return client.hget(f"job:{job_id}:meta", "cancelled")
        return self._safe_redis_operation(redis_operation) == "true"

    def _update_job_status(self, job_id: str, status: str, **kwargs):
        """Writes job status and optional metadata fields to Redis."""
        def redis_operation(client):
            mapping = {"status": status, **{k: str(v) for k, v in kwargs.items()}}
            pipe = client.pipeline(transaction=False)
            pipe.hset(f"job:{job_id}:meta", mapping=mapping)
            pipe.expire(f"job:{job_id}:meta", settings.results_ttl_seconds)
            pipe.execute()
        self._safe_redis_operation(redis_operation)

    # ── Frame inference ───────────────────────────────────────────────────────

    def process_frame_sync(
        self, frame: np.ndarray, timeout: int = None, mode: str = MODE_WATCH
    ) -> Dict[str, Any]:
        """Runs synchronous inference on a single numpy frame using the loaded detection pipeline."""
        if self.detection_pipeline is None:
            raise RuntimeError(
                "This VideoProcessor was created without a detection pipeline "
                "and can only be used for Redis queries (get_job_frames, get_job_status)."
            )
        try:
            return self.detection_pipeline.process_frame_direct_sync(frame, mode=mode)
        except Exception as e:
            logger.error(f"Frame processing failed: {e}", exc_info=True)
            return {"detections": [], "timing_ms": {"total_ms": 0}, "error": str(e)}
        
    # ── Content-aware frame sampling ──────────────────────────────────────────

    @staticmethod
    def _has_scene_changed(
        prev: np.ndarray,
        curr: np.ndarray,
        threshold: float = 0.02,
        scale: int = 4,
    ) -> bool:
        """
        Return True if the scene changed enough to warrant running inference.

        Downsample both frames to 1/scale resolution in grayscale, compute mean
        absolute pixel difference, compare against threshold.

        Cost: ~0.3ms at scale=4 on 1080p — negligible vs 18ms inference.

        threshold: fraction of max pixel value (255) that constitutes change.
          0.02 = 2% mean pixel shift — catches moving vehicles, ignores noise.
          Lower = more sensitive. Higher = more aggressive skipping.

        Why not optical flow? It costs ~5ms — more than we save by skipping.
        Mean abs diff is O(n), branchless, and accurate enough for traffic cams.
        """
        if prev is None:
            return True  # Always process first frame

        h, w = curr.shape[:2]
        small_h, small_w = max(1, h // scale), max(1, w // scale)

        prev_small = cv2.resize(
            cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (small_w, small_h)
        )
        curr_small = cv2.resize(
            cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (small_w, small_h)
        )

        diff = np.mean(np.abs(curr_small.astype(np.int16) - prev_small.astype(np.int16)))
        return (diff / 255.0) >= threshold

    # ── Main video processing loop ────────────────────────────────────────────

    def process_video_file(
        self,
        video_path: str,
        job_id: str,
        sample_rate: int = None,
        mode: str = MODE_WATCH,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Process a video file frame by frame.

        Key change: frames are passed directly to the model as numpy arrays.
        The previous cv2.imencode + buffer.tobytes() call that existed here
        has been removed — it was encoding to JPEG only for _infer() to
        immediately decode back to numpy, costing ~17ms CPU per frame.
        """
        effective_rate = sample_rate if sample_rate is not None else (
            settings.watch_frame_sample_rate if mode == MODE_WATCH
            else settings.download_frame_sample_rate
        )
        logger.info(f"[Job {job_id}] Starting: {video_path}, sample_rate={effective_rate}")

        with self.jobs_lock:
            self.active_jobs.add(job_id)

        self._update_job_status(job_id, "processing", started_at=time.time())

        results = {
            "job_id": job_id,
            "status": "processing",
            "frames_processed": 0,
            "video_info": {},
            "timing_ms": {},
            "detections": [],
            "failed_frames": 0,
        }

        cap = None
        frame_times = []
        failed_frames = 0
        last_progress_log = time.time()

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                results["status"] = "failed"
                results["error"] = (
                    "Video contains 0 frames. The file may be corrupted, "
                    "truncated, or use an unsupported codec."
                )
                self._update_job_status(job_id, "failed", error=results["error"])
                with self.jobs_lock:
                    self.active_jobs.discard(job_id)
                return results

            fps    = cap.get(cv2.CAP_PROP_FPS)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            video_info = {
                "total_frames":    total_frames,
                "fps":             fps,
                "width":           width,
                "height":          height,
                "duration_seconds": total_frames / fps if fps > 0 else 0,
            }

            def store_video_info(client):
                client.hset(f"job:{job_id}:meta", mapping=video_info)
            self._safe_redis_operation(store_video_info)
            results["video_info"] = video_info
            logger.info(f"[Job {job_id}] Video info: {video_info}")

            frame_count      = 0
            frames_processed = 0
            prev_frame       = None  # for content-aware scene change detection
            skipped_frames   = 0

            while True:
                ret, frame = cap.read()
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                if not ret:
                    break

                frame_count += 1

                # Cancellation check every 30 frames — one Redis round-trip per ~1s at 30fps
                if frame_count % 30 == 0 and self._check_cancelled(job_id):
                    logger.info(f"[Job {job_id}] Cancelled by user")
                    results["status"] = "cancelled"
                    break

                # ── Content-aware frame sampling ─────────────────────────────────
                # effective_rate acts as a hard minimum gap — never run inference
                # on consecutive frames even if the scene changed, which prevents
                # redundant detections on fast streams and bounds max throughput.
                # On top of that, skip frames where the scene hasn't changed enough
                # to produce different detections. For a static traffic camera this
                # skips the majority of frames in low-activity periods.
                if frame_count % effective_rate != 0:
                    continue

                if not self._has_scene_changed(prev_frame, frame, threshold=settings.scene_change_threshold):
                    skipped_frames += 1
                    continue

                if frames_processed >= settings.max_frames_per_video:
                    logger.warning(f"[Job {job_id}] Max frames reached: {frames_processed}")
                    break

                start_time = time.time()

                # ── DIRECT PATH ───────────────────────────────────────────────
                # Pass the numpy frame directly to the model, bypassing the previous
                # cv2.imencode → cv2.imdecode JPEG round-trip.
                try:
                    frame_result = self.process_frame_sync(frame, mode=mode)
                except Exception as e:
                    logger.error(f"Frame {frame_count} detection failed: {e}")
                    failed_frames += 1
                    self._store_error_frame(job_id, frame_count, str(e))
                    continue

                if "error" in frame_result:
                    failed_frames += 1
                    self._store_error_frame(job_id, frame_count, frame_result["error"])

                frame_time = (time.time() - start_time) * 1000
                frame_times.append(frame_time)

                traffic_analysis = analyze_traffic(frame_result.get("detections", []))

                frame_output = {
                    "timestamp_ms":     timestamp_ms,
                    "frame_number":     frame_count,
                    "detections":       frame_result.get("detections", []),
                    "traffic":          traffic_analysis,
                    "processing_time_ms": frame_time,
                    "detection_timing": frame_result.get("timing_ms", {}),
                    "has_error":        "error" in frame_result,
                }

                self._store_intermediate_result(job_id, frame_output, frames_processed)
                frames_processed += 1
                prev_frame = frame  # update reference for next scene change check

                if time.time() - last_progress_log > 5:
                    processed = min(frames_processed, settings.max_frames_per_video)
                    total     = min(total_frames, settings.max_frames_per_video)
                    progress  = (processed / total) * 100 if total > 0 else 0

                    logger.info(
                        f"[Job {job_id}] Progress: {progress:.1f}% "
                        f"({frames_processed}/{total} frames, "
                        f"{failed_frames} failed, {skipped_frames} skipped by scene change)"
                    )
                    self._update_job_status(job_id, "processing", progress=progress)

                    if progress_callback:
                        try:
                            progress_callback(
                                frames_processed,
                                min(total_frames, settings.max_frames_per_video),
                            )
                        except Exception as cb_err:
                            logger.debug(f"Progress callback error (non-critical): {cb_err}")

                    last_progress_log = time.time()

            # ── Timing summary ────────────────────────────────────────────────
            if frame_times:
                sorted_times = sorted(frame_times)
                p50_idx = len(sorted_times) // 2
                p95_idx = int(len(sorted_times) * 0.95)
                results["timing_ms"] = {
                    "avg_frame_ms":         sum(frame_times) / len(frame_times),
                    "min_frame_ms":         min(frame_times),
                    "max_frame_ms":         max(frame_times),
                    "p50_ms":               sorted_times[p50_idx],
                    "p95_ms":               sorted_times[p95_idx] if p95_idx < len(sorted_times) else sorted_times[-1],
                    "total_processing_ms":  sum(frame_times),
                    "frames_processed":     frames_processed,
                }

            if results["status"] != "cancelled":
                results["status"] = "completed"

            results["frames_processed"] = frames_processed
            results["failed_frames"]    = failed_frames

            if frames_processed == 0 and results["status"] == "completed":
                logger.error(
                    f"[Job {job_id}] Completed but yielded 0 frames — "
                    "file is likely corrupted or uses an unsupported codec."
                )
                results["status"] = "failed"
                results["error"]  = (
                    "Video processed but yielded 0 frames. "
                    "The file may be corrupted, truncated, or use an unsupported codec."
                )

            self._update_job_status(
                job_id,
                results["status"],
                frames_processed=frames_processed,
                failed_frames=failed_frames,
                completed_at=time.time(),
                **({"error": results["error"]} if results["status"] == "failed" else {}),
            )

            logger.info(
                f"[Job {job_id}] {results['status']}: "
                f"{frames_processed} frames, {failed_frames} failed"
            )

        except Exception as e:
            logger.error(f"[Job {job_id}] Failed: {str(e)}", exc_info=True)
            results["status"] = "failed"
            results["error"]  = str(e)
            self._update_job_status(job_id, "failed", error=str(e))

        finally:
            if cap:
                cap.release()
            with self.jobs_lock:
                self.active_jobs.discard(job_id)

        return results

    # ── Query helpers  ─────────────────────────────────────────────

    def get_job_frames(self, job_id: str, start: int = 0, limit: int = 100) -> Dict:
        def redis_operation(client):
            start_time = time.time()
            meta = client.hgetall(f"job:{job_id}:meta")
            if not meta:
                return {"frames": [], "total": 0, "job_exists": False}

            total_frames = int(meta.get("frames_processed", 0))
            status = meta.get("status", "unknown")
            end = min(start + limit, total_frames)

            batch_start = time.time()
            pipe = client.pipeline(transaction=False)
            for i in range(start, end):
                pipe.get(f"job:{job_id}:frame:{i}")
            raw_frames = pipe.execute()
            batch_time = (time.time() - batch_start) * 1000

            frames = []
            for raw in raw_frames:
                if raw:
                    try:
                        frames.append(json.loads(raw))
                    except Exception as e:
                        logger.warning(f"Failed to deserialize frame for job {job_id}: {e}")

            if batch_time > 100:
                logger.warning(
                    f"Slow Redis batch frame fetch for job {job_id}: "
                    f"{batch_time:.1f}ms for {end - start} frames"
                )

            errors = client.hlen(f"job:{job_id}:errors")
            total_time = (time.time() - start_time) * 1000
            if total_time > 100:
                logger.warning(f"Slow get_job_frames for job {job_id}: {total_time:.2f}ms")

            return {
                "frames":        frames,
                "total":         total_frames,
                "start":         start,
                "end":           end,
                "has_more":      end < total_frames,
                "job_status":    status,
                "failed_frames": errors,
                "job_exists":    True,
            }

        return self._safe_redis_operation(
            redis_operation,
            fallback={"frames": [], "total": 0, "error": "Redis unavailable"},
        )

    def cancel_job(self, job_id: str) -> bool:
        """Sets the cancellation flag in Redis for a running or pending job."""
        def redis_operation(client):
            status = client.hget(f"job:{job_id}:meta", "status")
            if status in ("processing", "pending"):
                client.hset(f"job:{job_id}:meta", "cancelled", "true")
                client.hset(f"job:{job_id}:meta", "status", "cancelling")
                logger.info(f"Job {job_id} cancellation requested")
                return True
            logger.warning(f"Job {job_id} cannot be cancelled (status: {status})")
            return False
        return self._safe_redis_operation(redis_operation, fallback=False)

    def get_job_status(self, job_id: str) -> Dict:
        def redis_operation(client):
            meta = client.hgetall(f"job:{job_id}:meta")
            if not meta:
                return {"job_exists": False}
            result = {"job_exists": True}
            for key, value in meta.items():
                if key in ("total_frames", "frames_processed", "failed_frames"):
                    try:
                        result[key] = int(value)
                    except Exception:
                        result[key] = value
                elif key in ("progress", "fps"):
                    try:
                        result[key] = float(value)
                    except Exception:
                        result[key] = value
                else:
                    result[key] = value
            return result
        return self._safe_redis_operation(
            redis_operation,
            fallback={"job_exists": False, "error": "Redis unavailable"},
        )
    

def get_queue_depth() -> int:
    """
    Returns number of jobs actively consuming worker resources.

    Intentionally excludes 'uploaded' — that status means the file is saved
    but no Celery task has been dispatched yet. No worker resources are consumed
    and the user may never start the job. Including it would cause queue depth
    to grow unboundedly from abandoned uploads.

    State machine:
        uploaded   → file saved, not counted (no worker resources)
        pending    → task dispatched, counted
        processing → worker active, counted
        cancelling → cancel requested, still running, counted
        completed/failed/cancelled → terminal, not counted

    Performance note: O(n) scan over job keys. Acceptable up to ~10,000 concurrent
    jobs (~5ms). Beyond that, a dedicated sorted set should be considered as a job registry.
    Called only at job submission time and metrics scraping — not in hot paths.
    """
    ACTIVE_STATUSES = {"pending", "processing", "cancelling"}
    try:
        client = get_redis_client()
        active = 0
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor, match="job:*:meta", count=100)
            if keys:
                pipe = client.pipeline(transaction=False)
                for key in keys:
                    pipe.hget(key, "status")
                statuses = pipe.execute()
                active += sum(1 for s in statuses if s in ACTIVE_STATUSES)
            if cursor == 0:
                break
        return active
    except Exception as e:
        logger.error(f"Queue depth check failed — failing closed: {e}")
        return settings.max_queue_size  # deny admission when Redis is unreachable