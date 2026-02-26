"""
FastAPI application server for real-time CV pipeline.
Handles image/video uploads, job management, WebSocket streaming,
and Prometheus metrics exposure. Uses Celery for background video processing.
"""

from fastapi import WebSocket, WebSocketDisconnect
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager 
import logging
import time
import uuid
import json
import asyncio
import os
from pathlib import Path
import re
import psutil
from datetime import datetime
from typing import Dict, Optional, List, Any, AsyncGenerator
from core.config import settings
from core.file_validation import validate_image_bytes, validate_video_header
from pipeline.detection import DetectionPipeline
from workers.video_tasks import process_video_task, celery_app, generate_download_video_task, process_and_render_download_task
from pipeline.video import get_redis_client, VideoProcessor, get_queue_depth
from pipeline.traffic import analyze_traffic
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from core.metrics import (
    HTTP_REQUEST_LATENCY, HTTP_REQUESTS_TOTAL,
    VIDEO_JOBS_SUBMITTED, VIDEO_JOBS_STARTED, UPLOAD_SIZE_BYTES,
    MODEL_LOADED, GPU_MEMORY_USED_MB,
    WS_CONNECTIONS_ACTIVE, WS_FRAMES_SENT,
    MODE_WATCH, MODE_DOWNLOAD, QUEUE_DEPTH, CPU_USAGE_PERCENT, MEMORY_USAGE_MB
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============= LIFECYCLE =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────
    global upload_dir, _api_process
    logger.info("Starting up FastAPI application...")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Upload directory: {upload_dir}")

    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")

    logger.info("✅ Pipeline will load on first request (lazy loading)")

    try:
        redis_client = get_redis_client()
        redis_client.ping()
        logger.info("✅ Redis connection successful")
    except Exception as e:
        logger.warning(f"⚠️ Redis not available at startup: {e}")

    _api_process = psutil.Process()
    _api_process.cpu_percent()  # prime — first call always returns 0.0
    logger.info("✅ Process metrics initialized")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("🛑 Shutting down FastAPI application...")
    now = time.time()

    for directory, label in [
        (Path(settings.upload_dir), "upload"),
        (Path(settings.output_dir), "output"),
    ]:
        if directory.exists():
            for f in directory.glob("*.mp4"):
                if now - f.stat().st_mtime > settings.results_ttl_seconds:
                    try:
                        f.unlink()
                        logger.debug(f"Cleaned up expired {label}: {f}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {f}: {e}")



app = FastAPI(
    title="Real-Time CV Pipeline",
    description="Computer vision for traffic monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

ui_path = Path(__file__).parent.parent / "ui"
if ui_path.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_path), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(ui_path / "index.html")
else:
    logger.warning(f"UI directory not found at {ui_path}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= GLOBALS =============
pipeline: DetectionPipeline = None
upload_dir: Path = None
_pipeline_lock = asyncio.Lock()
_api_process: psutil.Process = None



_video_processor: VideoProcessor = None

def get_video_processor_for_queries() -> VideoProcessor:
    """
    Returns a VideoProcessor used only for Redis queries.
    Does NOT load the ML model — detection_pipeline=None is intentional.
    Only get_job_frames() and get_job_status() are safe to call on this instance.
    """
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor(detection_pipeline=None)
    return _video_processor

# Bounds concurrent image inference — prevents GPU memory exhaustion
# under parallel requests. Value should match your GPU capacity.
_inference_semaphore = asyncio.Semaphore(settings.max_concurrent_image_requests)

# ============= BACKPRESSURE HELPERS =============
# Defined before any endpoint that calls them.
def _check_queue_capacity() -> None:
    """
    Reject new video jobs with 503 when the queue is at capacity.
    This is the primary backpressure mechanism for the async video pipeline.
    """
    depth = get_queue_depth()
    if depth >= settings.max_queue_size:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "queue_full",
                "message": "Server is at capacity. Please retry later.",
                "queue_depth": depth,
                "max_queue_size": settings.max_queue_size,
            }
        )

@asynccontextmanager
async def _acquire_with_timeout(semaphore: asyncio.Semaphore, timeout: float):
    """
    Acquire a semaphore with a timeout.
    Implements backpressure for concurrent image inference:
    callers wait up to `timeout` seconds for a free slot, then receive a 503.
    """
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
        try:
            yield
        finally:
            semaphore.release()
    except asyncio.TimeoutError:
        raise TimeoutError(f"Could not acquire inference slot within {timeout}s")


# ============= PIPELINE DEPENDENCY =============

async def get_pipeline() -> DetectionPipeline:
    """
    Lazy-load the detection pipeline with a double-checked async lock.
    Prevents concurrent initialization under parallel requests.
    """
    global pipeline
    if pipeline is not None:
        return pipeline  # Fast path — no lock needed once initialized
    async with _pipeline_lock:
        if pipeline is None:
            logger.info("🔄 Loading detection pipeline (first request)...")
            start = time.perf_counter()
            pipeline = DetectionPipeline()
            load_time = time.perf_counter() - start
            logger.info(f"✅ Pipeline loaded in {load_time:.2f}s")
    return pipeline


# ============= HEALTH =============

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Simple health check for orchestration.
    """
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """
    Readiness probe - checks if model is loaded.
    Raises:
        HTTPException: 503 if model not loaded
    """
    pipe = await get_pipeline()
    if pipe.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/live")
async def liveness_check() -> Dict[str, str]:
    """
    Liveness probe - always returns ok if process is running.
    """
    return {"status": "alive"}


# ============= METRICS =============

@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus scrape endpoint — updates process-level gauges on each scrape."""

    try:
        import torch
        GPU_MEMORY_USED_MB.set(
            torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        )
    except Exception:
        GPU_MEMORY_USED_MB.set(0)

    MODEL_LOADED.set(1 if pipeline is not None else 0)

    try:
        if _api_process is not None:
            CPU_USAGE_PERCENT.set(_api_process.cpu_percent())
            MEMORY_USAGE_MB.set(_api_process.memory_info().rss / (1024 * 1024))
    except Exception as e:
        logger.debug(f"Failed to get CPU/memory metrics: {e}")

    QUEUE_DEPTH.set(get_queue_depth())

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============= MIDDLEWARE =============

@app.middleware("http")
async def track_requests(request: Request, call_next) -> Response:
    """
    Middleware to track HTTP request metrics and log slow requests.
    """
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time

    path = request.url.path
    path = re.sub(
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        '{job_id}', path
    )

    if path != "/metrics":
        HTTP_REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=path,
            status=str(response.status_code)
        ).observe(duration)
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=path,
            status=str(response.status_code)
        ).inc()

    if duration > 1.0:
        logger.warning(
            f"Slow request: {request.method} {request.url.path} took {duration * 1000:.2f}ms"
        )

    logger.info(
        f"Request completed: {request.method} {request.url.path} "
        f"status={response.status_code} duration_ms={round(duration * 1000, 2)}"
    )
    response.headers["X-Process-Time-MS"] = str(round(duration * 1000, 2))
    return response


# ============= IMAGE ENDPOINTS =============

@app.post("/v1/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    request: Request = None
) -> Dict[str, Any]:
    """
    Detect objects in an uploaded image. Returns results synchronously.
    Inference is guarded by a semaphore — concurrent requests queue here
    rather than competing for GPU memory.
    """
    pipe = await get_pipeline()
    contents = await file.read()
    UPLOAD_SIZE_BYTES.labels(type="image").observe(len(contents))

    if len(contents) > settings.max_image_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_image_size_mb}MB"
        )

    # Validate using magic bytes — not the client-supplied Content-Type header
    detected_type = validate_image_bytes(contents)
    logger.debug(f"Image validated as {detected_type}: {file.filename}")

    try:
        async with _acquire_with_timeout(_inference_semaphore, settings.request_timeout_seconds):
            result = await pipe.process_image(contents)
    except TimeoutError:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "inference_busy",
                "message": "All inference slots are busy. Please retry.",
            }
        )
    except ValueError as e:
        # Raised by detection pipeline for decodable but invalid images
        raise HTTPException(
            status_code=400,
            detail={
                "error": "image_decode_failed",
                "message": str(e),
            }
        )

    ext_request_id = request.headers.get("x-request-id", "") if request else ""
    if ext_request_id:
        result["request_id"] = f"{result.get('request_id', '')}_{ext_request_id}"

    if result.get("error"):
        logger.error(f"Pipeline error: {result.get('message', 'Unknown error')}")
        return JSONResponse(status_code=500, content=result)

    result["traffic"] = analyze_traffic(result.get("detections", []))

    return JSONResponse(
        content=result,
        headers={"X-Total-Time-MS": str(round(result["timing_ms"]["total_ms"], 2))}
    )


# ============= VIDEO ENDPOINTS =============

@app.post("/v1/detect/video", status_code=202)
async def detect_video(
    file: UploadFile = File(..., description="Video file to process"),
    request: Request = None
) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    upload_path = Path(settings.upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    video_path = upload_path / f"{job_id}.mp4"

    try:
        # Step 1: Read header bytes first — validate format before touching disk
        header = await file.read(16)
        if not header:
            raise HTTPException(status_code=400, detail="Empty file")

        detected_type = validate_video_header(header, file.filename)
        logger.debug(f"Video validated as {detected_type}: {file.filename}")

        # Step 2: Stream write with live size enforcement
        max_bytes = settings.max_video_size_mb * 1024 * 1024
        bytes_written = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(video_path, "wb") as f:
            # Write the header bytes we already read
            f.write(header)
            bytes_written += len(header)

            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break

                bytes_written += len(chunk)

                # Abort the moment we exceed the limit — no point writing more
                if bytes_written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "error": "file_too_large",
                            "message": f"Video exceeds the maximum allowed size of {settings.max_video_size_mb}MB.",
                            "max_size_mb": settings.max_video_size_mb,
                        }
                    )

                f.write(chunk)

        file_size = bytes_written
        logger.info(f"Saved upload: {video_path} ({file_size / 1024 / 1024:.1f}MB)")

        # Step 3: Store metadata and return job_id
        try:
            redis_client = get_redis_client()
            redis_client.hset(f"job:{job_id}:meta", mapping={
                "status": "uploaded",
                "filename": file.filename,
                "file_size": str(file_size),
                "detected_type": detected_type,
                "uploaded_at": str(time.time()),
                "video_path": str(video_path),
            })
            redis_client.expire(f"job:{job_id}:meta", settings.results_ttl_seconds)
        except Exception as e:
            logger.warning(f"Redis metadata store failed (non-critical): {e}")

        UPLOAD_SIZE_BYTES.labels(type="video").observe(file_size)
        VIDEO_JOBS_SUBMITTED.inc()

        return {
            "job_id": job_id,
            "status": "uploaded",
            "filename": file.filename,
            "detected_type": detected_type,
            "file_size_mb": round(file_size / 1024 / 1024, 1),
            "message": "Video saved. Choose watch or download to begin processing.",
        }

    except HTTPException:
        if video_path.exists():
            video_path.unlink()
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        if video_path.exists():
            video_path.unlink()
        raise HTTPException(status_code=500, detail="Upload failed")


@app.post("/v1/jobs/{job_id}/start_watch", status_code=202)
async def start_watch_processing(job_id: str) -> Dict[str, Any]:
    """
    Start video processing in watch mode (sample_rate=3).
    Results are streamed via WebSocket for real-time overlay.
    Raises:
        HTTPException: 404 if job not found, 503 if queue full
    """
    _check_queue_capacity()
    try:
        redis_client = get_redis_client()
        meta = redis_client.hgetall(f"job:{job_id}:meta")

        if not meta:
            raise HTTPException(status_code=404, detail="Job not found")

        status = meta.get("status")
        if status not in ("uploaded", "completed"):
            return {
                "job_id": job_id,
                "status": status,
                "message": f"Job already in state: {status}"
            }

        video_path = meta.get("video_path")
        if not video_path or not Path(video_path).exists():
            raise HTTPException(status_code=404, detail="Video file not found")

        filename = meta.get("filename", "unknown")
        watch_sample_rate = settings.watch_frame_sample_rate

        redis_client.hset(f"job:{job_id}:meta", mapping={
            "status": "pending",
            "cancelled": "false",
            "mode": "watch",
            "frame_sample_rate": str(watch_sample_rate),
        })
        redis_client.expire(f"job:{job_id}:meta", settings.results_ttl_seconds)

        process_video_task.apply_async(
            args=[video_path, filename, job_id, watch_sample_rate],
            queue="video_processing",
            task_id=job_id,
        )

        VIDEO_JOBS_STARTED.labels(mode=MODE_WATCH).inc()
        logger.info(f"Watch processing started for job {job_id} (sample_rate={watch_sample_rate})")

        return {
            "job_id": job_id,
            "status": "pending",
            "mode": "watch",
            "frame_sample_rate": watch_sample_rate,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"start_watch failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start watch processing")


@app.post("/v1/jobs/{job_id}/start_download", status_code=202)
async def start_download_processing(job_id: str) -> Dict[str, Any]:
    """
    Start video processing in download mode (sample_rate=1).
    Processes all frames and generates annotated video file.
    Raises:
        HTTPException: 404 if job not found, 409 if job in progress
    """
    _check_queue_capacity()
    try:
        redis_client = get_redis_client()
        meta = redis_client.hgetall(f"job:{job_id}:meta")

        if not meta:
            raise HTTPException(status_code=404, detail="Job not found")

        status = meta.get("status")

        # Block if job is actively being processed — two tasks on same job corrupts results
        if status in ("pending", "processing"):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "job_in_progress",
                    "message": "This job is currently processing in watch mode. "
                            "Wait for it to complete before requesting a download.",
                    "current_status": status,
                    "hint": "Poll GET /v1/jobs/{job_id} until status is 'completed', then retry."
                }
            )

        if status not in ("uploaded",):
            if status == "completed":
                existing_download = meta.get("download_status")
                if existing_download == "ready":
                    return {"job_id": job_id, "status": "ready", "message": "Already rendered"}
                redis_client.hset(f"job:{job_id}:meta", "download_status", "queued")
                generate_download_video_task.delay(job_id)
                VIDEO_JOBS_STARTED.labels(mode=MODE_DOWNLOAD).inc()
                return {
                    "job_id": job_id,
                    "status": "queued",
                    "message": "Render queued from cached detections"
                }
            return {
                "job_id": job_id,
                "status": status,
                "message": f"Job already in state: {status}"
            }

        video_path = meta.get("video_path")
        if not video_path or not Path(video_path).exists():
            raise HTTPException(status_code=404, detail="Video file not found")

        filename = meta.get("filename", "unknown")
        download_sample_rate = settings.download_frame_sample_rate

        redis_client.hset(f"job:{job_id}:meta", mapping={
            "status": "pending",
            "cancelled": "false",
            "mode": "download",
            "frame_sample_rate": str(download_sample_rate),
        })
        redis_client.expire(f"job:{job_id}:meta", settings.results_ttl_seconds)

        process_and_render_download_task.apply_async(
            args=[video_path, filename, job_id, download_sample_rate],
            queue="video_processing",
            task_id=job_id,
        )

        VIDEO_JOBS_STARTED.labels(mode=MODE_DOWNLOAD).inc()
        logger.info(
            f"Download processing started for job {job_id} (sample_rate={download_sample_rate})"
        )

        return {
            "job_id": job_id,
            "status": "pending",
            "mode": "download",
            "frame_sample_rate": download_sample_rate,
            "message": "Processing started. Poll /v1/jobs/{job_id}/download_status for progress.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"start_download failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start download processing")


@app.get("/v1/jobs")
async def list_jobs(limit: int = 10) -> Dict[str, Any]:
    """
    List recent jobs. Uses SCAN (non-blocking) — safe for production Redis.
    Not intended for high-frequency polling.
    """
    limit = min(limit, 50)
    try:
        redis_client = get_redis_client()
        job_ids = []
        cursor = 0
        while True:
            cursor, keys = redis_client.scan(cursor, match="job:*:meta", count=100)
            for key in keys:
                parts = key.split(":")
                if len(parts) == 3:
                    job_ids.append(parts[1])
            if cursor == 0 or len(job_ids) >= limit:
                break

        jobs = []
        for job_id in job_ids[:limit]:
            meta = redis_client.hgetall(f"job:{job_id}:meta")
            if meta:
                jobs.append({
                    "job_id": job_id,
                    "status": meta.get("status", "unknown"),
                    "filename": meta.get("filename", "unknown"),
                    "progress": float(meta.get("progress", 0)),
                    "frames_processed": int(meta.get("frames_processed", 0)),
                    "mode": meta.get("mode", "unknown"),
                })

        return {"jobs": jobs, "count": len(jobs)}

    except Exception as e:
        logger.warning(f"Failed to list jobs: {e}")
        return {"jobs": [], "count": 0}


@app.get("/v1/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    include_frames: bool = False,
    start: int = 0,
    limit: int = 10
) -> Dict[str, Any]:
    limit = min(limit, 100)
    try:
        redis_client = get_redis_client()
        meta = redis_client.hgetall(f"job:{job_id}:meta")
    except Exception as e:
        logger.warning(f"Redis unavailable for job {job_id}: {e}")
        raise HTTPException(status_code=503, detail="Storage unavailable, please retry")

    if not meta:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job_result = redis_client.get(f"job:{job_id}")
    if job_result:
        result = json.loads(job_result)
        if include_frames and result.get("frames_processed", 0) > 0:
            vp = get_video_processor_for_queries()
            frames = vp.get_job_frames(job_id, start=start, limit=limit)
            result["frames"] = frames.get("frames", [])
            result["frames_pagination"] = {
                "start": start,
                "limit": limit,
                "total": frames.get("total", 0),
                "has_more": frames.get("has_more", False),
            }
        return result

    return {
        "job_id": job_id,
        "status": meta.get("status", "processing"),
        "filename": meta.get("filename"),
        "progress": float(meta.get("progress", 0)),
        "frames_processed": int(meta.get("frames_processed", 0)),
        "total_frames": int(meta.get("total_frames", 0)),
        "submitted_at": float(meta.get("submitted_at", 0)),
        "queue": meta.get("queue", "unknown"),
    }


@app.delete("/v1/jobs/{job_id}")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    """
    Cancel a running video processing job.    
    Raises:
        HTTPException: 404 if job not found
    """
    try:
        redis_client = get_redis_client()
        meta = redis_client.hgetall(f"job:{job_id}:meta")

        if not meta:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        current_status = meta.get("status", "")

        # Don't cancel jobs that are already in a terminal state
        if current_status in ("completed", "failed", "cancelled"):
            return {
                "job_id": job_id,
                "status": current_status,
                "message": f"Job is already in terminal state '{current_status}' — nothing to cancel.",
                "cancelled": False,
            }

        if current_status == "pending":
            final_status = "cancelled"   # never picked up — terminal immediately
        else:
            final_status = "cancelling"  # worker is running — it will self-terminate

        redis_client.hset(f"job:{job_id}:meta", mapping={
            "cancelled": "true",
            "status": final_status,
        })

        celery_app.control.revoke(job_id, terminate=True)
        logger.info(f"Job {job_id} cancelled via Redis + Celery revoke")

        video_path = Path(settings.upload_dir) / f"{job_id}.mp4"
        if video_path.exists():
            try:
                video_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete video file for cancelled job {job_id}: {e}")

        return {
            "job_id": job_id,
            "status": final_status,
            "message": "Job cancellation requested. The job will stop within 1 frame.",
            "cancelled": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Cancel failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel job")
    

# ============= VIDEO SERVING =============

@app.get("/v1/video/{job_id}")
async def get_video_file(job_id: str) -> FileResponse:
    """
    Serve the original uploaded video file.
    Used by the frontend for video playback.
    Raises:
        HTTPException: 404 if video not found
    """
    video_path = Path(settings.upload_dir) / f"{job_id}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/v1/jobs/{job_id}/download_status")
async def get_download_status(job_id: str) -> Dict[str, Any]:
    try:
        redis_client = get_redis_client()
        meta = redis_client.hgetall(f"job:{job_id}:meta")
        if not meta:
            raise HTTPException(status_code=404, detail="Job not found")

        download_status = meta.get("download_status", "not_started")
        progress = float(meta.get("download_progress", 0))

        return {
            "job_id": job_id,
            "download_status": download_status,
            "progress": progress,
            "ready": download_status == "ready",
            "error": meta.get("download_error") if download_status == "failed" else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download status check failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")


@app.get("/v1/jobs/{job_id}/download")
async def download_annotated_video(job_id: str) -> FileResponse:
    try:
        redis_client = get_redis_client()
        meta = redis_client.hgetall(f"job:{job_id}:meta")
        if not meta:
            raise HTTPException(status_code=404, detail="Job not found")

        download_status = meta.get("download_status", "")
        if download_status != "ready":
            raise HTTPException(
                status_code=425,
                detail={
                    "message": "Annotated video not ready yet",
                    "download_status": download_status,
                    "progress": float(meta.get("download_progress", 0)),
                }
            )

        video_path = Path(meta.get("download_video_path", ""))
        if not video_path.exists():
            redis_client.hset(f"job:{job_id}:meta", mapping={
                "download_status": "failed",
                "download_error": "File expired or deleted"
            })
            raise HTTPException(
                status_code=404,
                detail="Download file not found — it may have expired"
            )

        original_filename = meta.get("filename", "video")
        base_name = os.path.splitext(original_filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_annotated_{timestamp}.mp4"
        logger.info(f"Serving download for job {job_id}: {video_path}")

        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            filename=filename,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download serve failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Error serving download")


# ============= WEBSOCKET =============

@app.websocket("/ws/video/{job_id}")
async def websocket_detection_stream(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    WS_CONNECTIONS_ACTIVE.inc()
    logger.info(f"WebSocket connected for job {job_id}")

    # Fail fast — don't make client wait for pending timeout on a bad job_id
    try:
        redis_client = get_redis_client()
        meta = redis_client.hgetall(f"job:{job_id}:meta")
        if not meta:
            logger.warning(f"WebSocket: job {job_id} not found, closing immediately")
            await websocket.send_json({
                "error": "job_not_found",
                "message": f"Job {job_id} does not exist or has expired. Please upload again."
            })
            await websocket.close()
            WS_CONNECTIONS_ACTIVE.dec()
            return
    except Exception as e:
        # Redis unavailable — continue and let the loop handle it gracefully
        logger.warning(f"WebSocket: could not verify job {job_id} existence: {e}")

    try:
        vp = get_video_processor_for_queries()
        last_sent_frame = -1
        consecutive_empty_polls = 0
        max_empty_polls_pending = 25
        max_empty_polls_processing = 150
        frame_send_times = []
        last_log_time = time.time()
        last_activity = time.time()


        while True:
            if time.time() - last_activity > 60:  # 60 second idle timeout
                logger.warning(f"WebSocket {job_id} idle timeout, closing")
                break

            # Update last_activity whenever a frame is successfully sent
            loop_start = time.time()
            status = vp.get_job_status(job_id)
            status_time = (time.time() - loop_start) * 1000
            current_status = status.get("status", "unknown")
            total_frames = status.get("frames_processed", 0)

            if total_frames > last_sent_frame + 1:
                new_frames_start = last_sent_frame + 1
                frames_to_get = min(10, total_frames - new_frames_start)

                redis_start = time.time()
                page = vp.get_job_frames(job_id, start=new_frames_start, limit=frames_to_get)
                redis_time = (time.time() - redis_start) * 1000

                send_start = time.time()
                for frame in page.get("frames", []):
                    frame["job_status"] = current_status
                    await websocket.send_json(frame)
                    last_sent_frame += 1
                    last_activity = time.time()
                    WS_FRAMES_SENT.inc()
                send_time = (time.time() - send_start) * 1000

                frame_send_times.append({
                    "frames": frames_to_get,
                    "status_time": status_time,
                    "redis_time": redis_time,
                    "send_time": send_time,
                    "total": (time.time() - loop_start) * 1000
                })
                consecutive_empty_polls = 0
            else:
                consecutive_empty_polls += 1

            if time.time() - last_log_time > 10 and frame_send_times:
                avg_total = sum(t["total"] for t in frame_send_times) / len(frame_send_times)
                avg_redis = sum(t["redis_time"] for t in frame_send_times) / len(frame_send_times)
                avg_send = sum(t["send_time"] for t in frame_send_times) / len(frame_send_times)
                logger.info(
                    f"WebSocket perf for job {job_id}: "
                    f"avg_loop={avg_total:.1f}ms redis={avg_redis:.1f}ms "
                    f"send={avg_send:.1f}ms frames_sent={last_sent_frame + 1}"
                )
                frame_send_times = []
                last_log_time = time.time()

            if current_status in ["completed", "failed", "cancelled"]:
                if last_sent_frame >= total_frames - 1:
                    logger.info(f"Job {job_id} complete, sent all {last_sent_frame + 1} frames")
                    await websocket.send_json({"status": "complete", "job_status": current_status})
                    break
            elif current_status == "pending":
                if consecutive_empty_polls > max_empty_polls_pending:
                    logger.warning(f"Job {job_id} never started (pending timeout), closing")
                    await websocket.send_json({
                        "error": "job_timeout",
                        "message": "Job never started processing. It may have been rejected or lost."
                    })
                    break
            elif current_status == "processing":
                if consecutive_empty_polls > max_empty_polls_processing:
                    logger.warning(f"Job {job_id} stalled during processing, closing")
                    await websocket.send_json({
                        "error": "job_stalled",
                        "message": "Job stopped producing frames. The worker may have crashed."
                    })
                    break

            await asyncio.sleep(0.2)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {job_id}: {e}")
        try:
            await websocket.send_json({
                "error": "stream_error",
                "message": "An unexpected error occurred in the detection stream."
            })
        except Exception:
            pass
    finally:
        WS_CONNECTIONS_ACTIVE.dec()
        try:
            await websocket.close()
        except Exception as close_err:
            logger.debug(f"WebSocket close error (already closed by client): {close_err}")