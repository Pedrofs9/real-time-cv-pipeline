"""
Prometheus metrics definitions.
All metrics used throughout the application (not shown real-time on the UI) are centralized here.
"""
from prometheus_client import Counter, Histogram, Gauge, CONTENT_TYPE_LATEST
import logging

logger = logging.getLogger(__name__)

# ── Mode constants — use these everywhere, never raw strings ──────────────────
MODE_IMAGE    = "image"
MODE_WATCH    = "watch"
MODE_DOWNLOAD = "download"
VALID_MODES   = (MODE_IMAGE, MODE_WATCH, MODE_DOWNLOAD)

# ── Inference metrics ─────────────────────────────────────────────────────────
INFERENCE_LATENCY = Histogram(
    "cv_inference_latency_seconds",
    "End-to-end inference latency per frame (decode + preprocess + infer + postprocess)",
    ["mode"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0],
)

INFERENCE_DECODE_LATENCY = Histogram(
    "cv_decode_latency_seconds",
    "Frame decode + preprocess time",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
)

INFERENCE_MODEL_LATENCY = Histogram(
    "cv_model_latency_seconds",
    "Raw model forward-pass time",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5],
)

INFERENCE_POSTPROCESS_LATENCY = Histogram(
    "cv_postprocess_latency_seconds",
    "NMS + result serialisation time",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05],
)

INFERENCE_SERIALIZE_LATENCY = Histogram(
    "cv_serialize_latency_seconds",
    "Result serialisation time (dict construction + JSON prep)",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01],
)

FRAMES_PROCESSED = Counter(
    "cv_frames_processed_total",
    "Total frames processed successfully",
    ["mode"],
)

FRAMES_FAILED = Counter(
    "cv_frames_failed_total",
    "Total frames that failed processing",
    ["mode", "reason"],
)

DETECTIONS_PER_FRAME = Histogram(
    "cv_detections_per_frame",
    "Number of detections returned per frame",
    buckets=[0, 1, 2, 5, 10, 15, 20, 30, 50],
)

# ── HTTP metrics ──────────────────────────────────────────────────────────────
HTTP_REQUEST_LATENCY = Histogram(
    "cv_http_request_latency_seconds",
    "HTTP request latency by endpoint",
    ["method", "endpoint", "status"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

HTTP_REQUESTS_TOTAL = Counter(
    "cv_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

# ── Video job metrics ─────────────────────────────────────────────────────────
VIDEO_JOBS_SUBMITTED = Counter(
    "cv_video_jobs_submitted_total",
    "Total video jobs submitted (upload only, before mode is chosen)",
)

VIDEO_JOBS_STARTED = Counter(
    "cv_video_jobs_started_total",
    "Total video jobs started, by mode",
    ["mode"],
)

VIDEO_JOBS_COMPLETED = Counter(
    "cv_video_jobs_completed_total",
    "Total video jobs completed",
    ["mode", "status"],
)

VIDEO_PROCESSING_DURATION = Histogram(
    "cv_video_processing_duration_seconds",
    "Total time to process a video job end-to-end",
    ["mode"],
    buckets=[5, 10, 30, 60, 120, 300, 600],
)

VIDEO_FRAMES_PER_JOB = Histogram(
    "cv_video_frames_per_job",
    "Number of frames processed per video job",
    ["mode"],
    buckets=[100, 300, 500, 1000, 1500, 2000, 3000, 5000, 10000],
)

# ── System metrics ────────────────────────────────────────────────────────────
MODEL_LOADED = Gauge(
    "cv_model_loaded",
    "1 if the detection model is loaded in the API process, 0 otherwise",
)

GPU_MEMORY_USED_MB = Gauge(
    "cv_gpu_memory_used_mb",
    "GPU memory allocated by the API process (MB). 0 on CPU.",
)

UPLOAD_SIZE_BYTES = Histogram(
    "cv_upload_size_bytes",
    "Size of uploaded files in bytes",
    ["type"],
    buckets=[
        1_000_000, 5_000_000, 10_000_000, 25_000_000,
        50_000_000, 100_000_000, 200_000_000, 500_000_000,
    ],
)

# Add near other system metrics (around line 100-120)
CPU_USAGE_PERCENT = Gauge(
    "cv_cpu_usage_percent",
    "CPU usage percentage of the API process",
)

MEMORY_USAGE_MB = Gauge(
    "cv_memory_usage_mb",
    "Memory usage of the API process in MB",
)

QUEUE_DEPTH = Gauge(
    "cv_queue_depth",
    "Number of video jobs currently pending in the Celery queue",
)

# ── WebSocket metrics ─────────────────────────────────────────────────────────
WS_CONNECTIONS_ACTIVE = Gauge(
    "cv_ws_connections_active",
    "Number of active WebSocket connections",
)

WS_FRAMES_SENT = Counter(
    "cv_ws_frames_sent_total",
    "Total detection frames sent over WebSocket",
)