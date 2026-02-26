"""
Application configuration management using Pydantic Settings.
Loads from environment variables (defined in .env) and provides validation.
Single source of truth for all tunable parameters.
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field
from typing import Literal, Optional
import os
import logging

logger = logging.getLogger(__name__)

def _default_device() -> str:
    """Determine default inference device. Lazy-imports torch to avoid
    paying the import cost in processes that don't need GPU detection."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    SINGLE SOURCE OF TRUTH for all configuration.
    """
    
    # ============= Pydantic config =============
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=()
    )

    # ============= API SETTINGS =============
    api_host: str = Field("0.0.0.0", description="Host to bind the API server to")
    api_port: int = Field(8000, description="Port to bind the API server to", ge=1, le=65535)
    workers: int = Field(1, description="Number of worker processes", ge=1, le=8)
    reload: bool = Field(False, description="Auto-reload on code changes (dev only!)")
    
    # ============= MODEL SETTINGS =============
    model_path: str = Field("models/yolov8n.pt", description="Path to model weights")
    confidence_threshold: float = Field(0.5, description="Minimum confidence for detections (0-1)", ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, description="IoU threshold for NMS", ge=0.0, le=1.0)

    device: Literal["cpu", "cuda", "mps"] = Field(
        default_factory=_default_device,
        description="Device to run inference on"
    )

    model_backend: Literal["pytorch", "onnx", "tensorrt"] = Field("pytorch", description="Inference backend")
    use_fp16: bool = Field(
        False,
        description=(
            "Cast model weights to float16 before inference. "
            "Only effective when device=cuda — silently ignored on CPU/MPS. "
            "Expect 20-40% reduction in model_ms on Turing (RTX 20xx) or later. "
            "Set USE_FP16=true in .env to enable."
        )
    )

    
    # ============= PERFORMANCE SETTINGS =============
    model_input_size: int = Field(640, description="Resize images to this size (maintains aspect ratio)", ge=32, le=4096)
    batch_size: int = Field(1, description="Batch size for inference", ge=1, le=32)
    num_workers: int = Field(2, description="Number of dataloader workers", ge=0, le=8)
    
    # ============= ROBUSTNESS SETTINGS =============
    max_image_size_mb: int = Field(10, description="Maximum uploaded file size in MB", ge=1, le=100)
    request_timeout_seconds: int = Field(30, description="Request timeout in seconds", ge=1, le=300)
    max_queue_size: int = Field(100, description="Maximum queue size for background tasks", ge=1, le=10000)
    max_concurrent_image_requests: int = Field(
    4,
    description="Max simultaneous image inference requests. Set to match your GPU parallelism.",
    ge=1,
    le=32,
    )
    # ============= RESOURCE LIMITS =============
    gpu_memory_fraction: float = Field(0.8, description="Max fraction of GPU memory to use", ge=0.1, le=1.0)
    cpu_threads: int = Field(4, description="Number of CPU threads for OpenCV/etc", ge=1, le=32)
    
    # ============= LOGGING & MONITORING =============
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO", description="Logging level")
    enable_metrics: bool = Field(True, description="Enable Prometheus metrics")
    metrics_port: int = Field(8001, description="Port for metrics endpoint")
    
    # ============= STORAGE SETTINGS =============
    upload_dir: str = Field(
        "/app/data/uploads",
        description="Directory for raw uploaded video files"
    )
    output_dir: str = Field(
        "/app/data/outputs",
        description="Directory for processed annotated video outputs"
    )
    results_ttl_seconds: int = Field(
        3600,
        description="How long to keep results (1 hour)",
        ge=60,
        le=86400
    )
    delete_videos_after_processing: bool = Field(
        default=False,
        description=(
            "Delete source upload after download mode processing completes. "
            "Does NOT apply to watch mode — watch mode requires the source file "
            "to remain available for the /v1/video/{job_id} streaming endpoint. "
            "TODO: In a future version, implement a separate TTL-based cleanup for "
            "watch mode uploads once the WebSocket stream has fully closed."
        )
    )

    # ============= ALLOWED FILE TYPES =============
    allowed_image_types: list = Field(
        default=["image/jpeg", "image/png", "image/bmp", "image/webp"],
        description="Allowed image MIME types. Must be decodable by OpenCV."
    )
    allowed_video_types: list = Field(
        default=["video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska"],
        description="Allowed video MIME types. Must be decodable by OpenCV/ffmpeg."
    )

    # ============= REDIS SETTINGS =============
    redis_host: str = Field(
        default="localhost",
        description="Redis server hostname",
    )
    redis_port: int = Field(
        default=6379,
        description="Redis server port",
        ge=1,
        le=65535
    )
    redis_db: int = Field(
        default=0,
        description="Redis database number",
        ge=0,
        le=15
    )
    
    # ============= VIDEO PROCESSING SETTINGS =============
    video_timeout_seconds: int = Field(
        default=300,
        description="Max video processing time in seconds",
        ge=60,
        le=3600
    )
    max_video_size_mb: int = Field(
        default=100,
        description="Maximum video file size in MB",
        ge=1,
        le=1000
    )
    download_frame_sample_rate: int = Field(
        default=1,
        description="Process every Nth frame for DOWNLOAD mode (1 = all frames). Maps to DOWNLOAD_FRAME_SAMPLE_RATE env var.",
        ge=1, le=30
    )
    watch_frame_sample_rate: int = Field(
        default=3,
        description="Process every Nth frame for WATCH mode (higher = faster stream, fewer detections). Maps to WATCH_FRAME_SAMPLE_RATE env var.",
        ge=1, le=30
    )

    scene_change_threshold: float = Field(
        default=0.02,
        description=(
            "Minimum mean pixel change (as fraction of 255) between frames "
            "to trigger inference. 0.02 = 2% change. "
            "Lower = more sensitive (more frames processed). "
            "Higher = more aggressive skipping (faster, may miss slow-moving objects). "
            "0.0 disables content-aware sampling entirely (same as before). "
            "Set SCENE_CHANGE_THRESHOLD in .env to tune."
        ),
        ge=0.0,
        le=1.0,
    )

    max_frames_per_video: int = Field(
        default=300,
        description="Maximum number of frames to process per video",
        ge=1,
        le=10000
    )
    # ============= CELERY SETTINGS =============
    celery_max_retries: int = Field(
        default=3,
        description="Maximum number of Celery task retries",
        ge=0,
        le=10
    )
    celery_retry_delay: int = Field(
        default=60,
        description="Seconds between retries",
        ge=10,
        le=3600
    )
    celery_queue: str = Field(
        default="video_processing",
        description="Default Celery queue name"
    )

# Create global settings instance
settings = Settings()

# ============= VALIDATION FUNCTION =============
def validate_settings(s: Settings = None, strict: bool = True) -> list:
    """
    Validate settings - call this explicitly on app startup.
    
    Args:
        s: Settings to validate (uses global if None)
        strict: If True, raise on errors; if False, return warnings
    
    Returns:
        List of warnings (empty if no warnings)
    """
    if s is None:
        s = settings
    
    errors = []
    warnings = []

    # Model file exists check
    if s.model_backend == "pytorch":
        if not os.path.exists(s.model_path):
            warnings.append(
                f"Model file not found at config path: {s.model_path}. "
                "This may be expected if running in Docker where the path differs from the host. "
                "The worker will validate the path at startup."
            )
    
    # Upload directory writable check
    try:
        os.makedirs(s.upload_dir, exist_ok=True)
        test_file = os.path.join(s.upload_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        msg = f"Upload directory not writable: {e}"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)
    
    # Check CUDA availability if requested
    if s.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                msg = "CUDA device requested but not available"
                if strict:
                    errors.append(msg)
                else:
                    warnings.append(msg + " - will fall back to CPU")
        except ImportError:
            msg = "PyTorch not installed, cannot verify CUDA"
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
    
    if strict and errors:
        raise RuntimeError(
            "Configuration validation failed:\n" + 
            "\n".join(errors)
        )
    
    return warnings

# ============= HELPER FUNCTIONS =============
def get_settings_summary(s: Settings = None) -> dict:
    """Get a summary of settings for logging/debugging."""
    if s is None:
        s = settings
    
    return {
        "api": f"{s.api_host}:{s.api_port}",
        "model": {
            "path": s.model_path,
            "device": s.device,
            "backend": s.model_backend,
            "confidence": s.confidence_threshold
        },
        "limits": {
            "max_image_size_mb": s.max_image_size_mb,
            "timeout_seconds": s.request_timeout_seconds,
        }
    }

# Optional: Add a helper for development
def quick_validate():
    """Quick validation for development - prints warnings but doesn't raise"""
    warnings = validate_settings(strict=False)
    if warnings:
        print("⚠️  Configuration warnings:")
        for w in warnings:
            print(f"  • {w}")
    else:
        print("✅ Configuration looks good!")
    return warnings
