from pipeline.detection import DetectionPipeline
from pipeline.video import VideoProcessor
from core.config import settings
import logging
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

_detection_pipeline = None
_processor = None
_load_error: Exception = None  # Stores the reason init failed, for clear diagnostics


def get_processor() -> VideoProcessor:
    """
    Return the initialized VideoProcessor.

    If the processor was not initialized at startup (signal didn't fire,
    or load_model raised), attempt lazy initialization now before failing.
    This makes the worker self-healing for the case where worker_init.connect
    did not fire — e.g. certain Celery + solo pool + platform combinations.
    """
    global _processor, _load_error

    if _processor is not None:
        return _processor

    # If we already know why init failed, log it clearly before retrying
    if _load_error is not None:
        logger.warning(
            f"Previous load_model() failed with: {_load_error}. "
            "Attempting lazy re-initialization..."
        )

    try:
        load_model()
    except Exception as e:
        raise RuntimeError(
            f"Processor could not be initialized: {e}. "
            f"Check that the model file exists at '{settings.model_path}' "
            f"and that the device '{settings.device}' is available."
        ) from e

    return _processor


def load_model() -> None:
    """
    Load the detection pipeline and video processor.

    Called explicitly by the worker_init signal handler at startup.
    Also called lazily by get_processor() if the signal did not fire.

    Raises RuntimeError if the model file is missing or loading fails.
    Sets _load_error on failure so get_processor() can report clearly.
    """
    global _detection_pipeline, _processor, _load_error

    logger.info("Loading detection pipeline...")
    logger.info(f"  model_path : {settings.model_path}")
    logger.info(f"  device     : {settings.device}")
    logger.info(f"  use_fp16   : {settings.use_fp16}")

    # Hard check — fail immediately with a clear message rather than letting
    # YOLO raise a cryptic FileNotFoundError deep in Ultralytics internals.
    model_path = Path(settings.model_path)
    if not model_path.exists():
        _load_error = FileNotFoundError(
            f"Model weights not found at '{model_path}'. "
            "Run scripts/download_yolov8.py or mount the models/ directory."
        )
        raise _load_error

    try:
        _detection_pipeline = DetectionPipeline()
        _processor = VideoProcessor(_detection_pipeline)
        _load_error = None  # Clear any previous error on success
        logger.info(f"✅ Detection pipeline ready on {settings.device}")
    except Exception as e:
        _detection_pipeline = None
        _processor = None
        _load_error = e
        logger.error(f"❌ DetectionPipeline failed to load: {e}", exc_info=True)
        raise


def unload_model() -> None:
    """Releases the detection pipeline and processor from memory and clears the GPU cache."""
    global _detection_pipeline, _processor
    _detection_pipeline = None
    _processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Detection pipeline and processor unloaded")