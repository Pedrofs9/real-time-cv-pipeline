"""
Celery application configuration.
Defines broker/backend connections, task routing, queue definitions,
and serialization settings. Single source of truth for all Celery configuration.
"""
from celery import Celery
from core.config import settings
import logging

logger = logging.getLogger(__name__)

# Create Celery instance
celery_app = Celery(
    "video_tasks",
    broker=f"redis://{settings.redis_host}:{settings.redis_port}/0",
    backend=f"redis://{settings.redis_host}:{settings.redis_port}/1",
    include=["workers.video_tasks"]
)

# Optional configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Timezone
    timezone="UTC",
    enable_utc=True,
    
    # Task execution
    task_track_started=True,
    task_time_limit=settings.video_timeout_seconds,
    task_soft_time_limit=settings.video_timeout_seconds - 5,
    task_acks_late=True,  # Only acknowledge after completion
    task_reject_on_worker_lost=True,
    
    # Results
    result_expires=settings.results_ttl_seconds,
    
    # Queues
    task_default_queue="video_processing",
    task_queues={
        "video_processing": {
            "exchange": "video_processing",
            "routing_key": "video.process"
        },
        "high_priority": {
            "exchange": "high_priority",
            "routing_key": "video.process.high"
        }
    },
    
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["workers"])

logger.info(f"Celery configured: broker={settings.redis_host}:{settings.redis_port}")