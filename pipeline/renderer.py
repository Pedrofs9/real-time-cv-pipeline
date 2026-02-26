"""
Annotated video renderer.
Shared by both download tasks — single source of truth for all rendering logic.
Handles drawing bounding boxes, traffic overlays, and timestamps on video frames.
"""
import bisect
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import cv2
import redis as redis_lib

from core.config import settings

logger = logging.getLogger(__name__)

# Color constants
COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    "person":     (107, 107, 255),
    "car":        (196, 205, 78),
    "truck":      (209, 183, 69),
    "bus":        (180, 206, 150),
    "motorcycle": (167, 234, 255),
    "bicycle":    (221, 160, 221),
}
DEFAULT_COLOR: Tuple[int, int, int] = (61, 217, 255)

STATUS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "LIGHT":    (72, 187, 120),
    "MODERATE": (85, 173, 246),
    "HEAVY":    (101, 101, 245),
}



def load_detections_from_redis(redis_client, job_id: str, total_frames: int) -> tuple[Dict, List]:
    """
    Batch-load all detection frames from Redis.
    Returns (ts_to_detection dict, sorted_timestamps list).
    """
    pipe = redis_client.pipeline(transaction=False)
    for i in range(total_frames):
        pipe.get(f"job:{job_id}:frame:{i}")
    raw_frames = pipe.execute()

    detection_frames = []
    for raw in raw_frames:
        if raw:
            try:
                detection_frames.append(json.loads(raw))
            except Exception:
                detection_frames.append(None)
        else:
            detection_frames.append(None)

    ts_to_detection = {
        fd["timestamp_ms"]: fd
        for fd in detection_frames
        if fd and fd.get("timestamp_ms") is not None
    }
    sorted_ts = sorted(ts_to_detection.keys())
    return ts_to_detection, sorted_ts


def render_annotated_video(
    source_video_path: str,
    output_path: str,
    ts_to_detection: Dict,
    sorted_ts: List,
    job_id: str,
    redis_client,
    progress_start: float = 0.0,
    progress_end: float = 100.0,
) -> str:
    """
    Render source video with bounding boxes and traffic overlay burned in.

    Args:
        source_video_path: Input video path.
        output_path: Where to write the annotated .mp4.
        ts_to_detection: Mapping of timestamp_ms → detection frame dict.
        sorted_ts: Sorted list of timestamp keys for bisect lookup.
        job_id: Used for Redis progress updates.
        redis_client: Active Redis client for progress reporting.
        progress_start: Progress % value at start of render phase (for two-phase jobs).
        progress_end: Progress % value at end of render phase.

    Returns:
        output_path on success. Raises RuntimeError on failure.
    """
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for rendering: {source_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    source_frame_idx = 0
    last_detection = None
    progress_range = progress_end - progress_start

    generated_at_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            source_frame_idx += 1

            # Nearest-detection lookup — carry forward on gaps between sampled frames
            if sorted_ts:
                pos = bisect.bisect_right(sorted_ts, current_ts_ms)
                if pos > 0:
                    last_detection = ts_to_detection[sorted_ts[pos - 1]]
                elif not last_detection:
                    last_detection = ts_to_detection[sorted_ts[0]]

            if last_detection:
                _draw_detections(frame, last_detection.get("detections", []))
                _draw_traffic_overlay(frame, last_detection.get("traffic", {}))

            _draw_timestamp(frame, current_ts_ms, width, height, generated_at_str)
            out.write(frame)

            # Report progress every 30 frames
            if source_frame_idx % 30 == 0 and total_source_frames > 0:
                render_fraction = source_frame_idx / total_source_frames
                total_pct = progress_start + render_fraction * progress_range
                try:
                    redis_client.hset(f"job:{job_id}:meta", mapping={
                        "download_progress": f"{total_pct:.1f}",
                        "download_status": "rendering",
                    })
                except Exception:
                    pass  # Non-critical — rendering continues regardless

    finally:
        cap.release()
        out.release()

    return output_path


def _draw_detections(frame, detections: List[Dict]):
    """
    Draw bounding boxes and labels on the frame.
    """
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        color = COLOR_MAP.get(det["class_name"], DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['score']:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_traffic_overlay(frame: np.ndarray, traffic: Dict[str, Any]) -> None:
    """
    Draw traffic statistics overlay on the frame.
    """
    if not traffic:
        return
    counts = traffic.get("counts", {})
    density = traffic.get("density", 0)
    status_str = traffic.get("status", "UNKNOWN")

    panel = frame.copy()
    cv2.rectangle(panel, (10, 10), (280, 210), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.65, frame, 0.35, 0, frame)

    y = 38
    # Label clarifies density is vehicle-based
    cv2.putText(frame, f"TRAFFIC: {status_str} ({density:.0f}%)",
                (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                STATUS_COLORS.get(status_str, (255, 255, 255)), 2, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, f"Vehicles: {counts.get('total_vehicles', 0)}",
                (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    y += 24
    for lbl, key in [
        ("Cars", "cars"),
        ("Motorcycles", "motorcycles"),
        ("Trucks", "trucks"),
        ("Buses", "buses"),
        ("Bicycles", "bicycles"),
        ("Persons", "persons"),   # shown for context, not used in density
    ]:
        cv2.putText(frame, f"{lbl}: {counts.get(key, 0)}",
                    (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        y += 24    

def _draw_timestamp(frame, current_ts_ms: float, width: int, height: int, generated_at: str):
    """
    Draw timestamp overlay on the frame.
    """
    ts_str = f"{current_ts_ms / 1000:.2f}s"
    cv2.putText(frame, f"{generated_at}  |  t={ts_str}",
                (width - 320, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)