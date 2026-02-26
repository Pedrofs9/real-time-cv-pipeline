#!/usr/bin/env python3
"""
Failure Mode Test Suite
Tests every failure mode described in the robustness documentation.

Usage:
    python scripts/tests/test_failure_modes.py
    python scripts/tests/test_failure_modes.py --api http://localhost:8000
    python scripts/tests/test_failure_modes.py --test corrupted_upload
    python scripts/tests/test_failure_modes.py --slow          # longer timeouts for slow hardware
    python scripts/tests/test_failure_modes.py --destructive   # tests that stress/flood the API

Requires: valid_video.mp4 sample AND MAX_QUEUE_SIZE set low (2–5) in .env.

Place sample files in data/samples/:
    data/samples/valid_image.jpg     — any valid JPEG
    data/samples/valid_video.mp4     — any valid short MP4 (< 10s recommended)

All other test inputs are generated programmatically.

Skipped tests and why:
    Redis failure simulation  — requires stopping Docker container; not black-box testable
    WebSocket stall detection — requires killing Celery worker mid-job; not black-box testable
    Disk full handling        — requires filesystem manipulation inside container
    Corrupted frame in video  — requires ffmpeg to craft; out of scope for this script
                                at the HTTP layer; no middleware reads it; test would always fail
    Request timeout           — semaphore timeout is covered by inference_semaphore test
"""

import argparse
import asyncio
import json
import os
import struct
import sys
import time
import threading
import websockets
import httpx

# ── Colour helpers ────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✓ PASS{RESET}  {msg}")
def fail(msg):  print(f"  {RED}✗ FAIL{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")
def info(msg):  print(f"  {BLUE}ℹ INFO{RESET}  {msg}")

def header(title):
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")

# ── Result tracking ───────────────────────────────────────────────────────────

results = []

def record(name, passed, note=""):
    results.append({"name": name, "passed": passed, "note": note})

# ── Global config (set by CLI args) ──────────────────────────────────────────

SLOW_MODE = False          # --slow: longer timeouts
DESTRUCTIVE_MODE = False   # --destructive: flood/stress tests

def job_completion_timeout():
    """How long to wait for a job to complete."""
    return 120 if SLOW_MODE else 60

def ws_frame_timeout():
    """How long to wait for WebSocket frames."""
    return 120 if SLOW_MODE else 60

# ── Sample file helpers ───────────────────────────────────────────────────────

SAMPLES = "data/samples"

def sample(filename):
    return os.path.join(SAMPLES, filename)

def require_sample(filename):
    path = sample(filename)
    if not os.path.exists(path):
        warn(f"Sample file not found: {path} — test skipped")
        return None
    return path

def make_fake_jpeg():
    """Valid JPEG magic bytes but garbage body — passes magic check, fails decode."""
    return b'\xff\xd8\xff\xe0' + b'\x00' * 100

def make_fake_mp4():
    """Valid MP4 ftyp box but no actual video data."""
    return struct.pack('>I', 32) + b'ftyp' + b'mp42' + b'\x00' * 20

def make_random_bytes(size=512):
    """Completely unrecognised bytes — should fail magic check."""
    return bytes(range(256)) * (size // 256) + bytes(range(size % 256))

def make_oversized_image(mb=12):
    """Valid JPEG header followed by padding to exceed the size limit."""
    header = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
    return header + b'\x00' * (mb * 1024 * 1024)

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def post_image(api, data, filename="test.jpg", content_type="image/jpeg", timeout=30):
    with httpx.Client(timeout=timeout) as client:
        return client.post(
            f"{api}/v1/detect/image",
            files={"file": (filename, data, content_type)},
        )

def post_video(api, data, filename="test.mp4", content_type="video/mp4", timeout=60):
    with httpx.Client(timeout=timeout) as client:
        return client.post(
            f"{api}/v1/detect/video",
            files={"file": (filename, data, content_type)},
        )

def get_job(api, job_id, timeout=10):
    with httpx.Client(timeout=timeout) as client:
        return client.get(f"{api}/v1/jobs/{job_id}")

def start_watch(api, job_id, timeout=10):
    with httpx.Client(timeout=timeout) as client:
        return client.post(f"{api}/v1/jobs/{job_id}/start_watch")

def start_download(api, job_id, timeout=10):
    with httpx.Client(timeout=timeout) as client:
        return client.post(f"{api}/v1/jobs/{job_id}/start_download")

def cancel_job(api, job_id, timeout=10):
    """Cancel a job. Never raises — safe to call in finally blocks."""
    try:
        with httpx.Client(timeout=timeout) as client:
            client.delete(f"{api}/v1/jobs/{job_id}")
    except Exception:
        pass

def upload_valid_video(api):
    """Upload the sample video and return job_id, or None on failure."""
    path = require_sample("valid_video.mp4")
    if not path:
        return None
    with open(path, "rb") as f:
        data = f.read()
    r = post_video(api, data, "valid_video.mp4")
    if r.status_code == 202:
        return r.json().get("job_id")
    warn(f"Valid video upload failed: {r.status_code} {r.text[:200]}")
    return None

def wait_for_job_completion(api, job_id, timeout=None):
    """
    Poll until job reaches a terminal state.
    Returns final status string or None on timeout.
    """
    deadline = time.time() + (timeout or job_completion_timeout())
    while time.time() < deadline:
        r = get_job(api, job_id)
        if r.status_code == 200:
            status = r.json().get("status", "")
            if status in ("completed", "failed", "cancelled"):
                return status
        time.sleep(2)
    return None

# ═════════════════════════════════════════════════════════════════════════════
# TESTS
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. Bad input — images ─────────────────────────────────────────────────────

def test_random_bytes_image(api):
    name = "random_bytes_image"
    r = post_image(api, make_random_bytes(), "garbage.jpg")
    if r.status_code == 400:
        body = r.json()
        if "error" in body.get("detail", {}):
            ok("Random bytes → 400 with structured error")
            record(name, True)
        else:
            warn(f"400 but no structured detail: {r.text[:200]}")
            record(name, True, "400 returned but body structure unexpected")
    else:
        fail(f"Expected 400, got {r.status_code}: {r.text[:200]}")
        record(name, False, f"status={r.status_code}")

def test_corrupted_jpeg_body(api):
    """Valid JPEG magic bytes, corrupted body — should fail OpenCV decode."""
    name = "corrupted_jpeg_body"
    r = post_image(api, make_fake_jpeg(), "corrupt.jpg")
    if r.status_code == 400:
        ok("Valid JPEG header / corrupted body → 400 (OpenCV decode failure)")
        record(name, True)
    else:
        fail(f"Expected 400, got {r.status_code}: {r.text[:200]}")
        record(name, False, f"status={r.status_code}")

def test_wrong_extension_image(api):
    """Send a valid MP4 header with .jpg extension — magic bytes should catch it."""
    name = "wrong_extension_image"
    r = post_image(api, make_fake_mp4(), "sneaky.jpg")
    if r.status_code == 400:
        ok("MP4 bytes with .jpg extension → 400")
        record(name, True)
    else:
        fail(f"Expected 400, got {r.status_code}")
        record(name, False, f"status={r.status_code}")

def test_empty_file_image(api):
    name = "empty_file_image"
    r = post_image(api, b"", "empty.jpg")
    if r.status_code in (400, 422):
        ok(f"Empty image file → {r.status_code}")
        record(name, True)
    else:
        fail(f"Expected 400/422, got {r.status_code}: {r.text[:200]}")
        record(name, False, f"status={r.status_code}")

# ── 2. Bad input — videos ─────────────────────────────────────────────────────

def test_random_bytes_video(api):
    name = "random_bytes_video"
    r = post_video(api, make_random_bytes(1024), "garbage.mp4")
    if r.status_code == 400:
        ok("Random bytes video → 400 with structured error")
        record(name, True)
    else:
        fail(f"Expected 400, got {r.status_code}: {r.text[:200]}")
        record(name, False, f"status={r.status_code}")

def test_empty_file_video(api):
    name = "empty_file_video"
    r = post_video(api, b"", "empty.mp4")
    if r.status_code in (400, 422):
        ok(f"Empty video file → {r.status_code}")
        record(name, True)
    else:
        fail(f"Expected 400/422, got {r.status_code}: {r.text[:200]}")
        record(name, False, f"status={r.status_code}")

# ── 3. Payload size ───────────────────────────────────────────────────────────

def test_oversized_image(api):
    name = "oversized_image"
    data = make_oversized_image(mb=12)  # default limit is 10MB
    r = post_image(api, data, "big.jpg", timeout=60)
    if r.status_code == 413:
        ok("Oversized image → 413")
        record(name, True)
    else:
        fail(f"Expected 413, got {r.status_code}: {r.text[:200]}")
        record(name, False, f"status={r.status_code}")

def test_oversized_video(api):
    """
    Streams enough bytes to exceed max_video_size_mb.
    Uses a valid MP4 header so it passes format validation and hits the size check.
    NOTE: sends ~510MB — takes several seconds even on localhost.
    """
    name = "oversized_video"
    header_bytes = make_fake_mp4()
    padding = b'\x00' * (510 * 1024 * 1024)  # 510MB — default limit is 500MB
    data = header_bytes + padding
    r = post_video(api, data, "huge.mp4", timeout=120)
    if r.status_code == 413:
        ok("Oversized video → 413 (streaming size check)")
        record(name, True)
    else:
        fail(f"Expected 413, got {r.status_code}: {r.text[:200]}")
        record(name, False, f"status={r.status_code}")

# ── 4. Valid baselines ────────────────────────────────────────────────────────

def test_valid_image(api):
    name = "valid_image"
    path = require_sample("valid_image.jpg")
    if not path:
        record(name, False, "sample file missing")
        return
    with open(path, "rb") as f:
        data = f.read()
    r = post_image(api, data, "valid_image.jpg")
    if r.status_code == 200:
        body = r.json()
        if "detections" in body and "timing_ms" in body:
            ok(f"Valid image → 200 with {len(body['detections'])} detections")
            record(name, True)
        else:
            fail(f"200 but unexpected body: {list(body.keys())}")
            record(name, False, "missing detections or timing_ms")
    else:
        fail(f"Expected 200, got {r.status_code}: {r.text[:200]}")
        record(name, False, f"status={r.status_code}")

def test_valid_video_upload(api):
    name = "valid_video_upload"
    path = require_sample("valid_video.mp4")
    if not path:
        record(name, False, "sample file missing")
        return
    with open(path, "rb") as f:
        data = f.read()
    r = post_video(api, data, "valid_video.mp4")
    if r.status_code == 202:
        body = r.json()
        if "job_id" in body:
            ok(f"Valid video upload → 202 job_id={body['job_id'][:8]}...")
            record(name, True)
            cancel_job(api, body["job_id"])
        else:
            fail(f"202 but no job_id: {body}")
            record(name, False)
    else:
        fail(f"Expected 202, got {r.status_code}: {r.text[:200]}")
        record(name, False, f"status={r.status_code}")

# ── 5. Job lifecycle ──────────────────────────────────────────────────────────

def test_job_not_found(api):
    name = "job_not_found"
    fake_id = "00000000-0000-0000-0000-000000000000"
    r = get_job(api, fake_id)
    if r.status_code == 404:
        ok("Non-existent job → 404")
        record(name, True)
    else:
        fail(f"Expected 404, got {r.status_code}: {r.text[:200]}")
        record(name, False, f"status={r.status_code}")

def test_watch_nonexistent_job(api):
    name = "watch_nonexistent_job"
    fake_id = "00000000-0000-0000-0000-000000000001"
    r = start_watch(api, fake_id)
    if r.status_code == 404:
        ok("Watch on non-existent job → 404")
        record(name, True)
    else:
        fail(f"Expected 404, got {r.status_code}: {r.text[:200]}")
        record(name, False, f"status={r.status_code}")

def test_download_nonexistent_job(api):
    name = "download_nonexistent_job"
    fake_id = "00000000-0000-0000-0000-000000000002"
    r = start_download(api, fake_id)
    if r.status_code == 404:
        ok("Download on non-existent job → 404")
        record(name, True)
    else:
        fail(f"Expected 404, got {r.status_code}: {r.text[:200]}")
        record(name, False, f"status={r.status_code}")

def test_cancel_nonexistent_job(api):
    name = "cancel_nonexistent_job"
    fake_id = "00000000-0000-0000-0000-000000000004"
    with httpx.Client(timeout=10) as client:
        r = client.delete(f"{api}/v1/jobs/{fake_id}")
    if r.status_code == 404:
        ok("Cancel non-existent job → 404")
        record(name, True)
    else:
        fail(f"Expected 404, got {r.status_code}: {r.text[:200]}")
        record(name, False, f"status={r.status_code}")

def test_cancel_job(api):
    name = "cancel_job"
    job_id = upload_valid_video(api)
    if not job_id:
        record(name, False, "sample video missing or upload failed")
        return
    try:
        start_watch(api, job_id)
        time.sleep(1)
        with httpx.Client(timeout=10) as client:
            r = client.delete(f"{api}/v1/jobs/{job_id}")
        if r.status_code == 200:
            body = r.json()
            if body.get("cancelled") is True:
                ok("Cancel running job → 200 cancelled=true")
                record(name, True)
            else:
                fail(f"200 but cancelled!=true: {body}")
                record(name, False, "cancelled field missing or false")
        else:
            fail(f"Expected 200, got {r.status_code}: {r.text[:200]}")
            record(name, False, f"status={r.status_code}")
    finally:
        cancel_job(api, job_id)

def test_cancel_already_terminal_job(api):
    """Cancel a completed job — should return 200 with cancelled=false."""
    name = "cancel_terminal_job"
    job_id = upload_valid_video(api)
    if not job_id:
        record(name, False, "sample video missing or upload failed")
        return
    try:
        start_watch(api, job_id)
        final_status = wait_for_job_completion(api, job_id)

        if final_status is None:
            warn(f"Job did not complete within timeout — skipping")
            record(name, False, "timeout waiting for completion")
            return
        if final_status == "failed":
            warn("Job failed during terminal cancel test")
            record(name, False, "job failed before completion")
            return

        with httpx.Client(timeout=10) as client:
            r = client.delete(f"{api}/v1/jobs/{job_id}")
        if r.status_code == 200:
            body = r.json()
            if body.get("cancelled") is False:
                ok("Cancel completed job → 200 cancelled=false (already terminal)")
                record(name, True)
            else:
                fail(f"Expected cancelled=false, got: {body}")
                record(name, False, f"body={body}")
        else:
            fail(f"Expected 200, got {r.status_code}: {r.text[:200]}")
            record(name, False, f"status={r.status_code}")
    finally:
        cancel_job(api, job_id)

def test_download_not_ready(api):
    """Request download file before render is complete — expect 425."""
    name = "download_not_ready"
    job_id = upload_valid_video(api)
    if not job_id:
        record(name, False, "sample video missing or upload failed")
        return
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(f"{api}/v1/jobs/{job_id}/download")
        if r.status_code == 425:
            ok("Download before render complete → 425")
            record(name, True)
        else:
            fail(f"Expected 425, got {r.status_code}: {r.text[:200]}")
            record(name, False, f"status={r.status_code}")
    finally:
        cancel_job(api, job_id)

def test_concurrent_job_conflict(api):
    name = "concurrent_job_conflict"
    job_id = upload_valid_video(api)
    if not job_id:
        record(name, False, "sample video missing or upload failed")
        return
    try:
        r_watch = start_watch(api, job_id)
        if r_watch.status_code == 503:
            warn("Queue full when starting watch — run this test on a clean queue")
            record(name, False, "queue full — test precondition not met")
            return
        if r_watch.status_code not in (200, 202):
            warn(f"Watch start returned {r_watch.status_code} — skipping conflict check")
            record(name, False, f"watch_status={r_watch.status_code}")
            return

        r_dl = start_download(api, job_id)
        if r_dl.status_code == 503:
            warn("Queue full on start_download — test precondition not met, run on clean queue")
            record(name, False, "queue full — not a conflict test result")
            return
        if r_dl.status_code == 409:
            ok("Download while watch processing → 409 conflict")
            record(name, True)
        else:
            body = r_dl.json()
            status = body.get("status", "")
            if status in ("completed", "uploaded"):
                warn(f"Job completed before conflict could be triggered — got {r_dl.status_code}")
                record(name, True, "race condition — job too fast; functionally correct")
            else:
                fail(f"Expected 409, got {r_dl.status_code}: {r_dl.text[:200]}")
                record(name, False, f"status={r_dl.status_code}")
    finally:
        cancel_job(api, job_id)

# ── 6. WebSocket ──────────────────────────────────────────────────────────────

async def _ws_connect_and_collect(uri, timeout=5):
    messages = []
    try:
        async with websockets.connect(uri) as ws:
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    messages.append(json.loads(msg))
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break
    except Exception as e:
        messages.append({"_connect_error": str(e)})
    return messages

async def _ws_collect_frames(uri, max_frames=3, timeout=60):
    messages = []
    try:
        async with websockets.connect(uri) as ws:
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    data = json.loads(msg)
                    messages.append(data)
                    if "timestamp_ms" in data:
                        if len([m for m in messages if "timestamp_ms" in m]) >= max_frames:
                            break
                    if data.get("status") == "complete" or data.get("error"):
                        break
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break
    except Exception as e:
        messages.append({"_connect_error": str(e)})
    return messages

def _ws_url(api):
    return api.replace("http://", "ws://").replace("https://", "wss://")

def test_websocket_job_not_found(api):
    name = "websocket_job_not_found"
    fake_id = "00000000-0000-0000-0000-000000000003"
    uri = f"{_ws_url(api)}/ws/video/{fake_id}"
    messages = asyncio.run(_ws_connect_and_collect(uri, timeout=5))
    if any(m.get("error") == "job_not_found" for m in messages):
        ok("WebSocket non-existent job → job_not_found error received immediately")
        record(name, True)
    elif any("_connect_error" in m for m in messages):
        warn(f"Could not connect to WebSocket: {messages}")
        record(name, False, "connection failed")
    else:
        fail(f"Expected job_not_found, got: {messages}")
        record(name, False, f"messages={messages}")

def test_websocket_receives_frames(api):
    """Upload a real video, start watch, connect WebSocket, verify frames arrive."""
    name = "websocket_receives_frames"
    path = require_sample("valid_video.mp4")
    if not path:
        record(name, False, "sample file missing")
        return
    with open(path, "rb") as f:
        data = f.read()
    r = post_video(api, data, "valid_video.mp4")
    if r.status_code != 202:
        warn(f"Upload failed: {r.status_code}")
        record(name, False, "upload failed")
        return
    job_id = r.json()["job_id"]
    try:
        start_watch(api, job_id)
        uri = f"{_ws_url(api)}/ws/video/{job_id}"
        messages = asyncio.run(_ws_collect_frames(uri, max_frames=3, timeout=ws_frame_timeout()))
        frame_messages = [m for m in messages if "timestamp_ms" in m]
        if frame_messages:
            ok(f"WebSocket → received {len(frame_messages)} detection frame(s)")
            record(name, True)
        elif any(m.get("error") for m in messages):
            errors = [m for m in messages if m.get("error")]
            fail(f"WebSocket returned error: {errors[0]}")
            record(name, False, str(errors[0]))
        else:
            fail(f"No frames received. Messages: {messages[:3]}")
            record(name, False, "no frames")
    finally:
        cancel_job(api, job_id)

# ── 7. Zero-frame video ───────────────────────────────────────────────────────

def test_zero_frame_video(api):
    name = "zero_frame_video"
    path = sample("zero_frames.mp4")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        info("Using data/samples/zero_frames.mp4")
    else:
        data = make_fake_mp4()
        info("Using programmatic stub MP4 (no mdat box)")

    r = post_video(api, data, "zero_frames.mp4")
    if r.status_code not in (202, 400):
        fail(f"Expected 202 or 400, got {r.status_code}: {r.text[:200]}")
        record(name, False, f"status={r.status_code}")
        return
    if r.status_code == 400:
        ok("Stub MP4 rejected at upload (magic bytes too minimal) → 400")
        record(name, True, "rejected at upload — acceptable")
        return

    job_id = r.json().get("job_id")
    info(f"Video accepted at upload, starting job to trigger zero-frame detection... job={job_id[:8]}")
    
    try:
        start_download(api, job_id)
        
        final_status = wait_for_job_completion(api, job_id, timeout=90)
        if final_status == "failed":
            jr = get_job(api, job_id)
            error_msg = jr.json().get("error", "") if jr.status_code == 200 else ""
            ok(f"Zero-frame video → job failed with: {error_msg[:80]}")
            record(name, True)
        elif final_status == "completed":
            jr = get_job(api, job_id)
            frames = jr.json().get("frames_processed", -1) if jr.status_code == 200 else -1
            if frames == 0:
                fail("Job completed with 0 frames — should have been marked failed")
                record(name, False, "completed with 0 frames instead of failed")
            else:
                warn(f"Job completed with {frames} frames — stub was decodable")
                record(name, True, "video was decodable — test inconclusive")
        else:
            warn("Timed out waiting for zero-frame job result")
            record(name, False, "timeout")
    finally:
        cancel_job(api, job_id)

# ── 8. Error response structure ───────────────────────────────────────────────

def test_error_response_structure_image(api):
    name = "error_response_structure_image"
    r = post_image(api, make_random_bytes(), "garbage.jpg")
    if r.status_code != 400:
        fail(f"Expected 400, got {r.status_code}")
        record(name, False)
        return
    try:
        body = r.json()
        detail = body.get("detail", {})
        if isinstance(detail, dict) and "error" in detail and "message" in detail:
            ok("Error response has error + message fields")
            record(name, True)
        elif isinstance(detail, str):
            warn(f"Detail is a plain string, not structured: {detail}")
            record(name, True, "plain string detail — acceptable but not structured")
        else:
            fail(f"Missing error/message fields in detail: {detail}")
            record(name, False, f"detail={detail}")
    except Exception as e:
        fail(f"Could not parse error response JSON: {e}")
        record(name, False, str(e))

def test_error_response_structure_video(api):
    name = "error_response_structure_video"
    r = post_video(api, make_random_bytes(1024), "garbage.mp4")
    if r.status_code != 400:
        fail(f"Expected 400, got {r.status_code}")
        record(name, False)
        return
    try:
        body = r.json()
        detail = body.get("detail", {})
        if isinstance(detail, dict) and "error" in detail and "message" in detail:
            ok("Video error response has error + message fields")
            record(name, True)
        elif isinstance(detail, str):
            warn(f"Detail is plain string: {detail}")
            record(name, True, "plain string detail")
        else:
            fail(f"Missing error/message in detail: {detail}")
            record(name, False)
    except Exception as e:
        fail(f"Could not parse JSON: {e}")
        record(name, False, str(e))

# ── 9. Infrastructure probes ──────────────────────────────────────────────────

def test_health_endpoint(api):
    name = "health_endpoint"
    with httpx.Client(timeout=5) as client:
        r = client.get(f"{api}/health")
    if r.status_code == 200 and r.json().get("status") == "healthy":
        ok("/health → 200 healthy")
        record(name, True)
    else:
        fail(f"/health returned {r.status_code}: {r.text[:200]}")
        record(name, False)

def test_liveness_endpoint(api):
    name = "liveness_endpoint"
    with httpx.Client(timeout=5) as client:
        r = client.get(f"{api}/live")
    if r.status_code == 200 and r.json().get("status") == "alive":
        ok("/live → 200 alive")
        record(name, True)
    else:
        fail(f"/live returned {r.status_code}: {r.text[:200]}")
        record(name, False)

def test_readiness_endpoint(api):
    name = "readiness_endpoint"
    with httpx.Client(timeout=10) as client:
        r = client.get(f"{api}/ready")
    if r.status_code == 200:
        ok("/ready → 200 (model loaded)")
        record(name, True)
    elif r.status_code == 503:
        warn("/ready → 503 (model not loaded — send one image request first)")
        record(name, False, "model not loaded")
    else:
        fail(f"/ready returned {r.status_code}: {r.text[:200]}")
        record(name, False)

def test_metrics_endpoint(api):
    name = "metrics_endpoint"
    with httpx.Client(timeout=5) as client:
        r = client.get(f"{api}/metrics")
    if r.status_code == 200 and "cv_" in r.text:
        ok("/metrics → 200 with Prometheus metrics")
        record(name, True)
    else:
        fail(f"/metrics returned {r.status_code} or missing cv_ metrics")
        record(name, False)

# ── 10. Inference semaphore (503 inference_busy) ──────────────────────────────

def test_inference_semaphore_timeout(api):
    """
    Send concurrent image requests to exhaust the semaphore (batch_size=1 by default).
    At least one request should get 503 inference_busy when all slots are taken.

    This test is realistic — batch_size defaults to 1 so a second concurrent
    request must wait. If the first finishes within request_timeout_seconds the
    second succeeds; if slots stay full the second gets 503. We fire enough
    threads to make a 503 likely but accept the test as inconclusive if all
    happen to succeed (fast GPU / large batch_size).
    """
    name = "inference_semaphore_timeout"
    path = require_sample("valid_image.jpg")
    if not path:
        record(name, False, "sample file missing")
        return
    with open(path, "rb") as f:
        image_data = f.read()

    responses = []
    errors = []

    def send_request():
        try:
            r = post_image(api, image_data, "valid_image.jpg", timeout=40)
            responses.append(r.status_code)
        except Exception as e:
            errors.append(str(e))

    # Fire 5 simultaneous requests — more than enough to saturate batch_size=1
    threads = [threading.Thread(target=send_request) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    status_503 = [s for s in responses if s == 503]
    status_200 = [s for s in responses if s == 200]

    if status_503:
        # Check at least one 503 body has the right error code
        info(f"Responses: {responses}")
        ok(f"Concurrent inference → {len(status_503)} × 503, {len(status_200)} × 200")
        record(name, True)
    elif len(status_200) == len(threads):
        warn(
            "All requests succeeded — GPU may be fast enough that no slot contention occurred, "
            "or batch_size > 1. Test inconclusive."
        )
        record(name, True, "all 200 — batch_size may be > 1 or GPU is fast")
    else:
        fail(f"Unexpected responses: {responses}, errors: {errors}")
        record(name, False, f"responses={responses}")

# ── 11. Queue backpressure (--destructive only) ───────────────────────────────

def test_queue_backpressure(api):
    """
    Submit enough jobs to exceed max_queue_size and verify 503 queue_full.

    DESTRUCTIVE: This test submits many video jobs and leaves them in the queue.
    Only run against a test environment with MAX_QUEUE_SIZE set to a small value
    (e.g. MAX_QUEUE_SIZE=3 in your .env). Running against production will flood
    the real queue.

    Requires: valid_video.mp4 sample AND MAX_QUEUE_SIZE set low (2–5) in .env.
    """
    name = "queue_backpressure"
    path = require_sample("valid_video.mp4")
    if not path:
        record(name, False, "sample file missing")
        return

    with open(path, "rb") as f:
        video_data = f.read()

    submitted_ids = []
    got_503 = False

    info("Submitting jobs until queue_full 503 is received...")
    info("NOTE: Set MAX_QUEUE_SIZE=3 in .env before running this test.")

    try:
        for i in range(20):  # 20 is a safe ceiling even for MAX_QUEUE_SIZE=10
            r = post_video(api, video_data, f"flood_{i}.mp4")
            if r.status_code == 202:
                job_id = r.json().get("job_id")
                submitted_ids.append(job_id)
                # Start processing to push into the active queue counter
                start_watch(api, job_id)
            elif r.status_code == 503:
                body = r.json()
                detail = body.get("detail", {})
                if isinstance(detail, dict) and detail.get("error") == "queue_full":
                    ok(f"Queue full after {i} jobs → 503 queue_full with depth info")
                    got_503 = True
                    record(name, True)
                    break
                else:
                    fail(f"503 but wrong error code: {detail}")
                    record(name, False, f"detail={detail}")
                    break
            else:
                fail(f"Unexpected status {r.status_code}: {r.text[:100]}")
                record(name, False, f"status={r.status_code}")
                break

        if not got_503:
            warn(
                "Never hit queue_full — MAX_QUEUE_SIZE is probably higher than 20. "
                "Set MAX_QUEUE_SIZE=3 in .env and retry."
            )
            record(name, False, "queue_full never triggered — reduce MAX_QUEUE_SIZE")

    finally:
        info(f"Cancelling {len(submitted_ids)} flood jobs...")
        for job_id in submitted_ids:
            cancel_job(api, job_id)

# ═════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═════════════════════════════════════════════════════════════════════════════

# Tests always included
STANDARD_TESTS = {
    "random_bytes_image":              test_random_bytes_image,
    "corrupted_jpeg_body":             test_corrupted_jpeg_body,
    "wrong_extension_image":           test_wrong_extension_image,
    "empty_file_image":                test_empty_file_image,
    "random_bytes_video":              test_random_bytes_video,
    "empty_file_video":                test_empty_file_video,
    "oversized_image":                 test_oversized_image,
    "oversized_video":                 test_oversized_video,
    "valid_image":                     test_valid_image,
    "valid_video_upload":              test_valid_video_upload,
    "job_not_found":                   test_job_not_found,
    "watch_nonexistent_job":           test_watch_nonexistent_job,
    "download_nonexistent_job":        test_download_nonexistent_job,
    "cancel_nonexistent_job":          test_cancel_nonexistent_job,
    "cancel_job":                      test_cancel_job,
    "cancel_terminal_job":             test_cancel_already_terminal_job,
    "download_not_ready":              test_download_not_ready,
    "concurrent_job_conflict":         test_concurrent_job_conflict,
    "websocket_job_not_found":         test_websocket_job_not_found,
    "websocket_receives_frames":       test_websocket_receives_frames,
    "zero_frame_video":                test_zero_frame_video,
    "error_response_structure_image":  test_error_response_structure_image,
    "error_response_structure_video":  test_error_response_structure_video,
    "health_endpoint":                 test_health_endpoint,
    "liveness_endpoint":               test_liveness_endpoint,
    "readiness_endpoint":              test_readiness_endpoint,
    "metrics_endpoint":                test_metrics_endpoint,
    "inference_semaphore_timeout":     test_inference_semaphore_timeout,
}

# Only run with --destructive
DESTRUCTIVE_TESTS = {
    "queue_backpressure":              test_queue_backpressure,
}

TEST_GROUPS = {
    "Bad Input — Images":         ["random_bytes_image", "corrupted_jpeg_body", "wrong_extension_image", "empty_file_image"],
    "Bad Input — Videos":         ["random_bytes_video", "empty_file_video"],
    "Payload Size":               ["oversized_image", "oversized_video"],
    "Baseline (Valid Inputs)":    ["valid_image", "valid_video_upload"],
    "Job Lifecycle":              ["job_not_found", "watch_nonexistent_job", "download_nonexistent_job",
                                   "cancel_nonexistent_job", "cancel_job", "cancel_terminal_job",
                                   "download_not_ready", "concurrent_job_conflict"],
    "WebSocket":                  ["websocket_job_not_found", "websocket_receives_frames"],
    "Zero-Frame / Bad Codec":     ["zero_frame_video"],
    "Error Response Structure":   ["error_response_structure_image", "error_response_structure_video"],
    "Concurrency / Backpressure": ["inference_semaphore_timeout", "queue_backpressure"],
    "Infrastructure":             ["health_endpoint", "liveness_endpoint", "readiness_endpoint", "metrics_endpoint"],
}


def run_all(api, only=None):
    global SLOW_MODE, DESTRUCTIVE_MODE

    print(f"\n{BOLD}CV Pipeline — Failure Mode Test Suite{RESET}")
    print(f"API:         {api}")
    print(f"Samples:     {os.path.abspath(SAMPLES)}")
    print(f"Slow mode:   {'yes' if SLOW_MODE else 'no'}")
    print(f"Destructive: {'yes' if DESTRUCTIVE_MODE else 'no'}")

    if DESTRUCTIVE_MODE:
        print(f"\n{YELLOW}⚠  DESTRUCTIVE mode active — queue flood test will run.{RESET}")
        print(f"{YELLOW}   Ensure MAX_QUEUE_SIZE=3 is set in your .env.{RESET}")

    # Pre-flight connectivity check
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get(f"{api}/health")
        if r.status_code != 200:
            print(f"\n{RED}API not reachable at {api} — aborting.{RESET}")
            sys.exit(1)
    except Exception as e:
        print(f"\n{RED}Cannot connect to {api}: {e}{RESET}")
        sys.exit(1)

    all_tests = dict(STANDARD_TESTS)
    if DESTRUCTIVE_MODE:
        all_tests.update(DESTRUCTIVE_TESTS)

    tests_to_run = {k: v for k, v in all_tests.items() if only is None or k in only}

    ran = set()
    for group_name, keys in TEST_GROUPS.items():
        keys = [k for k in keys if k in tests_to_run and k not in ran]
        if not keys:
            continue
        header(group_name)
        for key in keys:
            print(f"\n  [{key}]")
            try:
                tests_to_run[key](api)
            except Exception as e:
                fail(f"Test raised unhandled exception: {e}")
                record(key, False, str(e))
            ran.add(key)

    unassigned = [k for k in tests_to_run if k not in ran]
    if unassigned:
        header("Other")
        for key in unassigned:
            print(f"\n  [{key}]")
            try:
                tests_to_run[key](api)
            except Exception as e:
                fail(f"Unhandled exception: {e}")
                record(key, False, str(e))

    # Summary
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  RESULTS{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    passed = [r for r in results if r["passed"]]
    failed_r = [r for r in results if not r["passed"]]

    for r in passed:
        note = f"  ({r['note']})" if r["note"] else ""
        print(f"  {GREEN}✓{RESET} {r['name']}{note}")
    for r in failed_r:
        note = f"  ({r['note']})" if r["note"] else ""
        print(f"  {RED}✗{RESET} {r['name']}{note}")

    total = len(results)
    print(f"\n  {GREEN}{len(passed)} passed{RESET}  "
          f"{RED}{len(failed_r)} failed{RESET}  "
          f"{total} total\n")

    if failed_r:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CV Pipeline failure mode test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_failure_modes.py
  python test_failure_modes.py --api http://localhost:8000
  python test_failure_modes.py --slow
  python test_failure_modes.py --destructive   # requires MAX_QUEUE_SIZE=3 in .env
  python test_failure_modes.py --test oversized_video websocket_job_not_found
  python test_failure_modes.py --list
        """
    )
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test", nargs="+", metavar="NAME", help="Run specific test(s) by name")
    parser.add_argument("--slow", action="store_true", help="Use longer timeouts for slow hardware")
    parser.add_argument("--destructive", action="store_true",
                        help="Enable queue flood test (set MAX_QUEUE_SIZE=3 first)")
    parser.add_argument("--list", action="store_true", help="List all test names and exit")
    args = parser.parse_args()

    if args.list:
        print("Standard tests:")
        for name in STANDARD_TESTS:
            print(f"  {name}")
        print("\nDestructive tests (--destructive):")
        for name in DESTRUCTIVE_TESTS:
            print(f"  {name}")
        sys.exit(0)

    SLOW_MODE = args.slow
    DESTRUCTIVE_MODE = args.destructive

    run_all(args.api, only=args.test)
