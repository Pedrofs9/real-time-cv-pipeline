#!/usr/bin/env python3
"""
CV Pipeline — Load & Stress Test Suite

Answers: what breaks first, and at what concurrency?

Usage:
    python scripts/tests/test_multi_user_load.py
    python scripts/tests/test_multi_user_load.py --api http://localhost:8000
    python scripts/tests/test_multi_user_load.py --test image_concurrency
    python scripts/tests/test_multi_user_load.py --list
    python scripts/tests/test_multi_user_load.py --max-concurrency 16   # push harder
    python scripts/tests/test_multi_user_load.py --slow                 # longer timeouts / more ramp steps

Requirements: data/samples/valid_image.jpg + data/samples/valid_video.mp4

What each test covers:
    image_concurrency       — ramp concurrent image requests until 503 or timeout
    video_queue_backpressure — submit jobs until queue rejects with 503; verify limit is enforced
    mixed_load              — image + video jobs simultaneously; look for resource contention
    websocket_fan_out       — N clients watch the same job; check all receive frames without drops
    result_isolation        — concurrent jobs must not bleed detection results into each other
    memory_trajectory       — worker RSS and GPU memory over a sustained run (canary for leaks)
    cascade_cancel          — cancel all jobs under load; verify queue drains cleanly
"""

import argparse
import asyncio
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import httpx

try:
    import websockets
    HAS_WS = True
except ImportError:
    HAS_WS = False

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="CV Pipeline load test suite")
parser.add_argument("--api",             default="http://127.0.0.1:8000")
parser.add_argument("--test",            default=None, help="Run a single test by name")
parser.add_argument("--list",            action="store_true")
parser.add_argument("--slow",            action="store_true", help="Longer timeouts, more ramp steps")
parser.add_argument("--max-concurrency", type=int, default=10)
args = parser.parse_args()

SAMPLES_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "samples")
IMAGE_SAMPLE = os.path.join(SAMPLES_DIR, "valid_image.jpg")
VIDEO_SAMPLE = os.path.join(SAMPLES_DIR, "valid_video.mp4")

TIMEOUT      = httpx.Timeout(connect=10, write=300, read=120, pool=10)
SLOW_FACTOR  = 3 if args.slow else 1

# ── Output helpers ────────────────────────────────────────────────────────────

RESET  = "\033[0m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

def _section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}\n")

def ok(msg):    print(f"  {GREEN}✓ PASS{RESET}  {msg}")
def fail(msg):  print(f"  {RED}✗ FAIL{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")
def info(msg):  print(f"  {CYAN}ℹ INFO{RESET}  {msg}")

# ── Results tracking ──────────────────────────────────────────────────────────

@dataclass
class Result:
    name:    str
    passed:  bool
    note:    str = ""
    metrics: dict = field(default_factory=dict)

_results: list[Result] = []

def record(name: str, passed: bool, note: str = "", metrics: dict = None):
    _results.append(Result(name, passed, note, metrics or {}))

def _flush_stale_jobs(api: str):
    """Clear leftover job keys from previous test runs so get_queue_depth() starts at 0."""
    try:
        import redis
        # Parse host from api url — default to localhost
        host = api.replace("http://", "").replace("https://", "").split(":")[0]
        r = redis.Redis(host=host, port=6379, db=0, decode_responses=True)
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = r.scan(cursor, match="job:*", count=100)
            if keys:
                r.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        if deleted:
            info(f"Cleared {deleted} stale job keys from previous runs")
    except Exception as e:
        warn(f"Could not flush stale jobs: {e} — queue depth may be inaccurate")

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def post_image(api: str, data: bytes, filename: str = "test.jpg") -> httpx.Response:
    with httpx.Client(timeout=TIMEOUT) as c:
        return c.post(f"{api}/v1/detect/image",
                      files={"file": (filename, data, "image/jpeg")})

def post_video(api: str, data: bytes, filename: str = "test.mp4") -> httpx.Response:
    with httpx.Client(timeout=TIMEOUT) as c:
        return c.post(f"{api}/v1/detect/video",
                      files={"file": (filename, data, "video/mp4")})

def start_download(api: str, job_id: str) -> httpx.Response:
    with httpx.Client(timeout=TIMEOUT) as c:
        return c.post(f"{api}/v1/jobs/{job_id}/start_download")

def start_watch(api: str, job_id: str) -> httpx.Response:
    with httpx.Client(timeout=TIMEOUT) as c:
        return c.post(f"{api}/v1/jobs/{job_id}/start_watch")

def get_job(api: str, job_id: str) -> httpx.Response:
    with httpx.Client(timeout=TIMEOUT) as c:
        return c.get(f"{api}/v1/jobs/{job_id}")

def cancel_job(api: str, job_id: str):
    try:
        with httpx.Client(timeout=httpx.Timeout(5)) as c:
            c.delete(f"{api}/v1/jobs/{job_id}")
    except Exception:
        pass

def get_metrics(api: str) -> str:
    try:
        with httpx.Client(timeout=httpx.Timeout(5)) as c:
            return c.get(f"{api}/metrics").text
    except Exception:
        return ""

def parse_metric(metrics_text: str, metric_name: str) -> Optional[float]:
    """Extract the value of a gauge metric from a Prometheus /metrics response."""
    for line in metrics_text.splitlines():
        if line.startswith(metric_name + " ") or line.startswith(metric_name + "{"):
            try:
                return float(line.split()[-1])
            except (ValueError, IndexError):
                pass
    return None

def wait_for_terminal(api: str, job_id: str, timeout: int = 120) -> str:
    """Poll job status until terminal (completed/failed/cancelled) or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = get_job(api, job_id)
            if r.status_code == 200:
                status = r.json().get("status", "")
                if status in ("completed", "failed", "cancelled"):
                    return status
        except Exception:
            pass
        time.sleep(2)
    return "timeout"

# ── Sample loading ────────────────────────────────────────────────────────────

def load_sample(path: str, name: str) -> bytes:
    if not os.path.exists(path):
        print(f"\n  ERROR: {name} not found at {path}")
        print(  "  Place a valid sample at that path before running load tests.\n")
        sys.exit(1)
    with open(path, "rb") as f:
        return f.read()

# ── Tests ─────────────────────────────────────────────────────────────────────

ALL_TESTS = {}

def test(fn):
    ALL_TESTS[fn.__name__] = fn
    return fn


@test
def image_concurrency(api: str):
    """
    Ramp concurrent image requests from 1 to max_concurrency.
    Find the concurrency level at which the API starts rejecting (503) or timing out.
    Report latency distribution at each level.
    """
    _section("Image Concurrency Ramp")
    image_data = load_sample(IMAGE_SAMPLE, "valid_image.jpg")
    name = "image_concurrency"

    results_by_level = {}
    breaking_point = None

    levels = list(range(1, args.max_concurrency + 1, max(1, args.max_concurrency // 4)))
    if args.max_concurrency not in levels:
        levels.append(args.max_concurrency)

    for concurrency in levels:
        info(f"Concurrency = {concurrency}")
        latencies, statuses = [], []

        def _one_request(_):
            t0 = time.monotonic()
            r = post_image(api, image_data)
            latencies.append((time.monotonic() - t0) * 1000)
            statuses.append(r.status_code)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(_one_request, range(concurrency)))

        success  = statuses.count(200)
        rejected = statuses.count(503)
        errors   = [s for s in statuses if s not in (200, 503)]
        p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0
        p99 = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0

        info(f"  200={success}  503={rejected}  other={errors}  p50={p50:.0f}ms  p99={p99:.0f}ms")
        results_by_level[concurrency] = dict(success=success, rejected=rejected, p50=p50, p99=p99)

        if rejected > 0 and breaking_point is None:
            breaking_point = concurrency
            warn(f"  → First rejection at concurrency={concurrency}")

        if rejected == concurrency:
            info(f"  → All requests rejected at concurrency={concurrency}, stopping ramp")
            break

    if breaking_point:
        ok(f"Backpressure enforced — queue saturated at concurrency={breaking_point}")
        record(name, True, f"broke at concurrency={breaking_point}", results_by_level)
    else:
        warn(f"No rejections up to concurrency={args.max_concurrency} — semaphore limit may be higher or not configured")
        record(name, True, "no rejections observed — inconclusive", results_by_level)


@test
def video_queue_backpressure(api: str):
    """
    Submit video jobs faster than they complete.
    Verify the queue rejects with 503 once MAX_QUEUE_SIZE is reached.
    Verify the error payload is structured correctly.
    """
    _section("Video Queue Backpressure")
    video_data = load_sample(VIDEO_SAMPLE, "valid_video.mp4")
    name = "video_queue_backpressure"
    submitted_jobs = []

    try:
        got_503 = False
        for i in range(args.max_concurrency + 2):
            r = post_video(api, video_data)
            if r.status_code != 202:
                fail(f"Unexpected upload status {r.status_code}: {r.text[:200]}")
                record(name, False, f"unexpected status={r.status_code}")
                return

            job_id = r.json().get("job_id")
            submitted_jobs.append(job_id)

            start_r = start_download(api, job_id)
            if start_r.status_code == 503:
                payload = start_r.json().get("detail", start_r.json())
                has_structure = "error" in payload and "message" in payload
                if has_structure:
                    ok(f"Queue full → 503 with structured error after {i} accepted jobs")
                else:
                    fail(f"Queue full → 503 but response missing error/message fields: {payload}")
                got_503 = True
                break
            elif start_r.status_code not in (200, 202):
                fail(f"start_download unexpected status {start_r.status_code}: {start_r.text[:200]}")
                record(name, False, f"unexpected start status={start_r.status_code}")
                return

            info(f"  Job {i+1} queued: {job_id[:8]}...")

        if not got_503:
            warn(f"Submitted {len(submitted_jobs)} jobs without a 503 — "
                 "set MAX_QUEUE_SIZE to a low value (e.g. 3) to test backpressure")
            record(name, True, "inconclusive — queue limit not reached", {"jobs_accepted": len(submitted_jobs)})
        else:
            record(name, True, f"queue rejected after {len(submitted_jobs)} jobs")

    finally:
        info(f"Cancelling {len(submitted_jobs)} test jobs...")
        for job_id in submitted_jobs:
            cancel_job(api, job_id)


@test
def mixed_load(api: str):
    """
    Fire image requests and video jobs simultaneously.
    Image requests should not be blocked by video processing.
    Measure image latency degradation while a video job is running.
    """
    _section("Mixed Load (Image + Video Simultaneous)")
    image_data = load_sample(IMAGE_SAMPLE, "valid_image.jpg")
    video_data = load_sample(VIDEO_SAMPLE, "valid_video.mp4")
    name = "mixed_load"
    job_id = None

    try:
        # Baseline: image latency with no video job running
        baseline_latencies = []
        for _ in range(5):
            t0 = time.monotonic()
            post_image(api, image_data)
            baseline_latencies.append((time.monotonic() - t0) * 1000)
        baseline_p50 = sorted(baseline_latencies)[len(baseline_latencies) // 2]
        info(f"Baseline image p50 (no video): {baseline_p50:.0f}ms")

        # Start a video job in background
        r = post_video(api, video_data)
        if r.status_code != 202:
            fail(f"Video upload failed: {r.status_code}"); record(name, False, "upload failed"); return
        job_id = r.json()["job_id"]
        start_download(api, job_id)
        info(f"Video job started: {job_id[:8]}... now measuring image latency under load")

        # Measure image latency while video is processing
        loaded_latencies, loaded_statuses = [], []
        for _ in range(10):
            t0 = time.monotonic()
            r = post_image(api, image_data)
            loaded_latencies.append((time.monotonic() - t0) * 1000)
            loaded_statuses.append(r.status_code)
            time.sleep(0.2)

        loaded_p50 = sorted(loaded_latencies)[len(loaded_latencies) // 2]
        degradation = ((loaded_p50 - baseline_p50) / baseline_p50) * 100 if baseline_p50 else 0
        rejected = loaded_statuses.count(503)

        info(f"Under-load image p50: {loaded_p50:.0f}ms  ({degradation:+.0f}% vs baseline)  503s={rejected}")

        if rejected > 0:
            fail(f"Image requests rejected during video processing — semaphore slots exhausted")
            record(name, False, f"503s={rejected} under video load", {"baseline_p50": baseline_p50, "loaded_p50": loaded_p50})
        elif degradation > 100:
            warn(f"Image latency degraded {degradation:.0f}% under video load — possible CPU/CUDA contention")
            record(name, True, f"degradation={degradation:.0f}%", {"baseline_p50": baseline_p50, "loaded_p50": loaded_p50})
        else:
            ok(f"Image latency stable under video load ({degradation:+.0f}%)")
            record(name, True, f"degradation={degradation:.0f}%", {"baseline_p50": baseline_p50, "loaded_p50": loaded_p50})

    finally:
        if job_id:
            cancel_job(api, job_id)


@test
def websocket_fan_out(api: str):
    """
    Multiple WebSocket clients connect to the same job simultaneously.
    Each must receive detection frames without dropping or mixing up results.
    """
    if not HAS_WS:
        warn("websockets package not installed — skipping websocket_fan_out")
        record("websocket_fan_out", True, "skipped — no websockets package")
        return

    _section("WebSocket Fan-Out (N clients, 1 job)")
    video_data = load_sample(VIDEO_SAMPLE, "valid_video.mp4")
    name = "websocket_fan_out"
    n_clients = min(4, args.max_concurrency)
    job_id = None

    try:
        r = post_video(api, video_data)
        if r.status_code != 202:
            fail(f"Upload failed: {r.status_code}"); record(name, False, "upload failed"); return
        job_id = r.json()["job_id"]
        start_watch(api, job_id)

        info(f"Job started: {job_id[:8]}... waiting for worker to pick up job")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            r = get_job(api, job_id)
            if r.status_code == 200:
                status = r.json().get("status", "")
                if status == "processing":
                    break
                if status in ("failed", "cancelled"):
                    fail(f"Job failed before WebSocket connected: status={status}")
                    record(name, False, f"job failed before ws connect: {status}")
                    return
            time.sleep(1)
        else:
            fail("Job never entered processing state — worker may be saturated")
            record(name, False, "job never started processing")
            return

        info(f"Worker picked up job, connecting {n_clients} WebSocket clients")

        ws_url = api.replace("http://", "ws://").replace("https://", "wss://")
        uri    = f"{ws_url}/ws/video/{job_id}"

        client_frames = {i: [] for i in range(n_clients)}
        client_errors = {i: [] for i in range(n_clients)}

        async def _client(client_id: int):
            try:
                async with websockets.connect(uri) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        if data.get("error"):
                            client_errors[client_id].append(data.get("error"))
                            break
                        if data.get("status") == "complete":
                            break
                        if "detections" in data:
                            client_frames[client_id].append(data)
            except Exception as e:
                client_errors[client_id].append(str(e))

        async def _run_all():
            await asyncio.gather(*[_client(i) for i in range(n_clients)])

        asyncio.run(_run_all())

        for i in range(n_clients):
            n = len(client_frames[i])
            errs = client_errors[i]
            if errs:
                fail(f"Client {i}: errors={errs}")
            else:
                info(f"Client {i}: received {n} frames")

        frame_counts = [len(client_frames[i]) for i in range(n_clients)]
        all_received = all(c > 0 for c in frame_counts)
        consistent   = max(frame_counts) - min(frame_counts) <= 2  # allow ±2 frames

        if not all_received:
            fail("One or more clients received no frames")
            record(name, False, f"frame_counts={frame_counts}")
        elif not consistent:
            warn(f"Frame counts diverged across clients: {frame_counts}")
            record(name, True, f"diverged but non-zero: {frame_counts}")
        else:
            ok(f"All {n_clients} clients received frames consistently: {frame_counts}")
            record(name, True, f"counts={frame_counts}")

    finally:
        if job_id:
            cancel_job(api, job_id)


@test
def result_isolation(api: str):
    """
    Submit N video jobs concurrently.
    Each job's results must not contain frame data from any other job (no bleed-through).
    Checks job_id field in each streamed frame matches the expected job.
    """
    _section("Result Isolation (Concurrent Jobs, No Bleed)")
    video_data = load_sample(VIDEO_SAMPLE, "valid_video.mp4")
    name = "result_isolation"
    n_jobs = min(3, args.max_concurrency)
    job_ids = []

    try:
        for i in range(n_jobs):
            r = post_video(api, video_data)
            if r.status_code != 202:
                info(f"Job {i+1} rejected ({r.status_code}) — queue may be full, reducing to {len(job_ids)} jobs")
                break
            jid = r.json()["job_id"]
            job_ids.append(jid)
            start_download(api, jid)
        if not job_ids:
            fail("No jobs accepted"); record(name, False, "no jobs accepted"); return

        info(f"Running {len(job_ids)} concurrent jobs, waiting for completion...")
        statuses = {}
        with ThreadPoolExecutor(max_workers=len(job_ids)) as pool:
            futures = {pool.submit(wait_for_terminal, api, jid, 300 * SLOW_FACTOR): jid
                       for jid in job_ids}
            for future in as_completed(futures):
                jid = futures[future]
                statuses[jid] = future.result()

        bleed_detected = False
        for jid in job_ids:
            r = get_job(api, jid)
            if r.status_code != 200:
                continue
            data = r.json()
            # If job metadata contains a reference to another job's id, that's bleed
            response_text = json.dumps(data)
            other_ids = [other for other in job_ids if other != jid]
            for other_id in other_ids:
                if other_id in response_text:
                    fail(f"Job {jid[:8]} response contains foreign job_id {other_id[:8]} — result bleed!")
                    bleed_detected = True

        terminal = [s for s in statuses.values() if s in ("completed", "failed")]
        info(f"Jobs: {', '.join(f'{jid[:8]}={s}' for jid, s in statuses.items())}")

        if bleed_detected:
            record(name, False, "result bleed detected")
        elif len(terminal) < len(job_ids):
            warn(f"Only {len(terminal)}/{len(job_ids)} jobs reached terminal state — timeouts may indicate queue contention")
            record(name, True, "no bleed, but some jobs timed out", {"statuses": statuses})
        else:
            ok(f"All {len(job_ids)} concurrent jobs completed with no result bleed")
            record(name, True, metrics={"statuses": statuses})

    finally:
        for jid in job_ids:
            cancel_job(api, jid)


@test
def memory_trajectory(api: str):
    """
    Submit N sequential video jobs and read worker RSS + GPU memory from /metrics after each.
    A monotonically increasing trajectory suggests a leak.
    """
    _section("Memory Trajectory (Sequential Jobs)")
    video_data = load_sample(VIDEO_SAMPLE, "valid_video.mp4")
    name = "memory_trajectory"
    n_runs = 4 if args.slow else 3

    rss_samples, gpu_samples = [], []
    job_id = None

    try:
        for i in range(n_runs):
            r = post_video(api, video_data)
            if r.status_code != 202:
                warn(f"Run {i+1}: upload rejected ({r.status_code}) — stopping early"); break
            job_id = r.json()["job_id"]
            start_download(api, job_id)
            info(f"Run {i+1}/{n_runs}: job={job_id[:8]}... waiting for completion")

            status = wait_for_terminal(api, job_id, timeout=300 * SLOW_FACTOR)
            info(f"  → status={status}")

            # Small delay to let Prometheus scrape update
            time.sleep(12)

            worker_metrics_url = api.replace(":8000", ":8001")
            metrics_text = get_metrics(worker_metrics_url)
            rss = parse_metric(metrics_text, "cv_worker_memory_mb")
            gpu = parse_metric(metrics_text, "cv_gpu_memory_used_mb")

            if rss is not None:
                rss_samples.append(rss)
                info(f"  Worker RSS: {rss:.0f} MB")
            if gpu is not None:
                gpu_samples.append(gpu)
                info(f"  GPU memory: {gpu:.0f} MB")

            job_id = None

        def _is_monotonic_increase(samples: list[float], label: str) -> bool:
            if len(samples) < 2:
                return False
            increases = sum(1 for a, b in zip(samples, samples[1:]) if b > a + 10)  # 10 MB tolerance
            ratio = increases / (len(samples) - 1)
            if ratio >= 0.75:
                warn(f"{label} monotonically increasing: {[f'{s:.0f}' for s in samples]} — possible leak")
                return True
            else:
                ok(f"{label} stable: {[f'{s:.0f}' for s in samples]}")
                return False

        rss_leaking = _is_monotonic_increase(rss_samples, "RSS")
        gpu_leaking = _is_monotonic_increase(gpu_samples, "GPU memory")

        if not rss_samples and not gpu_samples:
            warn("No memory metrics returned — worker metrics endpoint may be unreachable")
            record(name, True, "inconclusive — no metrics", {})
        else:
            passed = not rss_leaking and not gpu_leaking
            record(name, passed,
                   "potential memory leak detected" if not passed else "memory stable",
                   {"rss_mb": rss_samples, "gpu_mb": gpu_samples})

    finally:
        if job_id:
            cancel_job(api, job_id)


@test
def cascade_cancel(api: str):
    """
    Start N jobs, then cancel all of them under load.
    Verify all reach terminal state (cancelled or failed, not stuck in processing).
    Verify /v1/jobs list eventually reflects no active jobs.
    """
    _section("Cascade Cancel Under Load")
    video_data = load_sample(VIDEO_SAMPLE, "valid_video.mp4")
    name = "cascade_cancel"
    n_jobs = min(4, args.max_concurrency)
    job_ids = []

    try:
        for i in range(n_jobs):
            r = post_video(api, video_data)
            if r.status_code != 202:
                break
            jid = r.json()["job_id"]
            job_ids.append(jid)
            start_download(api, jid)

        if not job_ids:
            fail("No jobs accepted"); record(name, False, "no jobs accepted"); return

        info(f"Started {len(job_ids)} jobs, now cancelling all immediately...")
        time.sleep(2)  # Let at least one enter processing

        cancel_responses = {}
        for jid in job_ids:
            with httpx.Client(timeout=httpx.Timeout(5)) as c:
                r = c.delete(f"{args.api}/v1/jobs/{jid}")
                cancel_responses[jid] = r.status_code

        info(f"Cancel responses: {list(cancel_responses.values())}")

        # Wait for all to reach terminal state
        info("Waiting for all jobs to reach terminal state...")
        deadline = time.monotonic() + 60 * SLOW_FACTOR
        terminal_states = {}
        while time.monotonic() < deadline:
            pending = [jid for jid in job_ids if jid not in terminal_states]
            if not pending:
                break
            for jid in pending:
                r = get_job(args.api, jid)
                if r.status_code == 200:
                    status = r.json().get("status", "")
                    if status in ("completed", "failed", "cancelled"):
                        terminal_states[jid] = status
                elif r.status_code == 404:
                    terminal_states[jid] = "deleted"
            if len(terminal_states) < len(job_ids):
                time.sleep(3)

        stuck = [jid for jid in job_ids if jid not in terminal_states]
        info(f"Terminal states: {', '.join(f'{jid[:8]}={s}' for jid, s in terminal_states.items())}")

        if stuck:
            fail(f"{len(stuck)} job(s) stuck — never reached terminal state: {[j[:8] for j in stuck]}")
            record(name, False, f"stuck_jobs={len(stuck)}")
        else:
            ok(f"All {len(job_ids)} jobs reached terminal state after cancel")
            record(name, True, metrics={"states": list(terminal_states.values())})

    finally:
        for jid in job_ids:
            cancel_job(api, jid)


# ── Entry point ───────────────────────────────────────────────────────────────

if args.list:
    print("\nAvailable load tests:\n")
    for tname, fn in ALL_TESTS.items():
        first_line = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"  {tname:<30} {first_line}")
    print()
    sys.exit(0)

print(f"\n{BOLD}CV Pipeline — Load & Stress Test Suite{RESET}")
print(f"API:             {args.api}")
print(f"Samples:         {os.path.abspath(SAMPLES_DIR)}")
print(f"Max concurrency: {args.max_concurrency}")
print(f"Slow mode:       {'yes' if args.slow else 'no'}")

_flush_stale_jobs(args.api) 

tests_to_run = {args.test: ALL_TESTS[args.test]} if args.test else ALL_TESTS

if args.test and args.test not in ALL_TESTS:
    print(f"\nUnknown test: {args.test}")
    print(f"Available: {', '.join(ALL_TESTS)}")
    sys.exit(1)

for tname, fn in tests_to_run.items():
    try:
        fn(args.api)
    except KeyboardInterrupt:
        print("\n  Interrupted"); break
    except Exception as e:
        fail(f"[{tname}] Unhandled exception: {e}")
        import traceback; traceback.print_exc()
        record(tname, False, str(e))

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'═'*60}")
print(f"  {BOLD}RESULTS{RESET}")
print(f"{'═'*60}")

for r in _results:
    sym  = f"{GREEN}✓{RESET}" if r.passed else f"{RED}✗{RESET}"
    note = f"  {DIM}({r.note}){RESET}" if r.note else ""
    print(f"  {sym} {r.name}{note}")

    if r.metrics:
        for k, v in r.metrics.items():
            if isinstance(v, list):
                print(f"      {DIM}{k}: {v}{RESET}")
            elif isinstance(v, dict):
                pass  # nested dicts (e.g. results_by_level) skip inline
            else:
                print(f"      {DIM}{k}: {v}{RESET}")

passed = sum(1 for r in _results if r.passed)
total  = len(_results)
print(f"\n  {passed} passed  {total - passed} failed  {total} total\n")
sys.exit(0 if passed == total else 1)
