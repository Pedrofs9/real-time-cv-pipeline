#!/usr/bin/env python3
"""
scripts/collect_metrics_video.py

Runs a fixed video through the download pipeline NUM_RUNS times,
collects Prometheus metrics after each run, and prints an averaged summary.

Usage:
    python scripts/collect_metrics_video.py --label "baseline"
    python scripts/collect_metrics_video.py --label "input_320" --window 5m
"""
import argparse
import json
import os
import time
import subprocess
import sys
from datetime import datetime

import httpx

PROMETHEUS   = "http://localhost:9090"
RESULTS_FILE = "benchmark_results.json"
VIDEO_PATH   = "data/samples/UA-DETRAC.mp4"
NUM_RUNS     = 2


def window_to_seconds(window: str) -> int:
    """
    Convert Prometheus time window string to seconds.
    Examples: "5m" -> 300, "90s" -> 90, "1h" -> 3600
    """
    window = window.strip()
    if window.endswith('m'):
        return int(window[:-1]) * 60
    elif window.endswith('s'):
        return int(window[:-1])
    elif window.endswith('h'):
        return int(window[:-1]) * 3600
    else:
        try:
            return int(window)
        except ValueError:
            raise ValueError(f"Unsupported window format: '{window}'. Use e.g. '5m', '120s', '1h'")


def get_queries(window: str) -> dict:
    window_seconds = window_to_seconds(window)
    return {
        # ── Stage breakdown ────────────────────────────────────────────────────
        # decode/model/postprocess/serialize have no mode label
        "decode_p50_ms":      f'histogram_quantile(0.50, sum(increase(cv_decode_latency_seconds_bucket[{window}])) by (le)) * 1000',
        "model_p50_ms":       f'histogram_quantile(0.50, sum(increase(cv_model_latency_seconds_bucket[{window}])) by (le)) * 1000',
        "postprocess_p50_ms": f'histogram_quantile(0.50, sum(increase(cv_postprocess_latency_seconds_bucket[{window}])) by (le)) * 1000',
        "serialize_p50_ms":   f'histogram_quantile(0.50, sum(increase(cv_serialize_latency_seconds_bucket[{window}])) by (le)) * 1000',

        # ── End-to-end inference ───────────────────────────────────────────────
        "inference_p50_ms":   f'histogram_quantile(0.50, sum(increase(cv_inference_latency_seconds_bucket{{mode="download"}}[{window}])) by (le)) * 1000',
        "inference_p95_ms":   f'histogram_quantile(0.95, sum(increase(cv_inference_latency_seconds_bucket{{mode="download"}}[{window}])) by (le)) * 1000',

        # ── Throughput ─────────────────────────────────────────────────────────
        "throughput_fps":     f'sum(increase(cv_frames_processed_total{{mode="download"}}[{window}])) / {window_seconds}',

        # ── Job duration ───────────────────────────────────────────────────────
        "job_duration_s":     f'sum(increase(cv_video_processing_duration_seconds_sum{{mode="download"}}[{window}])) / sum(increase(cv_video_processing_duration_seconds_count{{mode="download"}}[{window}]))',

        # ── Resource usage — worker-side ───────────────────────────────────────
        "worker_cpu_percent":  f'avg_over_time(cv_worker_cpu_percent{{instance="worker:8001"}}[{window}])',
        "worker_memory_mb":    f'avg_over_time(cv_worker_memory_mb{{instance="worker:8001"}}[{window}])',
        "gpu_util_percent":    f'avg_over_time(cv_worker_gpu_util_percent{{instance="worker:8001"}}[{window}])',
        "gpu_memory_peak_mb":  f'max_over_time(cv_gpu_memory_used_mb{{instance="worker:8001"}}[{window}])',

        # ── Backpressure ───────────────────────────────────────────────────────
        "queue_depth_max":     f'max_over_time(cv_queue_depth[{window}])',
    }


def query_prometheus(promql: str) -> float | None:
    try:
        r = httpx.get(
            f"{PROMETHEUS}/api/v1/query",
            params={"query": promql},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        if data["status"] != "success":
            return None
        results = data.get("data", {}).get("result", [])
        if not results:
            return None
        value = float(results[0]["value"][1])
        # NaN from Prometheus means no data — treat as None
        if value != value:  # NaN check
            return None
        return round(value, 3)
    except Exception as e:
        print(f"  ⚠️  Query failed: {e}")
        return None


def flush_redis():
    """Clear Redis between runs to prevent metric bleed from previous jobs."""
    print("  🗑  Flushing Redis...")
    subprocess.run(
        ["docker", "exec", "cv-redis", "redis-cli", "FLUSHDB"],
        check=True, capture_output=True
    )
    time.sleep(2)


def run_video_job():
    print(f"  ▶  Submitting {VIDEO_PATH}...")
    subprocess.run([
        sys.executable, "scripts/submit_video.py",
        "--video", VIDEO_PATH,
        "--mode", "download",
    ], check=True)


def collect_metrics(label: str, window: str, wait: int) -> dict:
    print(f"  ⏱  Waiting {wait}s for Prometheus scrape...")
    time.sleep(wait)
    row = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "window": window,
    }
    for name, promql in get_queries(window).items():
        row[name] = query_prometheus(promql)
    return row


def average_runs(runs: list) -> dict:
    skip = {"label", "timestamp", "window"}
    numeric_keys = [k for k in runs[0] if k not in skip]
    avg = {k: v for k, v in runs[0].items() if k in skip}
    avg["label"] = runs[0]["label"]
    avg["timestamp"] = datetime.now().isoformat()
    for k in numeric_keys:
        values = [r[k] for r in runs if r.get(k) is not None]
        avg[k] = round(sum(values) / len(values), 3) if values else None
    return avg


def fmt(v) -> str:
    return f"{v:.2f}" if v is not None else "N/A"


def print_table(results: list):
    header = (
        f"{'Config':<22} {'Decode':>7} {'Model':>7} {'Post':>6} "
        f"{'InfP50':>7} {'InfP95':>7} {'FPS':>6} "
        f"{'GPU MB':>7} {'GPU%':>6} {'CPU%':>6} {'Job(s)':>7}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        print(
            f"{r['label'][:22]:<22} "
            f"{fmt(r.get('decode_p50_ms')):>7} "
            f"{fmt(r.get('model_p50_ms')):>7} "
            f"{fmt(r.get('postprocess_p50_ms')):>6} "
            f"{fmt(r.get('inference_p50_ms')):>7} "
            f"{fmt(r.get('inference_p95_ms')):>7} "
            f"{fmt(r.get('throughput_fps')):>6} "
            f"{fmt(r.get('gpu_memory_peak_mb')):>7} "
            f"{fmt(r.get('gpu_util_percent')):>6} "
            f"{fmt(r.get('worker_cpu_percent')):>6} "
            f"{fmt(r.get('job_duration_s')):>7}"
        )
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",  required=True, help="Config label, e.g. baseline, fp16, input_320")
    parser.add_argument("--window", default="5m",  help="Prometheus lookback window (default: 5m)")
    parser.add_argument("--wait",   type=int, default=20, help="Seconds to wait for Prometheus scrape after each run")
    args = parser.parse_args()

    # Validate window format early
    try:
        window_to_seconds(args.window)
    except ValueError as e:
        print(f"❌ {e}")
        raise SystemExit(1)

    print(f"\n{'='*50}")
    print(f"Benchmark: {args.label}  ({NUM_RUNS} runs, window={args.window})")
    print(f"{'='*50}")
    print("NOTE: Restart worker before changing optimization config:")
    print("      docker compose restart worker  # then wait ~15s\n")

    runs = []
    for i in range(NUM_RUNS):
        print(f"\n── Run {i+1}/{NUM_RUNS} ──────────────────────────────")
        flush_redis()
        run_video_job()
        metrics = collect_metrics(args.label, args.window, args.wait)
        runs.append(metrics)

        print(
            f"  inference_p50={fmt(metrics.get('inference_p50_ms'))}ms  "
            f"model_p50={fmt(metrics.get('model_p50_ms'))}ms  "
            f"fps={fmt(metrics.get('throughput_fps'))}  "
            f"gpu_util={fmt(metrics.get('gpu_util_percent'))}%"
        )

    avg = average_runs(runs)

    existing = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            existing = json.load(f)
    existing.append(avg)
    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    print_table(runs + [{"label": f">>> {args.label}_AVG", **avg}])
    print(f"\n✅ Average saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()