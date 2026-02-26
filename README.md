# Real-Time CV Pipeline

YOLOv8-based object detection pipeline for traffic monitoring. Detects vehicles and pedestrians in images and video via a REST + WebSocket API with a real-time browser UI.

## Architecture
```
Browser / CLI  ←──HTTP/WS──→  FastAPI (CPU)  ──Celery──→  Worker (GPU)
                                     ↕                          ↕
                                   Redis  ←─────────────────────
                                     ↑
                               Prometheus
                              (scrapes both)
```

Image requests are handled synchronously in the API container. Video jobs are
queued via Redis and processed by the Celery worker on GPU, with results streamed
back over WebSocket (watch mode) or returned as an annotated `.mp4` (download mode).

## Quickstart

**Prerequisites:** Docker + Docker Compose, NVIDIA GPU + `nvidia-container-toolkit` (optional — see CPU note below).
**On Linux/macOS:**
```bash
git clone <your-repo-url> && cd real-time-cv-pipeline
make run
```

`make run` creates `.env` from `.env.example`, downloads `yolov8n.pt`, and starts
all services.

**On Windows:** (`make` is not available by default):
```bash
copy .env.example .env
pip install ultralytics
python scripts/download_yolov8.py
docker compose up --build
```


| URL | Service |
|---|---|
| http://localhost:8000 | UI + API |
| http://localhost:8000/docs | Swagger |
| http://localhost:8000/metrics | Prometheus metrics (raw) |
| http://localhost:9090 | Prometheus |
| http://localhost:3000 | Grafana (`admin` / `changeme`) |

Grafana credentials can be changed via `GF_SECURITY_ADMIN_USER` and
`GF_SECURITY_ADMIN_PASSWORD` in `.env` before first run.

Other make targets: `make stop`, `make clean`, `make logs`, `make rebuild`.

**CPU-only (no GPU):** Set `DEVICE=cpu` in `.env` and remove the
`deploy.resources` block from the worker service in `docker-compose.yml`.

## Running locally (without Docker)
```bash
pip install -r requirements.txt
python scripts/download_yolov8.py
python main.py
```

Requires Redis running locally. The `main.py` entrypoint reads host, port,
workers, and log level from `.env`. For hot-reload during development,
set `RELOAD=true` in `.env`.

Video processing requires a separate Celery worker:
```bash
celery -A workers.video_tasks worker --pool=solo --loglevel=info
```

## Project structure
```
real-time-cv-pipeline/
├── api/
│   └── app.py                  # FastAPI app — all endpoints
├── core/
│   ├── config.py               # All settings — single source of truth
│   ├── metrics.py              # Prometheus metric definitions
│   ├── celery_config.py        # Celery broker/backend config
│   └── file_validation.py      # Magic byte file type validation
├── pipeline/
│   ├── base.py                 # Abstract pipeline base class
│   ├── detection.py            # YOLOv8 inference pipeline (FP32 + FP16)
│   ├── video.py                # Video processor + Redis frame storage
│   ├── traffic.py              # Traffic density analysis
│   └── renderer.py             # Annotated video renderer
├── workers/
│   ├── video_tasks.py          # Celery tasks (watch + download modes)
│   └── worker_init.py          # Worker model loading/unloading
├── ui/
│   ├── index.html
│   ├── script.js               # WebSocket client + canvas overlay
│   └── style.css
├── scripts/
│   ├── download_yolov8.py      # Download model weights
│   ├── submit_image.py         # CLI image client
│   ├── submit_video.py         # CLI video client (watch + download)
│   ├── collect_metrics_video.py # Benchmark script (Prometheus-based)
│   └── tests/
│       ├── test_failure_modes.py  # End-to-end failure mode test suite
│       └── test_multi_user_load.py # Multi-user load and stress tests
├── data/
│   ├── samples/                # Sample image + video for testing
│   ├── uploads/                # Raw uploaded videos (runtime)
│   └── outputs/                # Annotated video outputs (runtime)
├── models/
│   └── yolov8n.pt              # Model weights (downloaded via script)
├── docker-compose.yml
├── Dockerfile
├── prometheus.yml
├── main.py                     # Local development entrypoint (non-Docker)
├── requirements.txt
└── .env
```

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cuda` | Inference device (`cuda` / `cpu`) |
| `USE_FP16` | `false` | FP16 inference — CUDA only, ~30-50% faster model pass |
| `MODEL_INPUT_SIZE` | `640` | Resize input to this square size |
| `CONFIDENCE_THRESHOLD` | `0.5` | Detection confidence cutoff |
| `IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `DOWNLOAD_FRAME_SAMPLE_RATE` | `1` | Every Nth frame in download mode (1 = all frames) |
| `WATCH_FRAME_SAMPLE_RATE` | `1` | Every Nth frame in watch mode |
| `SCENE_CHANGE_THRESHOLD` | `0.02` | Skip frames with < 2% mean pixel change. `0.0` disables |
| `MAX_FRAMES_PER_VIDEO` | `5000` | Hard frame cap per job |
| `MAX_VIDEO_SIZE_MB` | `500` | Video upload size limit |
| `MAX_IMAGE_SIZE_MB` | `10` | Image upload size limit |
| `MAX_QUEUE_SIZE` | `5` | Max concurrent active video jobs before 503 |
| `MAX_CONCURRENT_IMAGE_REQUESTS` | `4` | Max simultaneous image inference slots |
| `REQUEST_TIMEOUT_SECONDS` | `30` | Image inference slot wait timeout |
| `RESULTS_TTL_SECONDS` | `3600` | How long job results are kept in Redis |
| `DELETE_VIDEOS_AFTER_PROCESSING` | `false` | Delete source after download render. Never applies to watch mode |
| `METRICS_PORT` | `8001` | Worker Prometheus scrape port |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG` / `INFO` / `WARNING` / `ERROR`) |

## API
```bash
# Image — synchronous, returns detections immediately
curl -X POST http://localhost:8000/v1/detect/image -F "file=@image.jpg"

# Video — upload first, then choose a processing mode
curl -X POST http://localhost:8000/v1/detect/video -F "file=@video.mp4"     # → job_id
curl -X POST http://localhost:8000/v1/jobs/{job_id}/start_watch             # start watch mode
curl -X POST http://localhost:8000/v1/jobs/{job_id}/start_download          # start download mode

# Watch mode streams detections over WebSocket — use the UI or submit_video.py,
# not curl (curl does not support WebSocket)

# Job management
curl http://localhost:8000/v1/jobs                    # list recent jobs
curl http://localhost:8000/v1/jobs/{job_id}           # job status + progress
curl -X DELETE http://localhost:8000/v1/jobs/{job_id} # cancel job

# Download flow
curl http://localhost:8000/v1/jobs/{job_id}/download_status   # poll until ready
curl -OJ http://localhost:8000/v1/jobs/{job_id}/download      # save annotated .mp4
```

## CLI Scripts
```bash
python scripts/submit_image.py --image data/samples/valid_image.jpg
python scripts/submit_video.py --video data/samples/valid_video.mp4 --mode watch
python scripts/submit_video.py --video data/samples/valid_video.mp4 --mode download
```

## Benchmarking

Runs a fixed video (`data/samples/UA-DETRAC.mp4`) through the pipeline N times
and prints a Prometheus-backed comparison table (latency breakdown, FPS, GPU
utilization, job duration).
```bash
python scripts/collect_metrics_video.py --label "baseline" --window 5m
docker compose restart worker  # after any config change, wait ~15s
python scripts/collect_metrics_video.py --label "fp16" --window 5m
```

## Observability

Prometheus scrapes `api:8000/metrics` and `worker:8001/metrics` every 10 seconds.

| Metric | Description |
|---|---|
| `cv_inference_latency_seconds` | End-to-end per-frame latency (decode → serialize) |
| `cv_decode_latency_seconds` | Frame decode + preprocess time |
| `cv_model_latency_seconds` | Raw model forward-pass time |
| `cv_postprocess_latency_seconds` | NMS + result extraction time |
| `cv_serialize_latency_seconds` | Result dict construction time |
| `cv_video_processing_duration_seconds` | Total job duration end-to-end |
| `cv_gpu_memory_used_mb` | GPU memory allocated by the worker |
| `cv_worker_gpu_util_percent` | GPU utilization percent |
| `cv_worker_cpu_percent` | Worker process CPU usage |
| `cv_worker_memory_mb` | Worker process RSS memory |
| `cv_queue_depth` | Active jobs currently in the queue |
| `cv_http_request_latency_seconds` | HTTP request latency by endpoint |

## Tests

Place `data/samples/valid_image.jpg` and `data/samples/valid_video.mp4` before
running. All tests run against a live stack.

**Failure mode suite** — validates every robustness behaviour (bad inputs,
oversized files, job lifecycle, WebSocket errors, backpressure):
```bash
python scripts/tests/test_failure_modes.py               # all tests
python scripts/tests/test_failure_modes.py --test oversized_video
python scripts/tests/test_failure_modes.py --slow        # longer timeouts for slow hardware
python scripts/tests/test_failure_modes.py --destructive # queue flood (set MAX_QUEUE_SIZE=3 first)
python scripts/tests/test_failure_modes.py --list        # show all test names
```

**Multi-user load suite** — simulates concurrent users to find what breaks first:
```bash
python scripts/tests/test_multi_user_load.py                         # all tests
python scripts/tests/test_multi_user_load.py --test image_concurrency
python scripts/tests/test_multi_user_load.py --max-concurrency 16    # push harder
python scripts/tests/test_multi_user_load.py --slow                  # longer timeouts
python scripts/tests/test_multi_user_load.py --list                  # show all test names
```
