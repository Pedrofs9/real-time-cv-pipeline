#!/usr/bin/env python3
"""
Usage:
    python scripts/submit_video.py --video data/samples/valid_video.mp4 --mode watch
    python scripts/submit_video.py --video data/samples/valid_video.mp4 --mode download
"""
import argparse, asyncio, json, os, sys, time
import httpx, websockets

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--mode",  required=True, choices=["watch", "download"])
parser.add_argument("--api",   default="http://127.0.0.1:8000")
parser.add_argument("--out",   default=None) #which folder to save them to
args = parser.parse_args()

TIMEOUT = httpx.Timeout(connect=10, write=300, read=120, pool=10)

# ── Step 1: Upload ────────────────────────────────────────────────────────────
print(f"Uploading {args.video} ({os.path.getsize(args.video)/1024/1024:.1f}MB)...")
with httpx.Client(timeout=TIMEOUT) as client:
    with open(args.video, "rb") as f:
        r = client.post(
            f"{args.api}/v1/detect/video",
            files={"file": (os.path.basename(args.video), f, "video/mp4")},
        )

if r.status_code != 202:
    print(f"Upload failed: {r.status_code}"); print(r.text); sys.exit(1)

job_id = r.json()["job_id"]
print(f"Job ID: {job_id}")

# ── Step 2: Start processing ──────────────────────────────────────────────────
endpoint = "start_watch" if args.mode == "watch" else "start_download"
with httpx.Client(timeout=TIMEOUT) as client:
    r = client.post(f"{args.api}/v1/jobs/{job_id}/{endpoint}")

if r.status_code not in (200, 202):
    print(f"Start failed: {r.status_code}"); print(r.text); sys.exit(1)

print(f"Mode: {args.mode} | Status: {r.json().get('status')}")

# ── Watch mode ────────────────────────────────────────────────────────────────
if args.mode == "watch":
    ws_url = args.api.replace("http://", "ws://").replace("https://", "wss://")

    async def stream():
        uri = f"{ws_url}/ws/video/{job_id}"
        print(f"Connecting to {uri}\n")
        async with websockets.connect(uri) as ws:
            async for message in ws:
                data = json.loads(message)
                if data.get("error"):
                    print(f"Error: {data.get('message', data['error'])}"); break
                if data.get("status") == "complete":
                    print(f"Stream complete — job_status={data.get('job_status')}"); break
                ts      = data.get("timestamp_ms", "—")
                n_det   = len(data.get("detections", []))
                traffic = data.get("traffic", {})
                print(f"  t={ts}ms  detections={n_det}  traffic={traffic.get('status','—')}  density={traffic.get('density','—')}%")

    asyncio.run(stream())

# ── Download mode ─────────────────────────────────────────────────────────────
# ── Download mode ─────────────────────────────────────────────────────────────
else:
    print("Polling for completion...")
    with httpx.Client(timeout=TIMEOUT) as client:
        while True:
            data = client.get(f"{args.api}/v1/jobs/{job_id}/download_status").json()
            status, progress = data.get("download_status"), data.get("progress", 0)
            print(f"  {status}  {progress:.1f}%")
            if status == "ready": break
            if status == "failed": print(f"Failed: {data.get('error')}"); sys.exit(1)
            time.sleep(3)

        # Get original filename for a meaningful output name
        meta = client.get(f"{args.api}/v1/jobs/{job_id}").json()
        base = os.path.splitext(meta.get("filename", "video"))[0]

        os.makedirs("data/outputs", exist_ok=True)
        out_path = args.out or f"data/outputs/{base}_annotated_{job_id[:8]}.mp4"

        print(f"Downloading to {out_path}...")
        with client.stream("GET", f"{args.api}/v1/jobs/{job_id}/download", timeout=600) as r:
            with open(out_path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    print(f"Saved: {out_path}")