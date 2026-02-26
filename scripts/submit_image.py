#!/usr/bin/env python3
"""
Usage:
    python scripts/submit_image.py --image data/samples/valid_image.jpg
    python scripts/submit_image.py --image data/samples/valid_image.jpg --api http://localhost:8000
"""
import argparse
import json
import sys
import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--api", default="http://127.0.0.1:8000")
args = parser.parse_args()

with open(args.image, "rb") as f:
    r = httpx.post(f"{args.api}/v1/detect/image", files={"file": f}, timeout=30)

print(json.dumps(r.json(), indent=2))

if r.status_code != 200:
    sys.exit(1)
