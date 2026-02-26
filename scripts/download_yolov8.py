#!/usr/bin/env python
"""
Download YOLOv8 model weights into the local models/ directory.

"""

from ultralytics import YOLO
from pathlib import Path
import shutil
import sys


MODEL_NAME = "yolov8n.pt"
MODELS_DIR = Path("models")
TARGET_PATH = MODELS_DIR / MODEL_NAME


def main():
    print("🔍 Checking model directory...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # If already exists, skip
    if TARGET_PATH.exists():
        print(f"Model already exists at {TARGET_PATH}")
        print("Skipping download.")
        return

    print("Downloading YOLOv8n weights...")
    
    # This downloads to ~/.cache/ultralytics
    model = YOLO(MODEL_NAME)

    # Find downloaded file in cache
    cached_path = Path(model.ckpt_path)

    if not cached_path.exists():
        print("Download failed.")
        sys.exit(1)

    print(f"Copying model to {TARGET_PATH}")
    shutil.copy2(cached_path, TARGET_PATH)

    size_mb = TARGET_PATH.stat().st_size / (1024 * 1024)
    print(f"Model saved to {TARGET_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
