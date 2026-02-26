# Dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download the YOLOv8 model (optional – can also be mounted as a volume)
#RUN python scripts/download_yolov8.py

# Expose the FastAPI port
EXPOSE 8000

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command (can be overridden by docker-compose)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]