"""
FP16 strategy: bypass Ultralytics' predict() pipeline entirely for FP16 mode.
Ultralytics does not guarantee dtype propagation through its internal preprocessor
after model.half() — it casts inputs to float32 before the forward pass on some
versions, causing c10::Half != float.

Solution: call self.model.model (the raw nn.Module) directly for both warmup and
inference when use_fp16=True. We handle preprocessing ourselves:
  resize → letterbox pad → BGR→RGB → normalize → float16 tensor → NMS

FP32 / CPU path — passes numpy to self.model() as before.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression
from typing import Dict, Any
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import uuid

from core.config import settings
from pipeline.base import ProcessingPipeline
from core.metrics import (
    INFERENCE_LATENCY, INFERENCE_DECODE_LATENCY,
    INFERENCE_MODEL_LATENCY, INFERENCE_POSTPROCESS_LATENCY,
    FRAMES_PROCESSED, FRAMES_FAILED, DETECTIONS_PER_FRAME,
    MODE_IMAGE, MODE_WATCH, MODE_DOWNLOAD, INFERENCE_SERIALIZE_LATENCY,
)

logger = logging.getLogger(__name__)


class DetectionPipeline(ProcessingPipeline):

    def __init__(self) -> None:
        super().__init__()
        logger.info(f"Loading model from {settings.model_path}...")
        start = time.perf_counter()

        try:
            self.model = YOLO(settings.model_path)
            self.model.to(settings.device)
            self.conf_threshold = settings.confidence_threshold
            self.iou_threshold = settings.iou_threshold

            self.use_fp16 = settings.use_fp16 and settings.device == "cuda"
            if self.use_fp16:
                self.model.half()
                logger.info("FP16 enabled - model weights cast to float16")
            else:
                if settings.use_fp16 and settings.device != "cuda":
                    logger.warning(
                        "USE_FP16=true but device is not CUDA - running FP32. "
                        "FP16 requires a CUDA-capable GPU."
                    )
                logger.info("Running FP32 inference")

            self._warmup()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

        load_time = time.perf_counter() - start
        logger.info(f"Model loaded in {load_time:.2f}s on {settings.device}")
        self.thread_pool = ThreadPoolExecutor(max_workers=settings.num_workers)

    def _warmup(self) -> None:
        """
        FP16: calls self.model.model (raw nn.Module) directly with a float16
        tensor, bypassing Ultralytics preprocessor which does not respect
        model.half() consistently across versions.

        FP32/CPU: standard Ultralytics call with uint8 numpy.
        """
        logger.info("Warming up model...")
        h = w = settings.model_input_size

        if self.use_fp16:
            dummy = torch.zeros(
                (1, 3, h, w), dtype=torch.float16, device=settings.device
            )
            with torch.no_grad():
                _ = self.model.model(dummy)
        else:
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            _ = self.model(
                dummy,
                imgsz=settings.model_input_size,
                conf=self.conf_threshold,
                verbose=False,
            )

        logger.info("Warmup complete")

    async def process(self, input_data: Any) -> Dict[str, Any]:
        """Routes input data to the appropriate processing method based on type."""
        if isinstance(input_data, bytes):
            return await self.process_image(input_data)
        raise ValueError(f"Unsupported input type: {type(input_data)}")

    async def process_image(
        self, image_bytes: bytes, mode: str = MODE_IMAGE
    ) -> Dict[str, Any]:
        """Runs image inference asynchronously by offloading to a thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool, self._infer, image_bytes, mode
        )

    def process_image_sync(
        self, image_bytes: bytes, mode: str = MODE_IMAGE
    ) -> Dict[str, Any]:
        """Synchronous wrapper around the inference pipeline for non-async callers."""
        return self._infer(image_bytes, mode)

    def process_frame_direct_sync(
        self, frame: np.ndarray, mode: str = MODE_WATCH
    ) -> Dict[str, Any]:
        """Runs inference directly on a numpy frame, bypassing the bytes decode step."""
        return self._infer_frame(frame, mode)

    def _infer(self, image_bytes: bytes, mode: str) -> Dict[str, Any]:
        """Decodes image bytes and dispatches to the model inference path."""
        request_id = str(uuid.uuid4())[:8]
        total_start = time.perf_counter()

        decode_start = time.perf_counter()
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(
                    "Image file header is valid but could not be decoded by OpenCV. "
                    "The file may be corrupted, truncated, or use an unsupported encoding "
                    "variant (e.g. CMYK JPEG, interlaced PNG). Please re-save and retry."
                )
        except ValueError:
            FRAMES_FAILED.labels(mode=mode, reason="decode_error").inc()
            raise
        except Exception as e:
            FRAMES_FAILED.labels(mode=mode, reason="decode_error").inc()
            return {"error": True, "message": str(e), "request_id": request_id,
                    "timing_ms": {"total_ms": 0}}

        decode_time = time.perf_counter() - decode_start
        INFERENCE_DECODE_LATENCY.observe(decode_time)
        return self._run_model(img, mode, request_id, total_start, decode_time)

    def _infer_frame(self, frame: np.ndarray, mode: str) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())[:8]
        total_start = time.perf_counter()

        decode_start = time.perf_counter()
        if frame is None or frame.size == 0:
            FRAMES_FAILED.labels(mode=mode, reason="decode_error").inc()
            return {
                "error": True,
                "message": "Received empty or null frame from video decoder.",
                "request_id": request_id,
                "timing_ms": {"total_ms": 0},
            }
        if frame.ndim != 3 or frame.shape[2] != 3:
            FRAMES_FAILED.labels(mode=mode, reason="decode_error").inc()
            return {
                "error": True,
                "message": f"Unexpected frame shape {frame.shape}. Expected HxWx3 BGR.",
                "request_id": request_id,
                "timing_ms": {"total_ms": 0},
            }
        decode_time = time.perf_counter() - decode_start
        INFERENCE_DECODE_LATENCY.observe(decode_time)
        return self._run_model(frame, mode, request_id, total_start, decode_time)

    def _preprocess_fp16(self, img: np.ndarray) -> tuple:
        """
        Manual preprocessing for the FP16 direct-model path.

        Letterbox resize, BGR->RGB, normalize to [0,1], cast to float16 CUDA tensor.
        Returns (tensor BCHW, scale_x, scale_y) for coordinate recovery.
        """
        h0, w0 = img.shape[:2]
        size = settings.model_input_size

        scale = size / max(h0, w0)
        nh, nw = int(h0 * scale), int(w0 * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((size, size, 3), dtype=np.uint8)
        padded[:nh, :nw] = resized

        tensor = torch.from_numpy(padded)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor[[2, 1, 0], :, :]   # BGR -> RGB
        tensor = tensor.to(device=settings.device, dtype=torch.float16).div_(255.0)
        tensor = tensor.unsqueeze(0)        # BCHW

        return tensor, w0 / nw, h0 / nh

    def _run_model(
        self,
        img: np.ndarray,
        mode: str,
        request_id: str,
        total_start: float,
        decode_time: float,
    ) -> Dict[str, Any]:
        """
        FP16: bypasses Ultralytics preprocessor, calls nn.Module directly,
              runs NMS via ultralytics.utils.ops.non_max_suppression.
        FP32: standard self.model() call.
        """
        infer_start = time.perf_counter()
        post_start = None
        detections = []

        try:
            if self.use_fp16:
                tensor, scale_x, scale_y = self._preprocess_fp16(img)
                with torch.no_grad():
                    raw = self.model.model(tensor)

                pred = raw[0] if isinstance(raw, (list, tuple)) else raw
                preds = non_max_suppression(
                    pred,
                    conf_thres=self.conf_threshold,
                    iou_thres=self.iou_threshold,
                )
                infer_time = time.perf_counter() - infer_start
                INFERENCE_MODEL_LATENCY.observe(infer_time)

                post_start = time.perf_counter()
                for det in preds:
                    if det is None or len(det) == 0:
                        continue
                    for *xyxy, conf, cls in det.tolist():
                        x1, y1, x2, y2 = xyxy
                        detections.append({
                            "bbox": [
                                round(x1 * scale_x, 1),
                                round(y1 * scale_y, 1),
                                round(x2 * scale_x, 1),
                                round(y2 * scale_y, 1),
                            ],
                            "score":      round(conf, 4),
                            "class_id":   int(cls),
                            "class_name": self.model.names[int(cls)],
                        })

            else:
                results = self.model(
                    img,
                    imgsz=settings.model_input_size,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                )
                infer_time = time.perf_counter() - infer_start
                INFERENCE_MODEL_LATENCY.observe(infer_time)

                post_start = time.perf_counter()
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detections.append({
                                "bbox":       [round(x1, 1), round(y1, 1),
                                               round(x2, 1), round(y2, 1)],
                                "score":      round(float(box.conf[0]), 4),
                                "class_id":   int(box.cls[0]),
                                "class_name": self.model.names[int(box.cls[0])],
                            })

        except Exception as e:
            FRAMES_FAILED.labels(mode=mode, reason="inference_error").inc()
            return {
                "error": True,
                "message": str(e),
                "request_id": request_id,
                "timing_ms": {"decode_ms": round(decode_time * 1000, 2), "total_ms": 0},
            }

        post_time = time.perf_counter() - (post_start or infer_start)
        INFERENCE_POSTPROCESS_LATENCY.observe(post_time)

        serial_start = time.perf_counter()
        result_dict = {
            "request_id": request_id,
            "detections": detections,
            "timing_ms": {
                "decode_ms":      round(decode_time * 1000, 2),
                "inference_ms":   round(infer_time * 1000, 2),
                "postprocess_ms": round(post_time * 1000, 2),
            },
        }
        serial_time = time.perf_counter() - serial_start
        INFERENCE_SERIALIZE_LATENCY.observe(serial_time)

        total_time = time.perf_counter() - total_start
        INFERENCE_LATENCY.labels(mode=mode).observe(total_time)
        FRAMES_PROCESSED.labels(mode=mode).inc()
        DETECTIONS_PER_FRAME.observe(len(detections))

        result_dict["timing_ms"]["serialize_ms"] = round(serial_time * 1000, 2)
        result_dict["timing_ms"]["total_ms"]     = round(total_time * 1000, 2)
        return result_dict