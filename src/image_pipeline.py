from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import pandas as pd

from src.live_pipeline import MODEL_PATH, load_calibration, pixel_to_ground
from src.single_camera_lab import DetectorConfig, detect_frame, load_yolo


@dataclass(frozen=True)
class ImageConfig:
    confidence: float = 0.15
    image_size: int = 1280
    device: str = "auto"
    max_det: int = 1000
    iou: float = 0.55
    half: bool = True
    tiled: bool = False
    tile_size: int = 768
    tile_overlap: float = 0.20
    roi: tuple[float,float,float,float] | None = None
    model_path: str | None = None


@dataclass
class ImageRun:
    name: str
    original_rgb: np.ndarray
    annotated_rgb: np.ndarray
    detections: pd.DataFrame
    annotated_jpeg: bytes
    calibration_name: str | None
    inference_ms: float = 0.0
    tiles_processed: int = 1
    device_used: str = "cpu"


def load_model(model_path: str | Path | None = None):
    return load_yolo(model_path or MODEL_PATH)


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không đọc được ảnh. Hãy dùng JPG/JPEG/PNG/WEBP/BMP hợp lệ.")
    return frame


def process_image_bytes(
    data: bytes,
    name: str,
    config: ImageConfig,
    calibration_name: str | None = None,
    model: Any | None = None,
) -> ImageRun:
    frame = decode_image(data)
    original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    model = model or load_model(config.model_path)
    dcfg = DetectorConfig(
        confidence=config.confidence, image_size=config.image_size, max_det=config.max_det,
        iou=config.iou, device=config.device, half=config.half, tiled=config.tiled,
        tile_size=config.tile_size, tile_overlap=config.tile_overlap, roi=config.roi,
    )
    det = detect_frame(frame, dcfg, model=model)
    rows = det.detections.copy()
    rows["ground_x"] = np.nan; rows["ground_y"] = np.nan
    if calibration_name and not rows.empty:
        calib = load_calibration(calibration_name)
        ground = [pixel_to_ground(float(r.foot_x), float(r.foot_y), *calib) for r in rows.itertuples()]
        rows["ground_x"] = [g[0] for g in ground]; rows["ground_y"] = [g[1] for g in ground]

    bgr = cv2.cvtColor(det.annotated_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if not ok:
        raise RuntimeError("Không mã hóa được ảnh kết quả.")
    return ImageRun(
        name=name, original_rgb=original_rgb, annotated_rgb=det.annotated_rgb,
        detections=rows, annotated_jpeg=encoded.tobytes(), calibration_name=calibration_name,
        inference_ms=det.inference_ms, tiles_processed=det.tiles_processed, device_used=det.device_used,
    )
