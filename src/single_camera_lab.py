from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import tempfile
import time
import uuid

import cv2
import numpy as np
import pandas as pd

from src.live_pipeline import MODEL_PATH


@dataclass(frozen=True)
class HardwareInfo:
    torch_available: bool
    cuda_available: bool
    device_name: str
    torch_version: str


@dataclass(frozen=True)
class DetectorConfig:
    confidence: float = 0.15
    image_size: int = 1280
    max_det: int = 1000
    iou: float = 0.55
    device: str = "auto"
    half: bool = True
    class_id: int = 0
    label: str = "Người"
    tiled: bool = False
    tile_size: int = 768
    tile_overlap: float = 0.20
    roi: tuple[float, float, float, float] | None = None  # normalized x1,y1,x2,y2 in [0,1]


@dataclass
class DetectionRun:
    annotated_rgb: np.ndarray
    detections: pd.DataFrame
    inference_ms: float
    tiles_processed: int
    roi_pixels: tuple[int, int, int, int]
    device_used: str


DETECTION_COLUMNS = [
    "detection_id", "class_id", "label", "confidence",
    "x1", "y1", "x2", "y2", "width", "height", "area",
    "foot_x", "foot_y", "tile_id",
]


def inspect_hardware() -> HardwareInfo:
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if cuda else "CPU"
        return HardwareInfo(True, cuda, str(name), str(torch.__version__))
    except Exception:
        return HardwareInfo(False, False, "CPU", "không xác định")


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "0" if inspect_hardware().cuda_available else "cpu"


def resolve_half(requested_half: bool, device: str) -> bool:
    return bool(requested_half and device != "cpu" and inspect_hardware().cuda_available)


def load_yolo(model_path: str | Path | None = None):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("Thiếu ultralytics. Chạy: python -m pip install -r requirements.txt") from e
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {path}")
    return YOLO(str(path))


def save_model_upload(data: bytes, original_name: str) -> str:
    folder = Path(tempfile.gettempdir()) / "crowdflow_v5_models"
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_name).suffix or ".pt"
    target = folder / f"{uuid.uuid4().hex}{suffix}"
    target.write_bytes(data)
    return str(target)


def normalized_roi_to_pixels(
    shape: tuple[int, ...], roi: tuple[float, float, float, float] | None
) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    if roi is None:
        return 0, 0, w, h
    x1n, y1n, x2n, y2n = [float(v) for v in roi]
    x1n, x2n = sorted((max(0.0, min(1.0, x1n)), max(0.0, min(1.0, x2n))))
    y1n, y2n = sorted((max(0.0, min(1.0, y1n)), max(0.0, min(1.0, y2n))))
    x1 = int(round(x1n * w)); x2 = int(round(x2n * w))
    y1 = int(round(y1n * h)); y2 = int(round(y2n * h))
    x1 = max(0, min(w - 1, x1)); y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2)); y2 = max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


def draw_roi_preview(frame_bgr: np.ndarray, roi: tuple[float, float, float, float] | None) -> np.ndarray:
    out = frame_bgr.copy()
    x1, y1, x2, y2 = normalized_roi_to_pixels(frame_bgr.shape, roi)
    cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 3)
    cv2.putText(out, "ROI - vung AI xu ly", (x1 + 8, min(y2 - 8, y1 + 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def generate_tiles(width: int, height: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    tile = max(128, int(tile_size))
    overlap = max(0.0, min(0.8, float(overlap)))
    step = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, max(1, width - tile + 1), step))
    ys = list(range(0, max(1, height - tile + 1), step))
    last_x = max(0, width - tile)
    last_y = max(0, height - tile)
    if not xs or xs[-1] != last_x:
        xs.append(last_x)
    if not ys or ys[-1] != last_y:
        ys.append(last_y)
    boxes: list[tuple[int, int, int, int]] = []
    seen = set()
    for y in ys:
        for x in xs:
            b = (int(x), int(y), int(min(width, x + tile)), int(min(height, y + tile)))
            if b not in seen:
                seen.add(b); boxes.append(b)
    return boxes


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(box[0], boxes[:, 0]); yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2]); yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    a = np.maximum(0.0, box[2]-box[0]) * np.maximum(0.0, box[3]-box[1])
    b = np.maximum(0.0, boxes[:,2]-boxes[:,0]) * np.maximum(0.0, boxes[:,3]-boxes[:,1])
    return inter / np.maximum(1e-9, a + b - inter)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.55) -> list[int]:
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0]); keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _iou_one_to_many(boxes[i], boxes[rest])
        order = rest[ious <= float(iou_threshold)]
    return keep


def _predict_one(model: Any, frame: np.ndarray, cfg: DetectorConfig, device: str, half: bool):
    return model.predict(
        frame,
        classes=[int(cfg.class_id)],
        conf=float(cfg.confidence),
        imgsz=int(cfg.image_size),
        max_det=int(cfg.max_det),
        iou=float(cfg.iou),
        device=device,
        half=half,
        verbose=False,
    )[0]


def _extract_result(result: Any, offset_x: int, offset_y: int, tile_id: int, cfg: DetectorConfig) -> list[dict]:
    rows: list[dict] = []
    if result.boxes is None or len(result.boxes) == 0:
        return rows
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes.cls is not None else np.full(len(boxes), cfg.class_id)
    for box, conf, cls_id in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(float, box)
        x1 += offset_x; x2 += offset_x; y1 += offset_y; y2 += offset_y
        w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
        rows.append({
            "class_id": int(cls_id), "label": cfg.label, "confidence": float(conf),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": w, "height": h, "area": w*h,
            "foot_x": (x1+x2)/2.0, "foot_y": y2, "tile_id": int(tile_id),
        })
    return rows


def detect_frame(frame_bgr: np.ndarray, cfg: DetectorConfig, model: Any | None = None) -> DetectionRun:
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Khung hình rỗng.")
    model = model or load_yolo()
    device = resolve_device(cfg.device)
    half = resolve_half(cfg.half, device)
    rx1, ry1, rx2, ry2 = normalized_roi_to_pixels(frame_bgr.shape, cfg.roi)
    roi_img = frame_bgr[ry1:ry2, rx1:rx2]
    if roi_img.size == 0:
        raise ValueError("ROI không hợp lệ.")

    all_rows: list[dict] = []
    t0 = time.perf_counter()
    tile_count = 1
    if cfg.tiled:
        tiles = generate_tiles(roi_img.shape[1], roi_img.shape[0], cfg.tile_size, cfg.tile_overlap)
        tile_count = len(tiles)
        for tid, (tx1, ty1, tx2, ty2) in enumerate(tiles):
            tile = roi_img[ty1:ty2, tx1:tx2]
            result = _predict_one(model, tile, cfg, device, half)
            all_rows.extend(_extract_result(result, rx1+tx1, ry1+ty1, tid, cfg))
        if all_rows:
            boxes = np.asarray([[r["x1"],r["y1"],r["x2"],r["y2"]] for r in all_rows], dtype=float)
            scores = np.asarray([r["confidence"] for r in all_rows], dtype=float)
            keep = nms_numpy(boxes, scores, cfg.iou)
            all_rows = [all_rows[i] for i in keep[: int(cfg.max_det)]]
    else:
        result = _predict_one(model, roi_img, cfg, device, half)
        all_rows = _extract_result(result, rx1, ry1, 0, cfg)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    all_rows.sort(key=lambda r: r["confidence"], reverse=True)
    for i, row in enumerate(all_rows, start=1):
        row["detection_id"] = i
    df = pd.DataFrame(all_rows, columns=DETECTION_COLUMNS) if all_rows else pd.DataFrame(columns=DETECTION_COLUMNS)

    out = frame_bgr.copy()
    if cfg.roi is not None:
        cv2.rectangle(out, (rx1, ry1), (rx2-1, ry2-1), (255, 160, 0), 2)
    for r in all_rows:
        x1,y1,x2,y2 = int(r["x1"]),int(r["y1"]),int(r["x2"]),int(r["y2"])
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,255), 2)
        cv2.circle(out, (int(r["foot_x"]), int(r["foot_y"])), 3, (0,0,255), -1)
        cv2.putText(out, f"{cfg.label} {r['detection_id']} {r['confidence']:.2f}", (x1, max(18,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0,0), (min(out.shape[1], 760), 46), (20,20,20), -1)
    mode = "TILED" if cfg.tiled else "TOAN/ROI"
    cv2.putText(out, f"{mode} | phat hien={len(df)} | {inference_ms:.0f}ms | imgsz={cfg.image_size} | conf={cfg.confidence:.2f}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2, cv2.LINE_AA)
    return DetectionRun(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), df, inference_ms, tile_count, (rx1,ry1,rx2,ry2), device)


def decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không đọc được ảnh.")
    return frame


def threshold_sweep(frame_bgr: np.ndarray, base_cfg: DetectorConfig, thresholds: Iterable[float], model: Any | None = None) -> pd.DataFrame:
    model = model or load_yolo()
    rows = []
    for conf in thresholds:
        cfg = DetectorConfig(**{**base_cfg.__dict__, "confidence": float(conf)})
        run = detect_frame(frame_bgr, cfg, model=model)
        rows.append({"confidence": float(conf), "count": len(run.detections), "inference_ms": run.inference_ms, "tiles": run.tiles_processed})
    return pd.DataFrame(rows)


def resolution_sweep(frame_bgr: np.ndarray, base_cfg: DetectorConfig, sizes: Iterable[int], model: Any | None = None) -> pd.DataFrame:
    model = model or load_yolo()
    rows = []
    for size in sizes:
        cfg = DetectorConfig(**{**base_cfg.__dict__, "image_size": int(size), "tiled": False})
        run = detect_frame(frame_bgr, cfg, model=model)
        rows.append({"image_size": int(size), "count": len(run.detections), "inference_ms": run.inference_ms})
    return pd.DataFrame(rows)


def count_error_metrics(predicted: int, ground_truth: int) -> dict[str, float]:
    gt = int(ground_truth); pred = int(predicted)
    abs_err = abs(pred-gt)
    pct = 100.0 * abs_err / gt if gt > 0 else (0.0 if pred == 0 else float("inf"))
    return {"predicted": pred, "ground_truth": gt, "absolute_error": abs_err, "absolute_error_pct": pct}
