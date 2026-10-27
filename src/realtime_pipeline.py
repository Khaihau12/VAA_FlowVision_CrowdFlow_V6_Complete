from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import pandas as pd

from src.live_pipeline import MODEL_PATH, load_calibration, pixel_to_ground
from src.single_camera_lab import normalized_roi_to_pixels, resolve_device, resolve_half


@dataclass(frozen=True)
class RealtimeConfig:
    confidence: float = 0.15
    image_size: int = 960
    target_fps: float = 15.0
    device: str = "auto"
    reconnect_delay_sec: float = 1.0
    max_history_rows: int = 20000
    max_det: int = 1000
    iou: float = 0.55
    half: bool = True
    roi: tuple[float, float, float, float] | None = None
    model_path: str | None = None


@dataclass
class LivePacket:
    frame_rgb: np.ndarray | None
    rows: pd.DataFrame
    capture_time: float | None
    inference_ms: float | None
    frame_seq: int
    connection_generation: int
    reconnect_count: int
    status: str
    error: str | None


def sanitize_source(source: str | int) -> str:
    if isinstance(source, int):
        return f"Webcam #{source}"
    text = str(source)
    try:
        parts = urlsplit(text)
        if not parts.scheme:
            return text
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        user = parts.username
        netloc = f"{user}:***@{host}{port}" if user else f"{host}{port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))
    except Exception:
        return "Luồng camera"


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "camera", "sample_index", "frame", "time_sec", "track_id", "confidence",
        "x1", "y1", "x2", "y2", "foot_x", "foot_y", "ground_x", "ground_y",
    ])


def _fps_from_times(times: deque[float]) -> float:
    if len(times) < 2:
        return 0.0
    dt = times[-1] - times[0]
    return (len(times) - 1) / dt if dt > 1e-9 else 0.0


class RealtimeStream:
    """RTSP/webcam realtime with independent capture and AI workers.

    V5 fixes the V4 bottleneck: camera capture and YOLO inference no longer depend on
    Streamlit's refresh interval. The capture worker always keeps only the newest frame;
    the AI worker runs in the background at target_fps and therefore does not accumulate
    stale RTSP frames. The UI only polls the latest annotated packet.
    """

    def __init__(self, source: str | int, camera_name: str, config: RealtimeConfig, calibration_name: str | None = None):
        self.source = source
        self.camera_name = camera_name
        self.config = config
        self.calibration_name = calibration_name
        self.source_display = sanitize_source(source)
        self._calibration = load_calibration(calibration_name) if calibration_name else None

        self._capture_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._infer_thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

        self._latest_bgr: np.ndarray | None = None
        self._latest_capture_time: float | None = None
        self._frame_seq = 0
        self._status = "STOPPED"
        self._error: str | None = None
        self._reconnect_count = 0
        self._connection_generation = 0
        self._stream_fps = 0.0
        self._width = 0
        self._height = 0
        self._capture_times: deque[float] = deque(maxlen=90)
        self._ai_times: deque[float] = deque(maxlen=90)

        self._model: Any | None = None
        self._model_generation = -1
        self._last_inferred_seq = -1
        self._sample_index = 0
        self._last_packet = LivePacket(None, _empty_rows(), None, None, 0, 0, 0, "STOPPED", None)
        self._history = _empty_rows()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    @property
    def status(self) -> dict[str, Any]:
        with self._capture_lock:
            return {
                "running": self.running,
                "status": self._status,
                "error": self._error,
                "reconnect_count": self._reconnect_count,
                "connection_generation": self._connection_generation,
                "frame_seq": self._frame_seq,
                "stream_fps": self._stream_fps,
                "measured_capture_fps": _fps_from_times(self._capture_times),
                "measured_ai_fps": _fps_from_times(self._ai_times),
                "width": self._width,
                "height": self._height,
                "source": self.source_display,
                "camera": self.camera_name,
                "calibration": self.calibration_name,
            }

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name=f"crowdflow-capture-{self.camera_name}", daemon=True)
        self._infer_thread = threading.Thread(target=self._inference_loop, name=f"crowdflow-ai-{self.camera_name}", daemon=True)
        self._thread.start()
        self._infer_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        cap = None
        with self._capture_lock:
            cap = self._cap
            self._cap = None
            self._status = "STOPPED"
        if cap is not None:
            try: cap.release()
            except Exception: pass
        for th in [self._thread, self._infer_thread]:
            if th is not None and th.is_alive():
                th.join(timeout=1.5)

    def _open_capture(self) -> cv2.VideoCapture:
        src = self.source
        if isinstance(src, str) and src.lower().startswith("rtsp://"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release(); cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        for prop_name, value in [("CAP_PROP_OPEN_TIMEOUT_MSEC", 5000), ("CAP_PROP_READ_TIMEOUT_MSEC", 5000)]:
            prop = getattr(cv2, prop_name, None)
            if prop is not None:
                try: cap.set(prop, value)
                except Exception: pass
        return cap

    def _capture_loop(self) -> None:
        first_connection = True
        while not self._stop_event.is_set():
            with self._capture_lock:
                self._status = "CONNECTING"; self._error = None
            cap = self._open_capture()
            if not cap.isOpened():
                with self._capture_lock:
                    self._status = "RECONNECTING"
                    self._error = "Không mở được nguồn camera. Kiểm tra URL/IP/tài khoản/mạng."
                    self._reconnect_count += 1
                cap.release(); self._stop_event.wait(max(0.2, self.config.reconnect_delay_sec)); continue

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            with self._capture_lock:
                self._cap = cap; self._status = "LIVE"; self._stream_fps = fps if np.isfinite(fps) else 0.0
                self._width = width; self._height = height; self._connection_generation += 1
                if not first_connection: self._reconnect_count += 1
            first_connection = False
            failed_reads = 0
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failed_reads += 1
                    if failed_reads >= 3: break
                    time.sleep(0.02); continue
                failed_reads = 0; now = time.time()
                with self._capture_lock:
                    self._latest_bgr = frame; self._latest_capture_time = now; self._frame_seq += 1
                    self._height, self._width = frame.shape[:2]; self._status = "LIVE"; self._error = None
                    self._capture_times.append(now)
            try: cap.release()
            except Exception: pass
            with self._capture_lock:
                self._cap = None
                if not self._stop_event.is_set():
                    self._status = "RECONNECTING"; self._error = "Mất frame từ camera; đang tự kết nối lại."
            if not self._stop_event.is_set(): self._stop_event.wait(max(0.2, self.config.reconnect_delay_sec))
        with self._capture_lock: self._status = "STOPPED"

    def _ensure_model(self, generation: int):
        if self._model is not None and self._model_generation == generation:
            return
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("Thiếu ultralytics. Chạy: python -m pip install -r requirements.txt") from e
        path = Path(self.config.model_path) if self.config.model_path else MODEL_PATH
        self._model = YOLO(str(path)); self._model_generation = generation; self._last_inferred_seq = -1

    def _inference_loop(self) -> None:
        # AI worker is independent of the Streamlit fragment refresh rate.
        while not self._stop_event.is_set():
            target = max(0.5, float(self.config.target_fps))
            t0 = time.perf_counter()
            try:
                self._infer_latest()
            except Exception as e:
                with self._capture_lock:
                    self._error = f"Lỗi AI: {e}"
            elapsed = time.perf_counter() - t0
            self._stop_event.wait(max(0.001, 1.0 / target - elapsed))

    def _infer_latest(self) -> LivePacket:
        with self._capture_lock:
            frame = None if self._latest_bgr is None else self._latest_bgr.copy()
            capture_time = self._latest_capture_time; frame_seq = self._frame_seq
            generation = self._connection_generation; reconnects = self._reconnect_count
            status = self._status; error = self._error
        if frame is None or frame_seq == self._last_inferred_seq:
            return self._last_packet

        with self._infer_lock:
            # Newest frame may have changed while waiting; stale frames are intentionally skipped.
            with self._capture_lock:
                if self._latest_bgr is not None and self._frame_seq != frame_seq:
                    frame = self._latest_bgr.copy(); capture_time = self._latest_capture_time; frame_seq = self._frame_seq
                    generation = self._connection_generation; reconnects = self._reconnect_count; status = self._status; error = self._error
            if frame_seq == self._last_inferred_seq:
                return self._last_packet
            self._ensure_model(generation)
            device = resolve_device(self.config.device); half = resolve_half(self.config.half, device)
            rx1, ry1, rx2, ry2 = normalized_roi_to_pixels(frame.shape, self.config.roi)
            roi_img = frame[ry1:ry2, rx1:rx2]
            t0 = time.perf_counter()
            result = self._model.track(
                roi_img, persist=True, tracker="bytetrack.yaml", classes=[0],
                conf=float(self.config.confidence), imgsz=int(self.config.image_size),
                max_det=int(self.config.max_det), iou=float(self.config.iou),
                device=device, half=half, verbose=False,
            )[0]
            infer_ms = (time.perf_counter() - t0) * 1000.0

            annotated = frame.copy(); rows: list[dict[str, Any]] = []
            if self.config.roi is not None:
                cv2.rectangle(annotated, (rx1,ry1), (rx2-1,ry2-1), (255,160,0), 2)
            elapsed = max(0.0, time.time() - (capture_time or time.time()))
            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.detach().cpu().numpy().astype(int)
                boxes = result.boxes.xyxy.detach().cpu().numpy(); confs = result.boxes.conf.detach().cpu().numpy()
                for track_id, box, conf in zip(ids, boxes, confs):
                    x1,y1,x2,y2 = map(float, box); x1 += rx1; x2 += rx1; y1 += ry1; y2 += ry1
                    foot_x, foot_y = (x1+x2)/2.0, y2; gx,gy = np.nan,np.nan
                    if self._calibration is not None: gx,gy = pixel_to_ground(foot_x, foot_y, *self._calibration)
                    rows.append({
                        "camera": self.camera_name, "sample_index": int(self._sample_index), "frame": int(frame_seq),
                        "time_sec": float(capture_time or time.time()), "track_id": int(track_id), "confidence": float(conf),
                        "x1":x1,"y1":y1,"x2":x2,"y2":y2,"foot_x":float(foot_x),"foot_y":float(foot_y),
                        "ground_x":float(gx),"ground_y":float(gy),
                    })
                    cv2.rectangle(annotated, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,255), 2)
                    cv2.circle(annotated, (int(foot_x),int(foot_y)), 3, (0,0,255), -1)
                    cv2.putText(annotated, f"ID {track_id} {conf:.2f}", (int(x1),max(18,int(y1)-5)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,255,255),1,cv2.LINE_AA)
            now = time.time(); self._ai_times.append(now)
            camfps = _fps_from_times(self._capture_times); aifps = _fps_from_times(self._ai_times)
            cv2.rectangle(annotated,(0,0),(min(annotated.shape[1],900),48),(20,20,20),-1)
            cv2.putText(annotated, f"{self.camera_name} | nguoi={len(rows)} | Cam={camfps:.1f} FPS | AI={aifps:.1f} FPS | {infer_ms:.0f}ms | tre={elapsed*1000:.0f}ms",
                        (10,31),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,255),2,cv2.LINE_AA)
            df = pd.DataFrame(rows) if rows else _empty_rows()
            if not df.empty:
                self._history = pd.concat([self._history,df],ignore_index=True)
                if len(self._history)>self.config.max_history_rows:
                    self._history=self._history.iloc[-self.config.max_history_rows:].reset_index(drop=True)
            self._last_inferred_seq=frame_seq; self._sample_index += 1
            self._last_packet=LivePacket(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB),df,capture_time,infer_ms,frame_seq,generation,reconnects,status,error)
            return self._last_packet

    def analyze_latest(self, force: bool = False) -> LivePacket:
        if force:
            return self._infer_latest()
        with self._infer_lock:
            packet = self._last_packet
            # Return the latest immutable-ish references; dataframes are not mutated after packet creation.
            return packet

    def history(self) -> pd.DataFrame:
        with self._infer_lock:
            return self._history.copy()

    def snapshot_jpeg(self) -> bytes | None:
        with self._infer_lock:
            packet = self._last_packet
            if packet.frame_rgb is None: return None
            bgr=cv2.cvtColor(packet.frame_rgb,cv2.COLOR_RGB2BGR)
            ok,enc=cv2.imencode(".jpg",bgr,[int(cv2.IMWRITE_JPEG_QUALITY),92])
            return enc.tobytes() if ok else None
