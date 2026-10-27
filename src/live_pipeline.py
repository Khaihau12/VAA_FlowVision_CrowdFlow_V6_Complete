from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import math
import tempfile
import uuid
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "model" / "yolo11n.pt"
CALIB_ROOT = ROOT / "assets" / "calibration"

CAMERA_CALIBRATION = {
    "Cam1 / CVLab1": ("intr_CVLab1.xml", "extr_CVLab1.xml"),
    "Cam2 / CVLab2": ("intr_CVLab2.xml", "extr_CVLab2.xml"),
    "Cam3 / CVLab3": ("intr_CVLab3.xml", "extr_CVLab3.xml"),
    "Cam4 / CVLab4": ("intr_CVLab4.xml", "extr_CVLab4.xml"),
    "Cam5 / IDIAP1": ("intr_IDIAP1.xml", "extr_IDIAP1.xml"),
    "Cam6 / IDIAP2": ("intr_IDIAP2.xml", "extr_IDIAP2.xml"),
    "Cam7 / IDIAP3": ("intr_IDIAP3.xml", "extr_IDIAP3.xml"),
}


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


@dataclass(frozen=True)
class TrackingConfig:
    confidence: float = 0.15
    image_size: int = 960
    target_fps: float = 15.0
    start_sec: float = 0.0
    max_seconds: float = 15.0
    device: str = "auto"
    max_det: int = 1000
    iou: float = 0.55
    half: bool = True
    roi: tuple[float, float, float, float] | None = None
    model_path: str | None = None


@dataclass
class CameraRun:
    camera: str
    calibration_name: str | None
    info: VideoInfo
    tracks: pd.DataFrame
    preview_rgb: np.ndarray | None
    video_path: str | None
    processed_frames: int
    elapsed_video_sec: float


@dataclass
class FusionRun:
    observations: pd.DataFrame
    pair_matches: pd.DataFrame
    occupancy: pd.DataFrame
    transitions: pd.DataFrame
    dwell: pd.DataFrame


def probe_video(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    duration = frames / fps if frames > 0 else 0.0
    return VideoInfo(fps=fps, frame_count=frames, width=width, height=height, duration_sec=duration)


def save_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    folder = Path(tempfile.gettempdir()) / "crowdflow_research_uploads"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{uuid.uuid4().hex}{suffix}"
    target.write_bytes(uploaded_file.getvalue())
    return str(target)


def _read_matrix(path: Path, tag: str) -> np.ndarray:
    tree = ET.parse(path)
    root = tree.getroot()
    node = root.find(tag)
    if node is None:
        raise ValueError(f"Không có '{tag}' trong {path.name}")
    data = node.find("data")
    if data is not None:
        rows = int(node.find("rows").text)
        cols = int(node.find("cols").text)
        arr = np.fromstring(data.text.strip(), dtype=np.float64, sep=" ")
        return arr.reshape(rows, cols)
    return np.fromstring((node.text or "").strip(), dtype=np.float64, sep=" ")


def load_calibration(calibration_name: str):
    if calibration_name not in CAMERA_CALIBRATION:
        raise KeyError(f"Bộ hiệu chuẩn không được hỗ trợ: {calibration_name}")
    intr_name, extr_name = CAMERA_CALIBRATION[calibration_name]
    intr_path = CALIB_ROOT / "intrinsic_original" / intr_name
    extr_path = CALIB_ROOT / "extrinsic" / extr_name
    K = _read_matrix(intr_path, "camera_matrix")
    D = _read_matrix(intr_path, "distortion_coefficients").reshape(-1)
    tree = ET.parse(extr_path)
    root = tree.getroot()
    rnode, tnode = root.find("rvec"), root.find("tvec")
    if rnode is None or tnode is None:
        raise ValueError(f"Không có rvec/tvec trong {extr_path.name}")
    rvec = np.fromstring((rnode.text or "").strip(), dtype=np.float64, sep=" ").reshape(3)
    tvec = np.fromstring((tnode.text or "").strip(), dtype=np.float64, sep=" ").reshape(3)
    return K, D, rvec, tvec


def pixel_to_ground(u: float, v: float, K, D, rvec, tvec) -> tuple[float, float]:
    pixel = np.array([[[float(u), float(v)]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pixel, K, D)[0, 0]
    ray_cam = np.array([undistorted[0], undistorted[1], 1.0], dtype=np.float64)
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    camera_center = -R.T @ t
    ray_world = R.T @ ray_cam
    if abs(ray_world[2]) < 1e-12:
        return np.nan, np.nan
    scale = -camera_center[2] / ray_world[2]
    if scale <= 0:
        return np.nan, np.nan
    P = camera_center + scale * ray_world
    return float(P[0]), float(P[1])


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _empty_tracks() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "camera", "sample_index", "frame", "time_sec", "track_id", "confidence",
        "x1", "y1", "x2", "y2", "foot_x", "foot_y", "ground_x", "ground_y"
    ])


def process_camera(
    video_path: str | Path,
    camera: str,
    config: TrackingConfig,
    calibration_name: str | None = None,
    progress: Callable[[float, str], None] | None = None,
    create_annotated_video: bool = True,
) -> CameraRun:
    """YOLO11n + ByteTrack on a video clip.

    V5 single-camera tracking: person class, ByteTrack, ROI support, higher-resolution
    inference defaults, max_det=1000 and optional FP16. Bottom-centre bbox foot point
    is still used for ground projection so the research geometry remains compatible.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "Thiếu thư viện ultralytics. Hãy chạy run_app.bat để cài các thư viện cần thiết cho V4."
        ) from e

    path = str(video_path)
    info = probe_video(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được {camera}")

    start_frame = max(0, int(round(config.start_sec * info.fps)))
    if info.frame_count > 0:
        start_frame = min(start_frame, max(0, info.frame_count - 1))
    max_frames = max(1, int(round(config.max_seconds * info.fps)))
    end_frame = start_frame + max_frames
    if info.frame_count > 0:
        end_frame = min(end_frame, info.frame_count)
    step = max(1, int(round(info.fps / max(0.5, config.target_fps))))
    estimated_samples = max(1, math.ceil(max(0, end_frame - start_frame) / step))

    calib = load_calibration(calibration_name) if calibration_name else None
    model_path = Path(config.model_path) if config.model_path else MODEL_PATH
    model = YOLO(str(model_path))
    device = _resolve_device(config.device)
    try:
        import torch
        use_half = bool(config.half and device != "cpu" and torch.cuda.is_available())
    except Exception:
        use_half = False

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows: list[dict] = []
    frame_id = start_frame
    sample_index = 0
    last_rgb = None

    writer = None
    out_path = None
    if create_annotated_video and info.width > 0 and info.height > 0:
        out_dir = Path(tempfile.gettempdir()) / "crowdflow_research_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{uuid.uuid4().hex}_{camera}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, max(1.0, min(config.target_fps, info.fps)), (info.width, info.height))
        if not writer.isOpened():
            writer = None
            out_path = None

    while frame_id < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame_id - start_frame) % step == 0:
            # V5: ROI keeps tiny people larger relative to the inference image while
            # coordinates are mapped back to the original frame for calibration/export.
            if config.roi is None:
                rx1, ry1, rx2, ry2 = 0, 0, frame.shape[1], frame.shape[0]
            else:
                x1n, y1n, x2n, y2n = config.roi
                rx1 = int(max(0, min(frame.shape[1]-1, round(min(x1n,x2n)*frame.shape[1]))))
                rx2 = int(max(rx1+1, min(frame.shape[1], round(max(x1n,x2n)*frame.shape[1]))))
                ry1 = int(max(0, min(frame.shape[0]-1, round(min(y1n,y2n)*frame.shape[0]))))
                ry2 = int(max(ry1+1, min(frame.shape[0], round(max(y1n,y2n)*frame.shape[0]))))
            infer_frame = frame[ry1:ry2, rx1:rx2]
            result = model.track(
                infer_frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                conf=float(config.confidence), imgsz=int(config.image_size),
                max_det=int(config.max_det), iou=float(config.iou),
                device=device, half=use_half, verbose=False,
            )[0]

            annotated = frame.copy()
            if config.roi is not None:
                cv2.rectangle(annotated, (rx1, ry1), (rx2-1, ry2-1), (255, 160, 0), 2)
            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.cpu().numpy().astype(int)
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for track_id, box, conf in zip(ids, boxes, confs):
                    x1, y1, x2, y2 = map(float, box)
                    x1 += rx1; x2 += rx1; y1 += ry1; y2 += ry1
                    foot_x, foot_y = (x1 + x2) / 2.0, y2
                    gx, gy = (np.nan, np.nan)
                    if calib is not None:
                        gx, gy = pixel_to_ground(foot_x, foot_y, *calib)
                    rows.append({
                        "camera": camera,
                        "sample_index": sample_index,
                        "frame": int(frame_id),
                        "time_sec": float(frame_id / info.fps),
                        "track_id": int(track_id),
                        "confidence": float(conf),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "foot_x": float(foot_x), "foot_y": float(foot_y),
                        "ground_x": float(gx), "ground_y": float(gy),
                    })
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                    cv2.circle(annotated, (int(foot_x), int(foot_y)), 5, (0, 0, 255), -1)
                    cv2.putText(annotated, f"ID {track_id} {conf:.2f}", (int(x1), max(22, int(y1)-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(
                annotated,
                f"{camera} | mau {sample_index} | t={frame_id/info.fps:.2f}s",
                (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
            )
            last_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            if writer is not None:
                writer.write(annotated)
            sample_index += 1
            if progress:
                progress(min(1.0, sample_index / estimated_samples), f"{camera}: {sample_index}/{estimated_samples} mẫu")

        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()

    tracks = pd.DataFrame(rows) if rows else _empty_tracks()
    return CameraRun(
        camera=camera,
        calibration_name=calibration_name,
        info=info,
        tracks=tracks,
        preview_rgb=last_rgb,
        video_path=out_path,
        processed_frames=sample_index,
        elapsed_video_sec=float(max(0, frame_id - start_frame) / info.fps),
    )


class _UnionFind:
    def __init__(self):
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(self, x: tuple[str, int]):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: tuple[str, int], b: tuple[str, int]):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _finite_ground(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = np.isfinite(df["ground_x"].to_numpy(float)) & np.isfinite(df["ground_y"].to_numpy(float))
    return df.loc[mask].copy()


def fuse_pair(cam1: pd.DataFrame, cam2: pd.DataFrame, threshold_cm: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same sampled timestamp + ground distance + Hungarian 1:1 matching.

    Matches are fused by averaging the two ground locations. Unmatched observations are
    preserved. Global IDs are connected components of matched local ByteTrack IDs.
    """
    a_all = _finite_ground(cam1)
    b_all = _finite_ground(cam2)
    uf = _UnionFind()
    match_rows: list[dict] = []
    frame_match_map: dict[tuple[int, str, int], tuple[str, int] | None] = {}

    sample_ids = sorted(set(a_all.get("sample_index", pd.Series(dtype=int)).tolist()) | set(b_all.get("sample_index", pd.Series(dtype=int)).tolist()))
    for sid in sample_ids:
        a = a_all[a_all["sample_index"] == sid].reset_index(drop=True)
        b = b_all[b_all["sample_index"] == sid].reset_index(drop=True)
        if a.empty or b.empty:
            continue
        p1 = a[["ground_x", "ground_y"]].to_numpy(float)
        p2 = b[["ground_x", "ground_y"]].to_numpy(float)
        dist = np.sqrt(np.sum((p1[:, None, :] - p2[None, :, :]) ** 2, axis=2))
        ri, ci = linear_sum_assignment(dist)
        for i, j in zip(ri, ci):
            d = float(dist[i, j])
            if d > threshold_cm:
                continue
            r1, r2 = a.iloc[i], b.iloc[j]
            n1 = ("cam1", int(r1.track_id))
            n2 = ("cam2", int(r2.track_id))
            uf.union(n1, n2)
            frame_match_map[(int(sid), "cam1", int(r1.track_id))] = n2
            frame_match_map[(int(sid), "cam2", int(r2.track_id))] = n1
            match_rows.append({
                "sample_index": int(sid),
                "time_sec": float((float(r1.time_sec) + float(r2.time_sec)) / 2.0),
                "cam1_track_id": int(r1.track_id),
                "cam2_track_id": int(r2.track_id),
                "cam1_x": float(r1.ground_x), "cam1_y": float(r1.ground_y),
                "cam2_x": float(r2.ground_x), "cam2_y": float(r2.ground_y),
                "cross_camera_distance_cm": d,
                "ground_x": float((r1.ground_x + r2.ground_x) / 2.0),
                "ground_y": float((r1.ground_y + r2.ground_y) / 2.0),
            })

    # Register every local ID so even never-matched tracks get a stable component.
    for r in a_all.itertuples():
        uf.find(("cam1", int(r.track_id)))
    for r in b_all.itertuples():
        uf.find(("cam2", int(r.track_id)))
    roots = sorted({uf.find(x) for x in uf.parent}, key=str)
    gid_map = {root: i + 1 for i, root in enumerate(roots)}

    obs: list[dict] = []
    matched_keys = set()
    for m in match_rows:
        sid = int(m["sample_index"])
        id1, id2 = int(m["cam1_track_id"]), int(m["cam2_track_id"])
        matched_keys.add((sid, "cam1", id1))
        matched_keys.add((sid, "cam2", id2))
        obs.append({
            "sample_index": sid, "time_sec": m["time_sec"],
            "ground_x": m["ground_x"], "ground_y": m["ground_y"],
            "source": "cam1+cam2", "cam1_track_id": id1, "cam2_track_id": id2,
            "cross_camera_distance_cm": m["cross_camera_distance_cm"],
            "global_id": gid_map[uf.find(("cam1", id1))],
        })

    for label, frame in [("cam1", a_all), ("cam2", b_all)]:
        for r in frame.itertuples():
            key = (int(r.sample_index), label, int(r.track_id))
            if key in matched_keys:
                continue
            node = (label, int(r.track_id))
            obs.append({
                "sample_index": int(r.sample_index), "time_sec": float(r.time_sec),
                "ground_x": float(r.ground_x), "ground_y": float(r.ground_y),
                "source": label,
                "cam1_track_id": int(r.track_id) if label == "cam1" else np.nan,
                "cam2_track_id": int(r.track_id) if label == "cam2" else np.nan,
                "cross_camera_distance_cm": np.nan,
                "global_id": gid_map[uf.find(node)],
            })

    observations = pd.DataFrame(obs)
    if not observations.empty:
        observations = observations.sort_values(["sample_index", "global_id", "source"]).reset_index(drop=True)
    matches = pd.DataFrame(match_rows)
    return observations, matches


def summarize_fusion(observations: pd.DataFrame, zone_model: dict, sample_interval_sec: float) -> FusionRun:
    from src.zones import assign_zones, ZONE_ORDER

    obs = observations.copy()
    if obs.empty:
        return FusionRun(
            observations=obs,
            pair_matches=pd.DataFrame(),
            occupancy=pd.DataFrame(columns=["sample_index", *ZONE_ORDER, "total"]),
            transitions=pd.DataFrame(columns=["from_zone", "to_zone", "count"]),
            dwell=pd.DataFrame(columns=["global_id", "zone", "samples", "dwell_sec"]),
        )

    obs["zone"] = assign_zones(obs, zone_model)
    occ = obs.groupby(["sample_index", "zone"]).size().unstack(fill_value=0).reindex(columns=ZONE_ORDER, fill_value=0)
    occ["total"] = occ.sum(axis=1)
    occ = occ.reset_index()

    transition_rows: list[dict] = []
    dwell_rows: list[dict] = []
    for gid, g in obs.sort_values(["global_id", "sample_index"]).groupby("global_id"):
        # One zone per global ID per sample. Duplicates should be rare; take first deterministically.
        seq = g.drop_duplicates("sample_index")[["sample_index", "zone"]].sort_values("sample_index")
        if seq.empty:
            continue
        zones = seq["zone"].tolist()
        samples = seq["sample_index"].astype(int).tolist()
        run_zone = zones[0]
        run_len = 1
        for prev_zone, zone, prev_sid, sid in zip(zones[:-1], zones[1:], samples[:-1], samples[1:]):
            if zone == run_zone and sid == prev_sid + 1:
                run_len += 1
            else:
                dwell_rows.append({"global_id": int(gid), "zone": run_zone, "samples": run_len, "dwell_sec": run_len * sample_interval_sec})
                run_zone = zone
                run_len = 1
            if zone != prev_zone:
                transition_rows.append({"global_id": int(gid), "from_zone": prev_zone, "to_zone": zone, "sample_index": int(sid)})
        dwell_rows.append({"global_id": int(gid), "zone": run_zone, "samples": run_len, "dwell_sec": run_len * sample_interval_sec})

    trans = pd.DataFrame(transition_rows)
    if trans.empty:
        trans_summary = pd.DataFrame(columns=["from_zone", "to_zone", "count"])
    else:
        trans_summary = trans.groupby(["from_zone", "to_zone"]).size().rename("count").reset_index().sort_values("count", ascending=False)
    dwell = pd.DataFrame(dwell_rows)
    return FusionRun(observations=obs, pair_matches=pd.DataFrame(), occupancy=occ, transitions=trans_summary, dwell=dwell)
