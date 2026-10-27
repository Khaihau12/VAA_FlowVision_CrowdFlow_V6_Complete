from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import load_all
from src.zones import ZONE_ORDER, assign_zones, fit_pca_zones, occupancy_table
from src.decision_engine import evaluate_zone
from src.live_pipeline import (
    TrackingConfig,
    process_camera,
    save_upload,
    fuse_pair,
    summarize_fusion,
    CAMERA_CALIBRATION,
)
from src.realtime_pipeline import RealtimeConfig, RealtimeStream
from src.image_pipeline import ImageConfig, load_model as load_image_model, process_image_bytes
from src.single_camera_lab import (
    DetectorConfig, detect_frame, decode_image_bytes, draw_roi_preview,
    threshold_sweep, resolution_sweep, count_error_metrics, inspect_hardware,
    load_yolo, save_model_upload,
)
from src.history_db import (
    DEFAULT_DB_PATH, init_db, log_zone_snapshot, log_total_snapshot,
    read_metrics, list_sources, db_stats, log_alerts, read_alerts,
    log_anomaly_alerts, read_anomaly_alerts,
    clear_database, seed_demo_history,
)
from src.forecast_engine import forecast_all, backtest_forecasts
from src.anomaly_engine import (
    detect_trajectory_anomalies, detect_crowd_anomalies, build_incident_table
)

st.set_page_config(
    page_title="VAA FlowVision CrowdFlow V6",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetric"] {background: rgba(255,255,255,0.04); border: 1px solid rgba(128,128,128,0.22); padding: 12px 14px; border-radius: 14px;}
.small-note {font-size: .86rem; opacity: .78;}
.research-card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 14px 16px; margin-bottom: 10px;}
.status-normal {border-left: 5px solid #2ca02c;}
.status-busy {border-left: 5px solid #ffb000;}
.status-congested {border-left: 5px solid #d62728;}
.v5-banner {border: 1px solid rgba(0,180,216,.4); background: rgba(0,180,216,.08); border-radius: 14px; padding: 12px 16px; margin-bottom: 14px;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def data_bundle():
    return load_all()


@st.cache_resource(show_spinner=False)
def cached_detector_model(model_path: str = ""):
    return load_yolo(model_path or None)


@st.cache_data(show_spinner=False)
def prepared_pair12():
    d = load_all()
    model = fit_pca_zones(d["gt"])
    pred_test = d["fusion"][d["fusion"]["split"] == "test"].copy()
    gt_test = d["gt"][d["gt"]["split"] == "test"].copy()
    pred_test["zone"] = assign_zones(pred_test, model)
    gt_test["zone"] = assign_zones(gt_test, model)
    pred_occ = occupancy_table(pred_test, model)
    gt_occ = occupancy_table(gt_test, model)
    return d, model, pred_test, gt_test, pred_occ, gt_occ


D, ZMODEL, PRED_TEST, GT_TEST, PRED_OCC, GT_OCC = prepared_pair12()
init_db()

st.sidebar.title("VAA FlowVision CrowdFlow V6")
st.sidebar.caption("Detection • Tracking • Fusion • Flow • Forecast • Anomaly • Decision Support")
page = st.sidebar.radio(
    "Màn hình",
    ["Tổng quan", "Phòng thí nghiệm 1 camera", "CCTV / RTSP thời gian thực", "Tải ảnh lên", "Phân tích video", "Bản đồ mặt bằng", "Phân tích khu vực", "Dự báo & cảnh báo sớm", "Điều hướng", "Bất thường & cảnh báo", "Lịch sử & cơ sở dữ liệu", "Đánh giá khoa học", "Dữ liệu"],
)

_hw = inspect_hardware()
st.sidebar.divider()
st.sidebar.subheader("Phần cứng AI")
if _hw.cuda_available:
    st.sidebar.success(f"GPU: {_hw.device_name}")
else:
    st.sidebar.warning("CUDA chưa hoạt động — YOLO đang dùng CPU nếu chọn Tự động")
st.sidebar.caption(f"PyTorch: {_hw.torch_version}")
st.sidebar.divider()
st.sidebar.subheader("Chính sách hỗ trợ quyết định")
busy = st.sidebar.slider("Ngưỡng đông", 0.40, 0.80, 0.65, 0.05)
congested = st.sidebar.slider("Ngưỡng ùn tắc", max(0.65, busy + 0.05), 1.20, 0.85, 0.05)
cap_a = st.sidebar.number_input("Sức chứa Khu A", 1, 200, 10)
cap_b = st.sidebar.number_input("Sức chứa Khu B", 1, 200, 12)
cap_c = st.sidebar.number_input("Sức chứa Khu C", 1, 200, 4)
CAP = {"Zone A": int(cap_a), "Zone B": int(cap_b), "Zone C": int(cap_c)}
st.sidebar.caption("Sức chứa và các ngưỡng là chính sách minh họa có thể điều chỉnh; chưa phải ngưỡng an toàn sân bay đã được xác thực.")

TEST_IDS = sorted(PRED_TEST["gt_index"].unique().tolist())


def _frame_controls(key: str = "frame") -> int:
    idx = st.slider("Mốc dữ liệu TEST", int(min(TEST_IDS)), int(max(TEST_IDS)), int(min(TEST_IDS)), key=key)
    st.caption(f"Mốc TEST {idx} • thời gian tham chiếu ≈ {idx * 0.5:.1f} giây từ đầu chuỗi WILDTRACK")
    return idx


def _current_counts(idx: int) -> dict[str, int]:
    row = PRED_OCC.reindex(TEST_IDS, fill_value=0).loc[idx]
    return {z: int(row.get(z, 0)) for z in ZONE_ORDER}


def _decisions(counts: dict[str, int]):
    return [evaluate_zone(z, counts[z], CAP, counts, busy, congested) for z in ZONE_ORDER]


STATUS_VI = {"NORMAL": "BÌNH THƯỜNG", "BUSY": "ĐÔNG", "CONGESTED": "ÙN TẮC"}
ACTION_VI = {"KEEP": "GIỮ NGUYÊN", "RESTRICT": "HẠN CHẾ", "REDIRECT": "ĐIỀU HƯỚNG LẠI"}
ZONE_VI = {"Zone A": "Khu A", "Zone B": "Khu B", "Zone C": "Khu C"}
CONNECTION_VI = {"STOPPED": "ĐÃ DỪNG", "CONNECTING": "ĐANG KẾT NỐI", "RECONNECTING": "ĐANG KẾT NỐI LẠI", "LIVE": "ĐANG TRỰC TIẾP"}
CALIB_VI = {k: k.replace("Cam1", "Camera 1").replace("Cam2", "Camera 2").replace("Cam3", "Camera 3").replace("Cam4", "Camera 4").replace("Cam5", "Camera 5").replace("Cam6", "Camera 6").replace("Cam7", "Camera 7") for k in CAMERA_CALIBRATION}

COT_VI = {
    "pair": "Cặp camera", "zone": "Khu vực", "timestamps": "Số mốc thời gian",
    "sample_index": "Chỉ số mẫu", "frame": "Khung hình", "time_sec": "Thời gian (giây)",
    "track_id": "ID bám vết", "confidence": "Độ tin cậy", "foot_x": "Điểm chân X", "foot_y": "Điểm chân Y",
    "ground_x": "Tọa độ X mặt bằng", "ground_y": "Tọa độ Y mặt bằng", "source": "Nguồn",
    "global_id": "ID toàn cục", "cam1_track_id": "ID Camera 1", "cam2_track_id": "ID Camera 2",
    "cross_camera_distance_cm": "Khoảng cách ghép (cm)", "from_zone": "Từ khu", "to_zone": "Đến khu",
    "count": "Số lượng", "mean": "Trung bình", "max": "Lớn nhất", "dwell_sec": "Thời gian lưu lại (giây)",
    "occupancy": "Số người hiện diện", "capacity": "Sức chứa", "utilization": "Mức sử dụng",
    "GT_mean_occupancy": "GT trung bình số người", "Pred_mean_occupancy": "Dự đoán trung bình số người",
    "Bias": "Độ lệch", "detection_precision": "Độ chính xác phát hiện", "detection_recall": "Độ bao phủ phát hiện",
    "detection_f1": "F1 phát hiện", "matched_ground_MAE_cm": "MAE mặt bằng của cặp ghép (cm)",
    "ID_switches": "Số lần đổi ID", "predicted_trajectories": "Số quỹ đạo dự đoán",
    "GT_transition_events": "Sự kiện chuyển vùng GT", "Pred_transition_events": "Sự kiện chuyển vùng dự đoán",
    "Transition_F1": "F1 chuyển vùng", "Transition_Count_WAPE": "WAPE số lượng chuyển vùng",
    "Transition_Distribution_TV": "TV phân bố chuyển vùng", "Temporal_Transition_Count_WAPE": "WAPE chuyển vùng theo thời gian",
    "detection_id": "ID phát hiện", "x1": "X1", "y1": "Y1", "x2": "X2", "y2": "Y2",
    "Baseline_Occupancy_MAE": "MAE số người cơ sở", "Baseline_Ground_F1": "F1 mặt bằng cơ sở",
    "Baseline_Ground_Recall": "Recall mặt bằng cơ sở", "Baseline_Ground_MAE_cm": "MAE mặt bằng cơ sở (cm)",
    "Joint_Failure_Lift": "Mức tăng lỗi kết hợp", "Tracking_IDF1": "IDF1 bám vết", "Tracking_MOTA": "MOTA bám vết",
    "Tracking_ID_Switches": "Số lần đổi ID", "EntryExit_F1": "F1 Vào/Ra", "Dwell_Visit_F1": "F1 lượt lưu lại",
    "Dwell_MAE_seconds": "MAE thời gian lưu lại (giây)", "Dwell_Matched_Visits": "Lượt lưu lại ghép khớp",
    "Flow_Event_F1": "F1 sự kiện luồng", "Flow_Count_WAPE": "WAPE số lượng luồng",
    "Flow_Distribution_TV": "TV phân bố luồng", "Temporal_Flow_WAPE": "WAPE luồng theo thời gian",
    "GT_events": "Sự kiện GT", "Pred_events": "Sự kiện dự đoán", "Precision": "Độ chính xác", "Recall": "Độ bao phủ",
    "Timing_MAE_seconds": "MAE thời điểm (giây)", "GT_complete_visits": "Lượt lưu lại hoàn chỉnh GT",
    "Pred_complete_visits": "Lượt lưu lại hoàn chỉnh dự đoán", "Matched_complete_visits": "Lượt hoàn chỉnh ghép khớp",
    "FP_visits": "Lượt dương tính giả", "FN_visits": "Lượt âm tính giả", "Visit_Precision": "Độ chính xác lượt lưu lại",
    "Visit_Recall": "Độ bao phủ lượt lưu lại", "Visit_F1": "F1 lượt lưu lại", "Dwell_RMSE_seconds": "RMSE thời gian lưu lại (giây)",
    "Dwell_Median_AE_seconds": "Trung vị sai số tuyệt đối lưu lại (giây)", "Dwell_Bias_seconds": "Độ lệch thời gian lưu lại (giây)",
    "Dwell_WAPE": "WAPE thời gian lưu lại", "Entry_Timing_MAE_seconds": "MAE thời điểm vào (giây)",
    "Exit_Timing_MAE_seconds": "MAE thời điểm ra (giây)", "Matched_GT_mean_dwell_seconds": "Thời gian lưu GT trung bình đã ghép (giây)",
    "Matched_Pred_mean_dwell_seconds": "Thời gian lưu dự đoán trung bình đã ghép (giây)", "Dwell_Pearson_r": "Tương quan Pearson thời gian lưu",
    "Transition_Precision": "Độ chính xác chuyển vùng", "Transition_Recall": "Độ bao phủ chuyển vùng",
    "Timing_Bias_seconds": "Độ lệch thời điểm (giây)", "GT_total_flow": "Tổng luồng GT", "Pred_total_flow": "Tổng luồng dự đoán",
    "Total_flow_error": "Sai số tổng luồng", "Total_flow_abs_error_pct": "Sai số tuyệt đối tổng luồng (%)",
    "Transition_Count_MAE": "MAE số lượng chuyển vùng", "Transition_Count_RMSE": "RMSE số lượng chuyển vùng",
    "Max_abs_transition_count_error": "Sai số tuyệt đối chuyển vùng lớn nhất", "temporal_bin_seconds": "Độ rộng ô thời gian (giây)",
    "number_bins": "Số ô thời gian", "directed_bin_cells": "Số ô hướng",
    "Temporal_Transition_Count_MAE": "MAE chuyển vùng theo thời gian", "Temporal_Transition_Count_RMSE": "RMSE chuyển vùng theo thời gian",
    "Total_Flow_per_Bin_MAE": "MAE tổng luồng mỗi ô thời gian", "TP": "Dương tính đúng", "FP": "Dương tính giả", "FN": "Âm tính giả",
    "recorded_at_iso": "Thời điểm", "sample_key": "Khóa mẫu", "horizon_min": "Tầm dự báo (phút)",
    "current": "Hiện tại", "forecast": "Dự báo", "lower": "Cận dưới tham khảo", "upper": "Cận trên tham khảo",
    "predicted_utilization": "Mức sử dụng dự báo", "slope_people_per_min": "Xu hướng (người/phút)",
    "reliability": "Mức tin cậy", "alternative": "Khu thay thế", "message": "Khuyến nghị",
}

def viet_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "zone" in out.columns:
        out["zone"] = out["zone"].map(lambda x: ZONE_VI.get(x, x))
    for c in ["from_zone", "to_zone"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: ZONE_VI.get(x, x))
    return out.rename(columns={c: COT_VI.get(c, c) for c in out.columns})


def _status_badge(dec):
    css = {"NORMAL": "status-normal", "BUSY": "status-busy", "CONGESTED": "status-congested"}[dec.status]
    alt = f" → {ZONE_VI.get(dec.alternative, dec.alternative)}" if dec.alternative else ""
    st.markdown(
        f"<div class='research-card {css}'><b>{ZONE_VI.get(dec.zone, dec.zone)}</b><br>"
        f"Số người: <b>{dec.occupancy}</b> / {dec.capacity} &nbsp;•&nbsp; Mức sử dụng: <b>{dec.utilization:.0%}</b><br>"
        f"Trạng thái: <b>{STATUS_VI.get(dec.status, dec.status)}</b> &nbsp;•&nbsp; Hành động: <b>{ACTION_VI.get(dec.action, dec.action)}{alt}</b><br>"
        f"<span class='small-note'>{dec.explanation.replace('Zone A', 'Khu A').replace('Zone B', 'Khu B').replace('Zone C', 'Khu C')}</span></div>",
        unsafe_allow_html=True,
    )


def ground_figure(frame: pd.DataFrame, title: str, show_ids: bool = False):
    view = frame.copy()
    view["khu_vuc_hien_thi"] = view["zone"].map(lambda x: ZONE_VI.get(x, x))
    hover = [c for c in ["source", "global_id", "cam1_track_id", "cam2_track_id", "cross_camera_distance_cm"] if c in view.columns]
    fig = px.scatter(
        view,
        x="ground_x", y="ground_y", color="khu_vuc_hien_thi",
        color_discrete_map={"Khu A": "#00b4d8", "Khu B": "#ffb703", "Khu C": "#ef476f"},
        hover_data=hover,
        category_orders={"khu_vuc_hien_thi": ["Khu A", "Khu B", "Khu C"]},
        labels={
            "ground_x": "Tọa độ X mặt bằng", "ground_y": "Tọa độ Y mặt bằng",
            "khu_vuc_hien_thi": "Khu vực", "source": "Nguồn", "global_id": "ID toàn cục",
            "cam1_track_id": "ID Camera 1", "cam2_track_id": "ID Camera 2",
            "cross_camera_distance_cm": "Khoảng cách ghép (cm)",
        },
        title=title,
    )
    fig.update_traces(marker=dict(size=12, opacity=.82, line=dict(width=1, color="white")))
    fig.update_layout(height=585, legend_title_text="Khu vực PCA", margin=dict(l=10, r=10, t=55, b=10))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _video_bytes(path: str | None):
    if not path:
        return None
    p = Path(path)
    try:
        return p.read_bytes() if p.exists() else None
    except OSError:
        return None


def _camera_summary(run, key_prefix: str):
    tracks = run.tracks
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số khung đã suy luận", run.processed_frames)
    c2.metric("Số bản ghi bám vết", len(tracks))
    c3.metric("Số ID cục bộ", int(tracks["track_id"].nunique()) if not tracks.empty else 0)
    mean_occ = tracks.groupby("sample_index").size().mean() if not tracks.empty else 0.0
    c4.metric("Số người trung bình", f"{mean_occ:.2f}")
    st.caption(
        f"Video: {run.info.width}×{run.info.height} • {run.info.fps:.2f} FPS • "
        f"thời lượng {run.info.duration_sec:.1f} giây • hiệu chuẩn: {run.calibration_name or 'không dùng'}"
    )
    if run.preview_rgb is not None:
        st.image(run.preview_rgb, caption=f"Khung cuối đã xử lý — {run.camera}", use_container_width=True)
    vb = _video_bytes(run.video_path)
    if vb:
        with st.expander("Xem video bám vết đã xuất"):
            st.video(vb)
            st.download_button(
                "Tải video MP4 đã đánh dấu",
                vb,
                file_name=f"{key_prefix}_tracked.mp4",
                mime="video/mp4",
                key=f"dl_video_{key_prefix}",
            )
    if not tracks.empty:
        st.download_button(
            "Tải CSV bám vết",
            tracks.to_csv(index=False).encode("utf-8"),
            file_name=f"{key_prefix}_tracks.csv",
            mime="text/csv",
            key=f"dl_csv_{key_prefix}",
        )


def _source_id_single(session) -> str:
    info = session.status
    return f"1 camera | {info['source']}"


def _log_live_counts(source_id: str, sample_key: str, counts: dict[str, int], recorded_at: float | None = None):
    decisions = _decisions(counts)
    log_zone_snapshot(source_id, sample_key, counts, CAP, decisions, recorded_at=recorded_at)
    return decisions


def _forecast_badges(forecasts: pd.DataFrame):
    if forecasts is None or forecasts.empty:
        st.caption("Chưa đủ lịch sử để tạo dự báo.")
        return
    alerts = forecasts[forecasts["status"].isin(["BUSY", "CONGESTED"])].copy()
    if alerts.empty:
        st.success("Trong các mốc +5/+10/+15 phút hiện chưa có khu nào vượt ngưỡng ĐÔNG theo mô hình dự báo nguyên mẫu.")
        return
    severity = {"CONGESTED": 2, "BUSY": 1}
    alerts["_sev"] = alerts["status"].map(severity)
    alerts = alerts.sort_values(["_sev", "horizon_min"], ascending=[False, True])
    for _, r in alerts.head(4).iterrows():
        status_vi = STATUS_VI.get(r["status"], r["status"])
        action_vi = ACTION_VI.get(r["action"], r["action"])
        alt = f" → {ZONE_VI.get(r['alternative'], r['alternative'])}" if r.get("alternative", "") else ""
        fn = st.error if r["status"] == "CONGESTED" else st.warning
        fn(
            f"**{ZONE_VI.get(r['zone'], r['zone'])} • +{int(r['horizon_min'])} phút: {status_vi}** — "
            f"dự báo {r['forecast']:.1f}/{int(r['capacity'])} người ({r['predicted_utilization']:.0%}). "
            f"Khuyến nghị: **{action_vi}{alt}**. Mức tin cậy: **{r['reliability']}**."
        )


def _forecast_for_source(source_id: str, window_minutes: float = 10.0):
    hist = read_metrics(source=source_id, since_minutes=max(60.0, window_minutes + 20.0))
    zone_hist = hist[hist["zone"].isin(ZONE_ORDER)].copy() if not hist.empty else hist
    if zone_hist.empty:
        return hist, pd.DataFrame(), {}
    forecasts, meta = forecast_all(
        zone_hist, CAP, ZONE_ORDER, horizons=(5, 10, 15), window_minutes=window_minutes,
        busy_threshold=busy, congested_threshold=congested,
    )
    return zone_hist, forecasts, meta


def _anomaly_controls(prefix: str):
    c1, c2, c3 = st.columns(3)
    loiter_sec = c1.number_input("Ngưỡng lưu lại bất thường (giây)", min_value=10, max_value=3600, value=120, step=10, key=f"{prefix}_loiter")
    speed_cm_s = c2.number_input("Ngưỡng tốc độ bất thường (cm/s)", min_value=50, max_value=1500, value=250, step=10, key=f"{prefix}_speed")
    max_gap = c3.number_input("Khoảng mất track tối đa để nối episode (giây)", min_value=0.5, max_value=10.0, value=2.0, step=0.5, key=f"{prefix}_gap")
    restricted = set(st.multiselect("Khu vực hạn chế", ZONE_ORDER, default=[], key=f"{prefix}_restricted"))
    wrongway_options = [f"{a} → {b}" for a in ZONE_ORDER for b in ZONE_ORDER if a != b]
    wrongway_selected = st.multiselect("Hướng di chuyển không hợp lệ", wrongway_options, default=[], key=f"{prefix}_wrongway")
    forbidden = {tuple(x.split(" → ")) for x in wrongway_selected}
    d1, d2, d3 = st.columns(3)
    surge_people = d1.number_input("Tăng nhanh đám đông: số người", min_value=1, max_value=100, value=4, step=1, key=f"{prefix}_surge")
    surge_window = d2.number_input("Cửa sổ tăng nhanh (số mẫu)", min_value=2, max_value=50, value=5, step=1, key=f"{prefix}_surge_win")
    persistent = d3.number_input("Ùn tắc kéo dài (số mẫu)", min_value=2, max_value=50, value=4, step=1, key=f"{prefix}_persistent")
    abrupt = st.checkbox("Bật chẩn đoán đổi hướng đột ngột (experimental)", value=False, key=f"{prefix}_abrupt")
    return {
        "loitering_sec": float(loiter_sec), "speed_threshold_cm_s": float(speed_cm_s),
        "max_gap_sec": float(max_gap), "restricted_zones": restricted,
        "forbidden_transitions": forbidden, "surge_people": float(surge_people),
        "surge_window_points": int(surge_window), "persistent_points": int(persistent),
        "detect_abrupt_turn": bool(abrupt),
    }


def _analyze_anomalies(observations: pd.DataFrame, occupancy_history: pd.DataFrame, decisions, cfg: dict):
    person = detect_trajectory_anomalies(
        observations, loitering_sec=cfg["loitering_sec"], speed_threshold_cm_s=cfg["speed_threshold_cm_s"],
        restricted_zones=cfg["restricted_zones"], forbidden_transitions=cfg["forbidden_transitions"],
        max_gap_sec=cfg["max_gap_sec"], detect_abrupt_turn=cfg["detect_abrupt_turn"],
    )
    crowd = detect_crowd_anomalies(
        occupancy_history, CAP, busy_threshold=busy, congested_threshold=congested,
        surge_people=cfg["surge_people"], surge_window_points=cfg["surge_window_points"],
        persistent_points=cfg["persistent_points"],
    )
    incidents = build_incident_table(person, crowd, decisions)
    return person, crowd, incidents


def _render_incidents(person: pd.DataFrame, crowd: pd.DataFrame, incidents: pd.DataFrame, prefix: str):
    a, b, c, d = st.columns(4)
    a.metric("Cảnh báo đối tượng", int(len(person)))
    b.metric("Cảnh báo đám đông", int(len(crowd)))
    high = 0 if incidents.empty else int(incidents["severity"].isin(["HIGH", "CRITICAL"]).sum())
    c.metric("Mức cao/nghiêm trọng", high)
    ids = 0 if person.empty else int(person["global_id"].nunique())
    d.metric("Đối tượng liên quan", ids)
    if incidents.empty:
        st.success("Chưa ghi nhận bất thường theo cấu hình hiện tại.")
        return
    st.markdown("**Hàng đợi sự cố ưu tiên**")
    st.dataframe(incidents, use_container_width=True, hide_index=True)
    t1, t2 = st.tabs(["Chi tiết đối tượng", "Chi tiết đám đông"])
    with t1:
        st.dataframe(person, use_container_width=True, hide_index=True) if not person.empty else st.caption("Không có cảnh báo đối tượng.")
    with t2:
        st.dataframe(crowd, use_container_width=True, hide_index=True) if not crowd.empty else st.caption("Không có cảnh báo đám đông.")
    st.download_button("Tải CSV hàng đợi sự cố", incidents.to_csv(index=False).encode("utf-8"), f"{prefix}_incident_queue.csv", "text/csv", key=f"{prefix}_incident_csv")


if page == "Tổng quan":
    st.title("Hệ thống theo dõi, phân tích và hỗ trợ điều hướng dòng người")
    st.caption("Nguyên mẫu nghiên cứu khoa học • YOLO11n + ByteTrack + hợp nhất đa camera trên mặt bằng + phân vùng PCA + hỗ trợ quyết định")
    st.markdown("<div class='v5-banner'><b>V6:</b> ưu tiên single-camera • ROI • tiled inference • quét confidence/độ phân giải • luồng camera và AI độc lập • giữ V4 forecast/SQLite/multi-camera để dùng sau khi detector đạt.</div>", unsafe_allow_html=True)
    v4stats = db_stats()
    v41, v42, v43 = st.columns(3)
    v41.metric("Nguồn realtime đã lưu", len(list_sources()))
    v42.metric("Bản ghi lịch sử SQLite", v4stats["metric_rows"])
    v43.metric("Tầm cảnh báo sớm", "5 / 10 / 15 phút")

    idx = _frame_controls("overview_frame")
    counts = _current_counts(idx)
    decisions = _decisions(counts)
    current = PRED_TEST[PRED_TEST["gt_index"] == idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số người dự đoán", len(current))
    c2.metric("Khu A / B / C", f"{counts['Zone A']} / {counts['Zone B']} / {counts['Zone C']}")
    c3.metric("Quan sát đã hợp nhất", int((current["source"] == "cam1+cam2").sum()))
    worst = max(decisions, key=lambda d: d.utilization)
    c4.metric("Hành động hệ thống", ACTION_VI.get(worst.action, worst.action), f"{ZONE_VI.get(worst.zone, worst.zone)}: {worst.utilization:.0%}")

    left, right = st.columns([1.8, 1])
    with left:
        st.plotly_chart(ground_figure(current, "Bản đồ hành khách trên mặt bằng — Pair12 TEST"), use_container_width=True)
    with right:
        st.subheader("Trạng thái đám đông")
        for dec in decisions:
            _status_badge(dec)
        st.info("Bộ hỗ trợ quyết định là lớp minh họa. Nó không thay thế quy trình an toàn hoặc điều hành thực tế và không được xem là ngưỡng vận hành sân bay đã được kiểm định.")

    st.subheader("Chuỗi xử lý")
    st.code("Video/CCTV → YOLO11n → ByteTrack → điểm chân → hiệu chuẩn camera → chiếu xuống mặt bằng → hợp nhất Hungarian → quan sát toàn cục → phân vùng PCA → số người/luồng/thời gian lưu lại → GIỮ NGUYÊN / HẠN CHẾ / ĐIỀU HƯỚNG LẠI", language="text")

elif page == "Phòng thí nghiệm 1 camera":
    st.title("Phòng thí nghiệm 1 camera — V6")
    st.caption("Ưu tiên sửa phát hiện/đếm/bám vết ở một camera trước khi đánh giá hợp nhất nhiều camera. Đây là nơi kiểm tra người nhỏ, đám đông, ROI, ngưỡng, độ phân giải và tốc độ.")
    st.warning("Quy tắc V6: nếu single-camera chưa đạt trên bộ TEST riêng, chưa dùng kết quả multi-camera để kết luận chất lượng hệ thống.")

    tab_img, tab_fps, tab_plan, tab_train = st.tabs(["1. Ảnh & đếm người", "2. Realtime & FPS", "3. Bộ ca kiểm thử", "4. Fine-tune model"] )

    with tab_img:
        st.subheader("Kiểm tra detector trên một ảnh")
        lab_upload = st.file_uploader("Tải một ảnh kiểm thử", type=["jpg","jpeg","png","webp","bmp"], key="v6_lab_image")
        model_mode = st.radio("Mô hình", ["YOLO11n đóng gói", "Model .pt tùy chỉnh / fine-tune"], horizontal=True, key="v6_model_mode")
        custom_model = None
        model_path = ""
        if model_mode.startswith("Model"):
            custom_model = st.file_uploader("Tải model .pt", type=["pt"], key="v6_custom_pt")
            if custom_model is not None:
                sig = (custom_model.name, custom_model.size)
                if st.session_state.get("v6_custom_model_sig") != sig:
                    st.session_state["v6_custom_model_path"] = save_model_upload(custom_model.getvalue(), custom_model.name)
                    st.session_state["v6_custom_model_sig"] = sig
                    cached_detector_model.clear()
                model_path = st.session_state.get("v6_custom_model_path", "")
                st.success(f"Đã nạp model tùy chỉnh: {custom_model.name}")
            else:
                st.info("Chưa tải model .pt; V6 sẽ không tự coi YOLO11n COCO là model head detector.")

        profile = st.selectbox(
            "Cấu hình thử nghiệm",
            ["Đám đông / người nhỏ", "Cân bằng", "Nhanh", "Tùy chỉnh"],
            help="Đám đông/người nhỏ mặc định dùng conf thấp + ảnh lớn + tiled inference. Cấu hình này chậm hơn nhưng phù hợp ảnh bạn vừa gửi.",
            key="v6_profile",
        )
        presets = {
            "Đám đông / người nhỏ": dict(conf=0.10, imgsz=1280, max_det=1000, tiled=True, tile=768, overlap=0.20),
            "Cân bằng": dict(conf=0.15, imgsz=960, max_det=1000, tiled=False, tile=768, overlap=0.20),
            "Nhanh": dict(conf=0.20, imgsz=640, max_det=500, tiled=False, tile=640, overlap=0.15),
        }
        if profile == "Tùy chỉnh":
            p1,p2,p3 = st.columns(3)
            lab_conf = p1.slider("Confidence", 0.03, 0.80, 0.15, 0.01, key="v6_lab_conf")
            lab_imgsz = p2.select_slider("Kích thước YOLO", options=[512,640,768,960,1024,1280,1536], value=1280, key="v6_lab_imgsz")
            lab_max_det = p3.number_input("Số detection tối đa", 50, 3000, 1000, 50, key="v6_lab_maxdet")
            p4,p5,p6 = st.columns(3)
            lab_tiled = p4.checkbox("Bật tiled inference", value=True, key="v6_lab_tiled")
            lab_tile = p5.selectbox("Kích thước tile", [512,640,768,896,1024], index=2, key="v6_lab_tile")
            lab_overlap = p6.slider("Chồng lấn tile", 0.05, 0.40, 0.20, 0.05, key="v6_lab_overlap")
        else:
            pr = presets[profile]
            lab_conf, lab_imgsz, lab_max_det = pr["conf"], pr["imgsz"], pr["max_det"]
            lab_tiled, lab_tile, lab_overlap = pr["tiled"], pr["tile"], pr["overlap"]
            st.code(f"conf={lab_conf} | imgsz={lab_imgsz} | max_det={lab_max_det} | tiled={lab_tiled} | tile={lab_tile} | overlap={lab_overlap}", language="text")

        h1,h2,h3 = st.columns(3)
        lab_device_ui = h1.selectbox("Thiết bị", ["Tự động", "CPU", "GPU / CUDA"], key="v6_lab_device")
        lab_device = {"Tự động":"auto", "CPU":"cpu", "GPU / CUDA":"0"}[lab_device_ui]
        lab_half = h2.checkbox("FP16 khi có GPU", value=True, key="v6_lab_half")
        lab_iou = h3.slider("IoU NMS", 0.30, 0.80, 0.55, 0.05, key="v6_lab_iou")

        roi = None
        if lab_upload is not None:
            frame = decode_image_bytes(lab_upload.getvalue())
            use_roi = st.checkbox("Chỉ xử lý vùng quan tâm ROI", value=False, key="v6_use_roi")
            if use_roi:
                rx = st.slider("ROI ngang: trái → phải (%)", 0, 100, (0,100), 1, key="v6_roi_x")
                ry = st.slider("ROI dọc: trên → dưới (%)", 0, 100, (0,100), 1, key="v6_roi_y")
                roi = (rx[0]/100.0, ry[0]/100.0, rx[1]/100.0, ry[1]/100.0)
                st.image(draw_roi_preview(frame, roi), caption="Xem trước ROI — chỉ vùng trong khung vàng được đưa vào AI", use_container_width=True)
            else:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Ảnh kiểm thử", use_container_width=True)

            cfg = DetectorConfig(
                confidence=float(lab_conf), image_size=int(lab_imgsz), max_det=int(lab_max_det),
                iou=float(lab_iou), device=lab_device, half=bool(lab_half),
                tiled=bool(lab_tiled), tile_size=int(lab_tile), tile_overlap=float(lab_overlap), roi=roi,
            )
            if st.button("▶ Chạy kiểm tra single-camera", type="primary", key="v6_run_single"):
                if model_mode.startswith("Model") and not model_path:
                    st.error("Hãy tải model .pt trước.")
                else:
                    try:
                        with st.spinner("Đang chạy detector..."):
                            model = cached_detector_model(model_path)
                            lab_run = detect_frame(frame, cfg, model=model)
                            st.session_state["v6_lab_run"] = lab_run
                            st.session_state["v6_lab_cfg"] = cfg
                            st.session_state["v6_lab_image_name"] = lab_upload.name
                    except Exception as e:
                        st.exception(e)

            lab_run = st.session_state.get("v6_lab_run")
            if lab_run is not None and st.session_state.get("v6_lab_image_name") == lab_upload.name:
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Số đối tượng phát hiện", len(lab_run.detections))
                m2.metric("Thời gian AI", f"{lab_run.inference_ms:.0f} ms")
                m3.metric("FPS tương đương", f"{1000/max(1,lab_run.inference_ms):.1f}")
                m4.metric("Số tile", lab_run.tiles_processed)
                st.image(lab_run.annotated_rgb, caption="Kết quả — xem bounding box để xác định model bỏ sót ở đâu", use_container_width=True)
                if not lab_run.detections.empty:
                    st.dataframe(viet_df(lab_run.detections), use_container_width=True, hide_index=True)
                    st.caption(f"Box nhỏ nhất: {lab_run.detections['height'].min():.1f}px cao • trung vị: {lab_run.detections['height'].median():.1f}px • thiết bị: {lab_run.device_used}")
                ok, enc = cv2.imencode('.jpg', cv2.cvtColor(lab_run.annotated_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY),94])
                if ok:
                    st.download_button("Tải ảnh kết quả V6", enc.tobytes(), f"v6_{Path(lab_upload.name).stem}.jpg", "image/jpeg", key="v6_lab_dl")

                st.markdown("#### So với số người thật")
                known_gt = st.checkbox("Tôi đã đếm thủ công số người thật trong ROI/ảnh", key="v6_has_gt")
                if known_gt:
                    gt_count = st.number_input("Số người thật (GT thủ công)", 0, 10000, max(0,len(lab_run.detections)), 1, key="v6_gt_count")
                    max_err = st.slider("Sai số đếm tuyệt đối % cho phép (tiêu chí sơ bộ)", 1, 50, 20, 1, key="v6_maxerr")
                    met = count_error_metrics(len(lab_run.detections), int(gt_count))
                    e1,e2,e3 = st.columns(3)
                    e1.metric("GT", int(gt_count)); e2.metric("Sai số tuyệt đối", int(met['absolute_error']))
                    e3.metric("Sai số %", "∞" if not np.isfinite(met['absolute_error_pct']) else f"{met['absolute_error_pct']:.1f}%")
                    if np.isfinite(met['absolute_error_pct']) and met['absolute_error_pct'] <= max_err:
                        st.success("ĐẠT tiêu chí đếm sơ bộ do bạn đặt. Muốn công bố khoa học vẫn cần annotation đầy đủ để tính Precision/Recall/mAP.")
                    else:
                        st.error("CHƯA ĐẠT single-camera theo tiêu chí đếm đang đặt — chưa nên chuyển sang multi-camera.")

                st.markdown("#### Chẩn đoán confidence / độ phân giải")
                q1,q2 = st.columns(2)
                if q1.button("Quét confidence 0.05 → 0.50", key="v6_sweep_conf"):
                    try:
                        with st.spinner("Đang quét confidence; có thể mất vài phút nếu bật tiled..."):
                            model = cached_detector_model(model_path)
                            sw = threshold_sweep(frame, st.session_state.get("v6_lab_cfg",cfg), [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50], model=model)
                            st.session_state["v6_conf_sweep"] = sw
                    except Exception as e: st.exception(e)
                if q2.button("Quét kích thước 640 / 960 / 1280", key="v6_sweep_size"):
                    try:
                        with st.spinner("Đang quét độ phân giải..."):
                            model = cached_detector_model(model_path)
                            sw = resolution_sweep(frame, st.session_state.get("v6_lab_cfg",cfg), [640,960,1280], model=model)
                            st.session_state["v6_size_sweep"] = sw
                    except Exception as e: st.exception(e)
                if "v6_conf_sweep" in st.session_state:
                    sw=st.session_state["v6_conf_sweep"]
                    st.dataframe(sw, use_container_width=True, hide_index=True)
                    fig=px.line(sw,x="confidence",y="count",markers=True,title="Số detection theo confidence")
                    st.plotly_chart(fig,use_container_width=True,key="v6_conf_chart")
                if "v6_size_sweep" in st.session_state:
                    sw=st.session_state["v6_size_sweep"]
                    st.dataframe(sw,use_container_width=True,hide_index=True)
                    fig=px.line(sw,x="image_size",y="count",markers=True,title="Số detection theo kích thước suy luận")
                    st.plotly_chart(fig,use_container_width=True,key="v6_size_chart")
        else:
            st.info("Tải ảnh lên để bắt đầu. Với ảnh đám đông giống ảnh bạn vừa thử, hãy chọn **Đám đông / người nhỏ** trước.")

    with tab_fps:
        st.subheader("Realtime: tách FPS camera và FPS AI")
        st.markdown("V6 đã sửa kiến trúc: **luồng camera** đọc frame liên tục và chỉ giữ frame mới nhất; **luồng AI** chạy nền độc lập theo target FPS. Streamlit chỉ hiển thị kết quả mới nhất nên không còn khóa tốc độ AI vào chu kỳ refresh giao diện như V4.")
        f1,f2,f3 = st.columns(3)
        f1.metric("Target gợi ý — Nhanh", "20–30 FPS")
        f2.metric("Target gợi ý — Cân bằng", "10–20 FPS")
        f3.metric("Đám đông / người nhỏ", "5–12 FPS")
        if _hw.cuda_available:
            st.success(f"CUDA đang khả dụng: {_hw.device_name}. Hãy chọn **Tự động** hoặc **GPU / CUDA** và bật FP16.")
        else:
            st.error("CUDA chưa khả dụng trong Python hiện tại. Nếu dùng CPU, tăng imgsz/tiled sẽ làm FPS giảm mạnh.")
        st.info("Mở màn hình **CCTV / RTSP thời gian thực** để đo riêng: FPS camera thực đo, FPS AI thực đo, thời gian suy luận và độ trễ frame.")

    with tab_plan:
        st.subheader("Bộ ca kiểm thử single-camera trước multi-camera")
        cases = pd.DataFrame([
            ["S1","1 người gần camera","Detection cơ bản"],
            ["S2","5–10 người","Nhiều người"],
            ["S3","Người xa / rất nhỏ","Small-object"],
            ["S4","20–50 người","Crowd"],
            ["S5","Đám đông rất dày","Dense crowd"],
            ["S6","Người che khuất nhau","Occlusion"],
            ["S7","Đi nhanh / nhòe","Motion blur"],
            ["S8","Sáng / tối / ngược sáng","Illumination"],
            ["S9","Ra/vào khung hình","Tracking lifecycle"],
            ["S10","Hai người cắt ngang nhau","ID switch"],
        ], columns=["Mã","Tình huống","Mục tiêu"] )
        cases["Đã test"] = False
        st.data_editor(cases, use_container_width=True, hide_index=True, key="v6_test_cases")
        st.markdown("**Chỉ số nên ghi:** Precision, Recall, mAP50/mAP50-95, Count MAE/WAPE, IDF1/HOTA/MOTA, ID Switch, FPS camera, FPS AI và latency.")
        st.warning("Một model pretrained COCO không phải là bằng chứng đã 'học đặc trưng camera của bạn'. Muốn thích nghi ảnh góc cao + người nhỏ + xe máy, cần fine-tune trên dữ liệu được annotation và giữ TEST tách riêng.")

    with tab_train:
        st.subheader("Fine-tune để model học đặc trưng camera")
        st.markdown("V6 có sẵn `train_detector.py`. Sau khi bạn gán nhãn dữ liệu theo YOLO, chạy một trong các lệnh dưới đây rồi lấy `best.pt` nạp lại vào **Phòng thí nghiệm 1 camera**, **Video** hoặc **CCTV/RTSP**.")
        st.code("python train_detector.py --data D:/CrowdFlowDataset/data.yaml --epochs 100 --imgsz 1280 --batch 8 --device 0 --name crowdflow_person_v5", language="powershell")
        st.markdown("Với cảnh cực đông/góc nhìn cao có thể tạo một dataset **head** riêng và dùng `examples/data_head_example.yaml`. Person detector và head detector phải được đánh giá trên cùng TEST hoặc protocol rõ ràng trước khi chọn.")
        st.info("Không dùng ảnh TEST để train. Nếu test chính là ảnh/video bạn dùng để báo cáo, hãy tách nó ra trước khi fine-tune để tránh data leakage.")
        st.markdown("**Đầu ra:** `runs/detect/<tên>/weights/best.pt` → upload vào mục model `.pt` tùy chỉnh của V6.")

elif page == "Phân tích video":
    st.title("Phân tích video trực tiếp")
    st.caption("Chạy suy luận thật bằng mô hình YOLO11n được đóng gói từ kết quả nghiên cứu, kết hợp ByteTrack. Không cần CSV có sẵn.")

    mode = st.radio(
        "Chế độ",
        ["1 camera — phát hiện + bám vết", "2 camera WILDTRACK — hợp nhất mặt bằng + phân tích"],
        horizontal=True,
    )

    with st.expander("Cấu hình suy luận V6", expanded=True):
        a, b, c, d, e = st.columns(5)
        conf = a.slider("Độ tin cậy", 0.03, 0.80, 0.15, 0.01, key="vid_conf")
        imgsz = b.selectbox("Kích thước ảnh YOLO", [640, 768, 960, 1280], index=2, key="vid_imgsz")
        target_fps = c.slider("FPS AI mục tiêu", 1, 30, 15, key="vid_fps")
        start_sec = d.number_input("Bắt đầu từ (s)", 0.0, 3600.0, 0.0, 1.0)
        max_seconds = e.slider("Độ dài xử lý (s)", 3, 120, 15)
        v1,v2,v3 = st.columns(3)
        device_ui = v1.radio("Thiết bị", ["Tự động", "CPU", "GPU / CUDA"], horizontal=True)
        device = {"Tự động": "auto", "CPU": "cpu", "GPU / CUDA": "0"}[device_ui]
        vid_max_det = v2.number_input("Số detection tối đa", 50, 3000, 1000, 50, key="vid_max_det")
        vid_half = v3.checkbox("FP16 khi có GPU", value=True, key="vid_half")
        vid_pt = st.file_uploader("Model .pt tùy chỉnh cho video (tùy chọn)", type=["pt"], key="vid_custom_pt")
        vid_model_path = None
        if vid_pt is not None:
            sig=(vid_pt.name,vid_pt.size)
            if st.session_state.get("vid_custom_sig") != sig:
                st.session_state["vid_custom_path"] = save_model_upload(vid_pt.getvalue(), vid_pt.name)
                st.session_state["vid_custom_sig"] = sig
            vid_model_path = st.session_state.get("vid_custom_path")
        config = TrackingConfig(
            confidence=float(conf), image_size=int(imgsz), target_fps=float(target_fps),
            start_sec=float(start_sec), max_seconds=float(max_seconds), device=device,
            max_det=int(vid_max_det), half=bool(vid_half), model_path=vid_model_path,
        )

    if mode.startswith("1 camera"):
        upload = st.file_uploader("Tải video lên", type=["mp4", "avi", "mov", "mkv"], key="single_video")
        cal_options = ["Không dùng hiệu chuẩn"] + list(CAMERA_CALIBRATION.keys())
        cal_choice = st.selectbox("Hiệu chuẩn mặt bằng (tùy chọn)", cal_options, format_func=lambda x: "Không dùng hiệu chuẩn" if x == "Không dùng hiệu chuẩn" else CALIB_VI.get(x, x))
        calibration = None if cal_choice == "Không dùng hiệu chuẩn" else cal_choice
        if calibration:
            st.warning("Tọa độ mặt bằng chỉ có ý nghĩa nếu video đúng camera WILDTRACK tương ứng. Với video tự quay, chọn 'Không dùng hiệu chuẩn' cho tới khi hiệu chuẩn camera riêng.")
        use_vid_roi = st.checkbox("Bật ROI cho video 1 camera", value=False, key="vid_use_roi")
        vid_roi = None
        if use_vid_roi:
            va,vb = st.columns(2)
            vx = va.slider("ROI ngang video (%)", 0,100,(0,100),1,key="vid_roi_x")
            vy = vb.slider("ROI dọc video (%)", 0,100,(35,100),1,key="vid_roi_y")
            vid_roi=(vx[0]/100.0,vy[0]/100.0,vx[1]/100.0,vy[1]/100.0)
        single_video_config = TrackingConfig(**{**config.__dict__, "roi": vid_roi})

        if st.button("▶ Chạy phân tích 1 camera", type="primary", disabled=upload is None):
            status = st.status("Đang chạy YOLO11n + ByteTrack...", expanded=True)
            progress_bar = st.progress(0.0)
            label = st.empty()
            try:
                path = save_upload(upload)
                def cb(p, text):
                    progress_bar.progress(float(p))
                    label.caption(text)
                run = process_camera(path, "camera1", single_video_config, calibration, progress=cb)
                st.session_state["single_run"] = run
                status.update(label="Hoàn tất phân tích video", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Phân tích thất bại", state="error", expanded=True)
                st.exception(e)

        run = st.session_state.get("single_run")
        if run is not None:
            st.subheader("Kết quả")
            _camera_summary(run, "single_camera")
            if run.calibration_name and not run.tracks.empty:
                valid = run.tracks[np.isfinite(run.tracks["ground_x"]) & np.isfinite(run.tracks["ground_y"])].copy()
                if not valid.empty:
                    valid["zone"] = assign_zones(valid, ZMODEL)
                    sid = int(valid["sample_index"].max())
                    cur = valid[valid["sample_index"] == sid]
                    st.plotly_chart(ground_figure(cur.assign(source="camera1"), "Bản đồ mặt bằng — khung cuối"), use_container_width=True)
                    with st.expander("⚠️ Bất thường quỹ đạo 1 camera (ID cục bộ)", expanded=False):
                        st.caption("Chỉ dùng khi video đúng hiệu chuẩn mặt bằng. ID ở đây là ID ByteTrack cục bộ trong camera, không phải nhận dạng danh tính.")
                        cfg_a = _anomaly_controls("single_video_anom")
                        local_obs = valid.copy()
                        local_obs["global_id"] = local_obs["track_id"].astype(int)
                        occ_long = (local_obs.groupby(["sample_index", "zone"]).size().unstack(fill_value=0).reset_index())
                        for z in ZONE_ORDER:
                            if z not in occ_long.columns:
                                occ_long[z] = 0
                        counts_now = {z: int((cur["zone"] == z).sum()) for z in ZONE_ORDER}
                        dec_now = _decisions(counts_now)
                        person_a, crowd_a, incidents_a = _analyze_anomalies(local_obs, occ_long, dec_now, cfg_a)
                        _render_incidents(person_a, crowd_a, incidents_a, "single_video")

    else:
        st.warning("V6: chỉ dùng multi-camera sau khi từng camera đã đạt kiểm thử single-camera. Chế độ này dành cho **WILDTRACK Camera 1 = CVLab1 và Camera 2 = CVLab2**; hai video phải đồng bộ thời gian và giữ đúng hình học gốc.")
        u1, u2 = st.columns(2)
        with u1:
            cam1_upload = st.file_uploader("Tải video Camera 1 / CVLab1", type=["mp4", "avi", "mov", "mkv"], key="pair_cam1")
        with u2:
            cam2_upload = st.file_uploader("Tải video Camera 2 / CVLab2", type=["mp4", "avi", "mov", "mkv"], key="pair_cam2")
        threshold = st.slider("Ngưỡng hợp nhất Hungarian (cm)", 25, 200, 100, 5)

        if st.button("▶ Chạy hợp nhất Pair12", type="primary", disabled=(cam1_upload is None or cam2_upload is None)):
            status = st.status("Đang chạy chuỗi xử lý Pair12...", expanded=True)
            progress_bar = st.progress(0.0)
            label = st.empty()
            try:
                p1 = save_upload(cam1_upload)
                p2 = save_upload(cam2_upload)

                def cb1(p, text):
                    progress_bar.progress(float(p) * 0.45)
                    label.caption(text)
                def cb2(p, text):
                    progress_bar.progress(0.45 + float(p) * 0.45)
                    label.caption(text)

                run1 = process_camera(p1, "cam1", config, "Cam1 / CVLab1", progress=cb1)
                run2 = process_camera(p2, "cam2", config, "Cam2 / CVLab2", progress=cb2)
                label.caption("Đang hợp nhất chéo camera bằng Hungarian...")
                obs, matches = fuse_pair(run1.tracks, run2.tracks, float(threshold))
                interval = 1.0 / max(0.5, config.target_fps)
                fusion = summarize_fusion(obs, ZMODEL, interval)
                progress_bar.progress(1.0)
                st.session_state["pair_run"] = {
                    "run1": run1, "run2": run2, "obs": fusion.observations,
                    "matches": matches, "occupancy": fusion.occupancy,
                    "transitions": fusion.transitions, "dwell": fusion.dwell,
                    "threshold": threshold,
                }
                status.update(label="Hoàn tất Pair12: bám vết → mặt bằng → hợp nhất → phân tích", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Chuỗi xử lý Pair12 thất bại", state="error", expanded=True)
                st.exception(e)

        pair = st.session_state.get("pair_run")
        if pair is not None:
            run1, run2 = pair["run1"], pair["run2"]
            obs, matches = pair["obs"], pair["matches"]
            st.subheader("1. Bám vết từng camera")
            t1, t2 = st.tabs(["Camera 1", "Camera 2"])
            with t1:
                _camera_summary(run1, "pair_cam1")
            with t2:
                _camera_summary(run2, "pair_cam2")

            st.subheader("2. Hợp nhất giữa các camera")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Quan sát toàn cục", len(obs))
            m2.metric("Cặp ghép khớp", len(matches))
            m3.metric("ID toàn cục ước lượng", int(obs["global_id"].nunique()) if not obs.empty else 0)
            mean_dist = matches["cross_camera_distance_cm"].mean() if not matches.empty else np.nan
            m4.metric("Khoảng cách ghép trung bình", "—" if not np.isfinite(mean_dist) else f"{mean_dist:.1f} cm")

            if not obs.empty:
                sample_ids = sorted(obs["sample_index"].unique().astype(int).tolist())
                sid = st.slider("Mẫu bản đồ mặt bằng", min(sample_ids), max(sample_ids), max(sample_ids), key="live_ground_sample")
                cur = obs[obs["sample_index"] == sid]
                st.plotly_chart(ground_figure(cur, f"Hợp nhất mặt bằng Pair12 — mẫu {sid}"), use_container_width=True)

                st.subheader("3. Số người hiện diện + hỗ trợ quyết định")
                occ = pair["occupancy"]
                row = occ[occ["sample_index"] == sid]
                counts = {z: 0 for z in ZONE_ORDER}
                if not row.empty:
                    counts.update({z: int(row.iloc[0].get(z, 0)) for z in ZONE_ORDER})
                decs = _decisions(counts)
                cols = st.columns(3)
                for col, dec in zip(cols, decs):
                    with col:
                        _status_badge(dec)

                fig_occ = go.Figure()
                for z in ZONE_ORDER:
                    if z in occ.columns:
                        fig_occ.add_trace(go.Scatter(x=occ["sample_index"], y=occ[z], mode="lines+markers", name=ZONE_VI.get(z, z)))
                fig_occ.update_layout(height=370, title="Số người theo khung lấy mẫu", xaxis_title="Chỉ số mẫu", yaxis_title="Số người")
                st.plotly_chart(fig_occ, use_container_width=True)

                st.subheader("4. Nguyên mẫu luồng di chuyển và thời gian lưu lại")
                l, r = st.columns(2)
                with l:
                    st.markdown("**Chuyển vùng có hướng**")
                    if pair["transitions"].empty:
                        st.caption("Chưa quan sát đủ sự kiện chuyển vùng trong đoạn video này.")
                    else:
                        st.dataframe(viet_df(pair["transitions"]), use_container_width=True, hide_index=True)
                with r:
                    st.markdown("**Lượt lưu lại (ước lượng từ các khung lấy mẫu)**")
                    if pair["dwell"].empty:
                        st.caption("Chưa có lượt lưu lại.")
                    else:
                        dwell_summary = pair["dwell"].groupby("zone")["dwell_sec"].agg(["count", "mean", "max"]).reset_index()
                        st.dataframe(viet_df(dwell_summary), use_container_width=True, hide_index=True)

                with st.expander("⚠️ Phát hiện bất thường đối tượng + đám đông", expanded=True):
                    st.caption("Module V6 dùng quỹ đạo mặt bằng và lịch sử occupancy để sàng lọc bất thường. Đây là lớp hỗ trợ quyết định; không phải nhận dạng danh tính hay kết luận an ninh tự động.")
                    cfg_pair = _anomaly_controls("offline_pair_anom")
                    person_pair, crowd_pair, incidents_pair = _analyze_anomalies(obs, occ, decs, cfg_pair)
                    _render_incidents(person_pair, crowd_pair, incidents_pair, "offline_pair12")

                dl1, dl2, dl3 = st.columns(3)
                dl1.download_button("Tải quan sát đã hợp nhất", obs.to_csv(index=False).encode("utf-8"), "live_pair12_fusion.csv", "text/csv")
                dl2.download_button("Tải các cặp ghép khớp", matches.to_csv(index=False).encode("utf-8"), "live_pair12_matches.csv", "text/csv")
                dl3.download_button("Tải dữ liệu số người hiện diện", occ.to_csv(index=False).encode("utf-8"), "live_pair12_occupancy.csv", "text/csv")

                st.warning("ID toàn cục, luồng và thời gian lưu lại ở chế độ video là kết quả suy luận nguyên mẫu trên đoạn video tải lên. Không được trộn các số này với chỉ số TEST tách riêng đã báo cáo nếu chưa chạy đúng quy trình đánh giá.")
            else:
                st.warning("Không có quan sát mặt bằng hợp lệ để hợp nhất. Hãy kiểm tra video có đúng Camera 1/Camera 2 WILDTRACK, hiệu chuẩn, cắt ảnh/độ phân giải và ngưỡng độ tin cậy hay không.")


elif page == "CCTV / RTSP thời gian thực":
    st.title("CCTV / RTSP thời gian thực")
    st.caption("Kết nối camera mạng hoặc webcam, lấy khung hình mới nhất và chạy YOLO11n + ByteTrack liên tục. Khi RTSP bị gián đoạn, hệ thống sẽ tự kết nối lại.")
    st.info("RTSP chạy từ máy đang mở app. Máy đó phải truy cập được IP camera trong cùng mạng/VPN. URL camera không được ghi vào file bởi ứng dụng này.")

    live_mode = st.radio(
        "Chế độ thời gian thực",
        ["1 camera thời gian thực", "2 camera thời gian thực + hợp nhất mặt bằng"],
        horizontal=True,
    )

    with st.expander("Cấu hình thời gian thực V6", expanded=True):
        st.caption("V6 không còn giới hạn 8 FPS. Luồng camera và luồng AI chạy nền độc lập; UI chỉ lấy kết quả mới nhất.")
        a, b, c, d = st.columns(4)
        live_conf = a.slider("Độ tin cậy", 0.03, 0.80, 0.15, 0.01, key="rt_conf")
        live_imgsz = b.selectbox("Kích thước ảnh YOLO", [640, 768, 960, 1280], index=2, key="rt_imgsz")
        live_fps = c.slider("FPS AI mục tiêu", 1, 30, 15, 1, key="rt_fps")
        live_device_ui = d.selectbox("Thiết bị", ["Tự động", "CPU", "GPU / CUDA"], key="rt_device")
        live_device = {"Tự động": "auto", "CPU": "cpu", "GPU / CUDA": "0"}[live_device_ui]
        e, f, g = st.columns(3)
        live_max_det = e.number_input("Số detection tối đa", 50, 3000, 1000, 50, key="rt_max_det")
        live_iou = f.slider("IoU NMS", 0.30, 0.80, 0.55, 0.05, key="rt_iou")
        live_half = g.checkbox("FP16 khi có GPU", value=True, key="rt_half")
        live_pt = st.file_uploader("Model .pt tùy chỉnh cho realtime (tùy chọn; để trống dùng YOLO11n)", type=["pt"], key="rt_custom_pt")
        live_model_path = None
        if live_pt is not None:
            sig=(live_pt.name,live_pt.size)
            if st.session_state.get("rt_custom_sig") != sig:
                st.session_state["rt_custom_path"] = save_model_upload(live_pt.getvalue(), live_pt.name)
                st.session_state["rt_custom_sig"] = sig
            live_model_path = st.session_state.get("rt_custom_path")
            st.caption(f"Realtime sẽ dùng model tùy chỉnh: {live_pt.name}")
        reconnect_delay = st.slider("Thời gian chờ kết nối lại (giây)", 0.5, 5.0, 1.0, 0.5, key="rt_reconnect")
        live_cfg = RealtimeConfig(
            confidence=float(live_conf), image_size=int(live_imgsz), target_fps=float(live_fps),
            device=live_device, reconnect_delay_sec=float(reconnect_delay), max_det=int(live_max_det),
            iou=float(live_iou), half=bool(live_half), model_path=live_model_path,
        )

    if live_mode.startswith("1 camera"):
        source_kind = st.radio("Nguồn", ["Camera IP qua RTSP / HTTP", "Webcam / USB"], horizontal=True, key="single_kind")
        if source_kind.startswith("Camera IP"):
            source_value = st.text_input(
                "Địa chỉ RTSP/HTTP", type="password", key="single_rtsp_url",
                placeholder="rtsp://username:password@192.168.1.100:554/stream1",
                help="Hỗ trợ RTSP và các luồng HTTP/MJPEG mà OpenCV có thể mở.",
            )
            live_source = source_value.strip()
        else:
            live_source = int(st.number_input("Chỉ số camera", 0, 10, 0, 1, key="single_webcam_index"))

        cal_opts = ["Không dùng hiệu chuẩn"] + list(CAMERA_CALIBRATION.keys())
        live_cal_ui = st.selectbox("Hiệu chuẩn mặt bằng (tùy chọn)", cal_opts, format_func=lambda x: "Không dùng hiệu chuẩn" if x == "Không dùng hiệu chuẩn" else CALIB_VI.get(x, x), key="single_live_cal")
        live_cal = None if live_cal_ui == "Không dùng hiệu chuẩn" else live_cal_ui
        if live_cal:
            st.warning("Chỉ dùng tọa độ mặt bằng và khu vực nếu luồng video đúng camera tương ứng. V6 ROI vẫn ánh xạ box về tọa độ ảnh gốc trước khi chiếu mặt bằng.")

        use_live_roi = st.checkbox("Bật ROI để bỏ vùng không có người", value=False, key="rt_use_roi")
        live_roi = None
        if use_live_roi:
            r1, r2 = st.columns(2)
            rtx = r1.slider("ROI ngang (%)", 0, 100, (0, 100), 1, key="rt_roi_x")
            rty = r2.slider("ROI dọc (%)", 0, 100, (35, 100), 1, key="rt_roi_y")
            live_roi = (rtx[0]/100.0, rty[0]/100.0, rtx[1]/100.0, rty[1]/100.0)
            st.caption("Với camera cố định như ảnh bạn gửi, cắt bỏ tòa nhà/bầu trời giúp người trong vùng đường đi lớn hơn tương đối và giảm tải AI.")
        single_cfg = RealtimeConfig(**{**live_cfg.__dict__, "roi": live_roi})

        b1, b2 = st.columns([1, 1])
        can_start = bool(live_source) if isinstance(live_source, str) else True
        if b1.button("▶ Bắt đầu thời gian thực", type="primary", disabled=not can_start, key="start_single_live"):
            old = st.session_state.get("single_live_session")
            if old is not None:
                old.stop()
            session = RealtimeStream(live_source, "CCTV-1", single_cfg, live_cal)
            session.start()
            st.session_state["single_live_session"] = session
            st.session_state["single_live_started_at"] = time.time()
        if b2.button("■ Dừng", key="stop_single_live"):
            old = st.session_state.pop("single_live_session", None)
            if old is not None:
                old.stop()
            st.success("Đã dừng camera thời gian thực.")

        @st.fragment(run_every=0.1)
        def render_single_live():
            session = st.session_state.get("single_live_session")
            if session is None:
                st.caption("Nhập nguồn camera và nhấn **Bắt đầu thời gian thực**.")
                return
            try:
                packet = session.analyze_latest()
            except Exception as e:
                st.error(f"Lỗi suy luận thời gian thực: {e}")
                return
            info = session.status
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Kết nối", CONNECTION_VI.get(info["status"], info["status"]))
            c2.metric("Số người hiện tại", len(packet.rows))
            c3.metric("FPS camera thực đo", f"{info.get('measured_capture_fps',0):.1f}")
            c4.metric("FPS AI thực đo", f"{info.get('measured_ai_fps',0):.1f}")
            c5.metric("Thời gian AI", "—" if packet.inference_ms is None else f"{packet.inference_ms:.0f} ms")
            age_ms = None if packet.capture_time is None else max(0.0,(time.time()-packet.capture_time)*1000.0)
            c6.metric("Tuổi frame", "—" if age_ms is None else f"{age_ms:.0f} ms")
            st.caption(f"Nguồn: `{info['source']}` • camera báo {info.get('stream_fps',0):.1f} FPS • đầu vào {info['width']}×{info['height']} • hiệu chuẩn: {info['calibration'] or 'không dùng'}")
            if info["error"]:
                st.warning(info["error"])
            if packet.frame_rgb is not None:
                st.image(packet.frame_rgb, caption="Khung hình thời gian thực đã đánh dấu", use_container_width=True)
            else:
                st.warning("Chưa nhận được khung hình. Camera có thể đang kết nối hoặc kết nối lại.")

            rows = packet.rows.copy()
            valid = rows[np.isfinite(rows["ground_x"]) & np.isfinite(rows["ground_y"])].copy() if not rows.empty else rows
            source_id = _source_id_single(session)
            if not valid.empty:
                valid["zone"] = assign_zones(valid, ZMODEL)
                valid["source"] = "camera-thoi-gian-thuc"
                counts = {z: int((valid["zone"] == z).sum()) for z in ZONE_ORDER}
                decisions = _log_live_counts(source_id, f"{packet.connection_generation}:{packet.frame_seq}", counts, packet.capture_time)
                left, right = st.columns([1.7, 1])
                with left:
                    st.plotly_chart(ground_figure(valid, "Bản đồ mặt bằng thời gian thực"), use_container_width=True, key="single_live_ground")
                with right:
                    st.markdown("**Hỗ trợ quyết định hiện tại**")
                    for dec in decisions:
                        _status_badge(dec)
                _, live_forecasts, _ = _forecast_for_source(source_id, 10.0)
                with st.expander("🔮 Cảnh báo sớm +5/+10/+15 phút", expanded=False):
                    _forecast_badges(live_forecasts)
            elif packet.frame_seq > 0:
                log_total_snapshot(source_id, f"{packet.connection_generation}:{packet.frame_seq}", len(rows), packet.capture_time)

            history = session.history()
            if not history.empty:
                h1, h2 = st.columns(2)
                h1.download_button(
                    "Tải CSV bám vết thời gian thực", history.to_csv(index=False).encode("utf-8"),
                    "cctv_realtime_tracks.csv", "text/csv", key="single_live_csv",
                )
                snap = session.snapshot_jpeg()
                if snap:
                    h2.download_button("Tải ảnh chụp nhanh JPG", snap, "cctv_snapshot.jpg", "image/jpeg", key="single_live_jpg")
                with st.expander("Các bản ghi bám vết gần nhất"):
                    st.dataframe(viet_df(history.tail(200)), use_container_width=True, hide_index=True)

        render_single_live()

    else:
        st.warning("V6 ưu tiên single-camera. Chỉ chuyển sang hợp nhất hai camera sau khi detector/tracker từng camera đạt tiêu chí TEST riêng; nếu mỗi camera bỏ sót phần lớn người thì fusion không thể sửa được.")
        a1, a2 = st.columns(2)
        with a1:
            url1 = st.text_input("Địa chỉ RTSP/HTTP Camera 1", type="password", key="dual_url1", placeholder="rtsp://...camera1...")
            cal1 = st.selectbox("Hiệu chuẩn Camera 1", list(CAMERA_CALIBRATION.keys()), index=0, format_func=lambda x: CALIB_VI.get(x, x), key="dual_cal1")
        with a2:
            url2 = st.text_input("Địa chỉ RTSP/HTTP Camera 2", type="password", key="dual_url2", placeholder="rtsp://...camera2...")
            cal2 = st.selectbox("Hiệu chuẩn Camera 2", list(CAMERA_CALIBRATION.keys()), index=1, format_func=lambda x: CALIB_VI.get(x, x), key="dual_cal2")
        f1, f2 = st.columns(2)
        live_threshold = f1.slider("Ngưỡng Hungarian (cm)", 25, 200, 100, 5, key="dual_thresh")
        sync_tolerance_ms = f2.slider("Độ lệch đồng bộ tối đa (ms)", 100, 2000, 600, 50, key="dual_sync")
        d1, d2 = st.columns(2)
        if d1.button("▶ Bắt đầu 2 camera", type="primary", disabled=not (url1.strip() and url2.strip()), key="start_dual_live"):
            for k in ["dual_live_1", "dual_live_2"]:
                old = st.session_state.get(k)
                if old is not None:
                    old.stop()
            s1 = RealtimeStream(url1.strip(), "cam1", live_cfg, cal1)
            s2 = RealtimeStream(url2.strip(), "cam2", live_cfg, cal2)
            s1.start(); s2.start()
            st.session_state["dual_live_1"] = s1
            st.session_state["dual_live_2"] = s2
            st.session_state["dual_joint_a"] = pd.DataFrame()
            st.session_state["dual_joint_b"] = pd.DataFrame()
            st.session_state["dual_joint_sample"] = 0
            st.session_state["dual_last_signature"] = None
        if d2.button("■ Dừng 2 camera", key="stop_dual_live"):
            for k in ["dual_live_1", "dual_live_2"]:
                old = st.session_state.pop(k, None)
                if old is not None:
                    old.stop()
            st.success("Đã dừng cả hai camera.")

        @st.fragment(run_every=0.2)
        def render_dual_live():
            s1 = st.session_state.get("dual_live_1")
            s2 = st.session_state.get("dual_live_2")
            if s1 is None or s2 is None:
                st.caption("Nhập 2 địa chỉ rồi nhấn **Bắt đầu 2 camera**.")
                return
            try:
                p1 = s1.analyze_latest()
                p2 = s2.analyze_latest()
            except Exception as e:
                st.error(f"Lỗi suy luận thời gian thực hai camera: {e}")
                return

            sync_ms = None
            if p1.capture_time is not None and p2.capture_time is not None:
                sync_ms = abs(p1.capture_time - p2.capture_time) * 1000.0
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Camera 1", CONNECTION_VI.get(s1.status["status"], s1.status["status"]), f"{len(p1.rows)} người")
            m2.metric("Camera 2", CONNECTION_VI.get(s2.status["status"], s2.status["status"]), f"{len(p2.rows)} người")
            m3.metric("Độ lệch đồng bộ", "—" if sync_ms is None else f"{sync_ms:.0f} ms")
            m4.metric("Số lần kết nối lại", s1.status["reconnect_count"] + s2.status["reconnect_count"])

            v1, v2 = st.columns(2)
            with v1:
                if p1.frame_rgb is not None:
                    st.image(p1.frame_rgb, caption="Camera 1", use_container_width=True)
            with v2:
                if p2.frame_rgb is not None:
                    st.image(p2.frame_rgb, caption="Camera 2", use_container_width=True)

            signature = (p1.frame_seq, p2.frame_seq)
            valid_sync = sync_ms is not None and sync_ms <= float(sync_tolerance_ms)
            a = p1.rows[np.isfinite(p1.rows["ground_x"]) & np.isfinite(p1.rows["ground_y"])].copy() if not p1.rows.empty else p1.rows
            b = p2.rows[np.isfinite(p2.rows["ground_x"]) & np.isfinite(p2.rows["ground_y"])].copy() if not p2.rows.empty else p2.rows
            if valid_sync and signature != st.session_state.get("dual_last_signature"):
                sid = int(st.session_state.get("dual_joint_sample", 0))
                if not a.empty:
                    a["sample_index"] = sid
                if not b.empty:
                    b["sample_index"] = sid
                ha = st.session_state.get("dual_joint_a", pd.DataFrame())
                hb = st.session_state.get("dual_joint_b", pd.DataFrame())
                ha = pd.concat([ha, a], ignore_index=True) if not a.empty else ha
                hb = pd.concat([hb, b], ignore_index=True) if not b.empty else hb
                # Rolling analysis window keeps realtime fusion bounded.
                min_sid = max(0, sid - 119)
                if not ha.empty:
                    ha = ha[ha["sample_index"] >= min_sid].reset_index(drop=True)
                if not hb.empty:
                    hb = hb[hb["sample_index"] >= min_sid].reset_index(drop=True)
                st.session_state["dual_joint_a"] = ha
                st.session_state["dual_joint_b"] = hb
                st.session_state["dual_joint_sample"] = sid + 1
                st.session_state["dual_last_signature"] = signature

            if not valid_sync:
                st.warning("Chưa hợp nhất khung hình hiện tại vì hai luồng video lệch thời gian quá ngưỡng. Việc bám vết từng camera vẫn tiếp tục.")
                return

            ha = st.session_state.get("dual_joint_a", pd.DataFrame())
            hb = st.session_state.get("dual_joint_b", pd.DataFrame())
            if ha.empty and hb.empty:
                st.caption("Chưa có quan sát mặt bằng hợp lệ.")
                return
            obs, matches = fuse_pair(ha, hb, float(live_threshold))
            interval = 1.0 / max(0.2, float(live_fps))
            fusion = summarize_fusion(obs, ZMODEL, interval)
            if fusion.observations.empty:
                st.caption("Chưa tạo được các quan sát đã hợp nhất.")
                return
            sid = int(fusion.observations["sample_index"].max())
            cur = fusion.observations[fusion.observations["sample_index"] == sid].copy()
            if "zone" not in cur.columns:
                cur["zone"] = assign_zones(cur, ZMODEL)
            counts = {z: int((cur["zone"] == z).sum()) for z in ZONE_ORDER}
            dual_source_id = f"2 camera | {s1.status['source']} + {s2.status['source']}"
            dual_decisions = _log_live_counts(dual_source_id, f"{p1.connection_generation}:{p1.frame_seq}|{p2.connection_generation}:{p2.frame_seq}", counts, max(p1.capture_time or 0, p2.capture_time or 0) or None)
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Số người sau hợp nhất", len(cur))
            q2.metric("Cặp ghép hiện tại", int((cur["source"] == "cam1+cam2").sum()))
            q3.metric("ID toàn cục trong cửa sổ", int(fusion.observations["global_id"].nunique()))
            q4.metric("Cửa sổ phân tích", f"{fusion.observations['sample_index'].nunique()} mẫu")
            left, right = st.columns([1.7, 1])
            with left:
                st.plotly_chart(ground_figure(cur, "Hợp nhất mặt bằng CCTV hai camera theo thời gian thực"), use_container_width=True, key="dual_live_ground")
            with right:
                for dec in dual_decisions:
                    _status_badge(dec)
            _, dual_forecasts, _ = _forecast_for_source(dual_source_id, 10.0)
            with st.expander("🔮 Cảnh báo sớm hợp nhất +5/+10/+15 phút", expanded=False):
                _forecast_badges(dual_forecasts)

            with st.expander("⚠️ Bất thường đối tượng + đám đông + hàng đợi sự cố", expanded=True):
                st.caption("V6 kết hợp bất thường quỹ đạo, bất thường occupancy và chính sách điều phối. Cảnh báo chỉ là decision support, không phải nhận dạng danh tính hoặc kết luận an ninh tự động.")
                cfg_live = _anomaly_controls("dual_live_anom")
                person_live, crowd_live, incidents_live = _analyze_anomalies(
                    fusion.observations, fusion.occupancy, dual_decisions, cfg_live
                )
                _render_incidents(person_live, crowd_live, incidents_live, "dual_live")
                sample_key = f"{p1.connection_generation}:{p1.frame_seq}|{p2.connection_generation}:{p2.frame_seq}"
                if not person_live.empty:
                    log_anomaly_alerts(person_live, dual_source_id, sample_key, "PERSON")
                if not crowd_live.empty:
                    log_anomaly_alerts(crowd_live, dual_source_id, sample_key, "CROWD")

            fleft, fright = st.columns(2)
            with fleft:
                st.markdown("**Các chuyển vùng có hướng gần đây**")
                if fusion.transitions.empty:
                    st.caption("Chưa đủ lịch sử để ghi nhận chuyển vùng.")
                else:
                    st.dataframe(viet_df(fusion.transitions.head(20)), use_container_width=True, hide_index=True)
            with fright:
                st.markdown("**Ước lượng thời gian lưu lại gần đây**")
                if fusion.dwell.empty:
                    st.caption("Chưa có dữ liệu thời gian lưu lại.")
                else:
                    ds = fusion.dwell.groupby("zone")["dwell_sec"].agg(["count", "mean", "max"]).reset_index()
                    st.dataframe(viet_df(ds), use_container_width=True, hide_index=True)
            st.download_button(
                "Tải CSV hợp nhất thời gian thực", fusion.observations.to_csv(index=False).encode("utf-8"),
                "dual_cctv_realtime_fusion.csv", "text/csv", key="dual_live_csv",
            )

        render_dual_live()

elif page == "Tải ảnh lên":
    st.title("Tải ảnh lên — phát hiện và phân tích hành khách")
    st.caption("Ảnh đơn không có bám vết theo thời gian; ứng dụng thực hiện phát hiện người, lấy điểm chân và tính tọa độ mặt bằng/khu vực nếu có hiệu chuẩn hợp lệ.")
    uploads = st.file_uploader(
        "Chọn một hoặc nhiều ảnh", type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True, key="image_uploads",
    )
    st.info("V6 mặc định ưu tiên người nhỏ: confidence thấp hơn, ảnh suy luận lớn hơn, max_det cao và có tiled inference. Nếu cần test có kiểm soát hơn, dùng màn hình **Phòng thí nghiệm 1 camera**.")
    ia, ib, ic = st.columns(3)
    img_conf = ia.slider("Độ tin cậy", 0.03, 0.80, 0.10, 0.01, key="img_conf")
    img_size = ib.selectbox("Kích thước ảnh YOLO", [640, 768, 960, 1280, 1536], index=3, key="img_size")
    img_device_ui = ic.selectbox("Thiết bị", ["Tự động", "CPU", "GPU / CUDA"], key="img_device")
    img_device = {"Tự động": "auto", "CPU": "cpu", "GPU / CUDA": "0"}[img_device_ui]
    id1,id2,id3 = st.columns(3)
    img_tiled = id1.checkbox("Tiled inference cho đám đông", value=True, key="img_tiled")
    img_tile_size = id2.selectbox("Kích thước tile", [512,640,768,896,1024], index=2, key="img_tile_size")
    img_max_det = id3.number_input("Số detection tối đa", 50, 3000, 1000, 50, key="img_max_det")
    img_cal_ui = st.selectbox("Hiệu chuẩn mặt bằng (tùy chọn)", ["Không dùng hiệu chuẩn"] + list(CAMERA_CALIBRATION.keys()), format_func=lambda x: "Không dùng hiệu chuẩn" if x == "Không dùng hiệu chuẩn" else CALIB_VI.get(x, x), key="img_cal")
    img_cal = None if img_cal_ui == "Không dùng hiệu chuẩn" else img_cal_ui
    if img_cal:
        st.warning("Hiệu chuẩn chỉ đúng nếu ảnh đến từ đúng camera WILDTRACK tương ứng và không thay đổi vùng cắt hoặc hình học ảnh.")

    if st.button("🔎 Phân tích ảnh", type="primary", disabled=not uploads, key="run_images"):
        if len(uploads) > 10:
            st.warning("Để tránh quá tải UI, V6 xử lý tối đa 10 ảnh mỗi lượt.")
        try:
            with st.spinner("Đang chạy YOLO11n trên ảnh..."):
                model = load_image_model()
                cfg = ImageConfig(
                    confidence=float(img_conf), image_size=int(img_size), device=img_device,
                    max_det=int(img_max_det), tiled=bool(img_tiled), tile_size=int(img_tile_size),
                    tile_overlap=0.20, half=True,
                )
                runs = []
                for up in uploads[:10]:
                    run = process_image_bytes(up.getvalue(), up.name, cfg, img_cal, model=model)
                    if img_cal and not run.detections.empty:
                        valid_mask = np.isfinite(run.detections["ground_x"]) & np.isfinite(run.detections["ground_y"])
                        run.detections["zone"] = ""
                        if valid_mask.any():
                            run.detections.loc[valid_mask, "zone"] = assign_zones(run.detections.loc[valid_mask], ZMODEL).values
                    runs.append(run)
                st.session_state["image_runs"] = runs
        except Exception as e:
            st.exception(e)

    runs = st.session_state.get("image_runs", [])
    if runs:
        tabs = st.tabs([r.name[:35] for r in runs])
        for idx, (tab, run) in enumerate(zip(tabs, runs)):
            with tab:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Số người phát hiện", len(run.detections))
                mean_conf = run.detections["confidence"].mean() if not run.detections.empty else np.nan
                c2.metric("Độ tin cậy trung bình", "—" if not np.isfinite(mean_conf) else f"{mean_conf:.3f}")
                c3.metric("Thời gian AI", f"{run.inference_ms:.0f} ms")
                c4.metric("Số tile", run.tiles_processed)
                zone_count = int((run.detections.get("zone", pd.Series(dtype=str)) != "").sum()) if "zone" in run.detections.columns else 0
                st.caption(f"Thiết bị: {run.device_used} • điểm mặt bằng/khu vực hợp lệ: {zone_count}")
                left, right = st.columns(2)
                with left:
                    st.image(run.original_rgb, caption="Ảnh gốc", use_container_width=True)
                with right:
                    st.image(run.annotated_rgb, caption="YOLO11n phát hiện người + điểm chân", use_container_width=True)
                if not run.detections.empty:
                    st.dataframe(viet_df(run.detections), use_container_width=True, hide_index=True)
                    if "zone" in run.detections.columns and (run.detections["zone"] != "").any():
                        g = run.detections[run.detections["zone"] != ""].copy()
                        g["source"] = "anh"
                        st.plotly_chart(ground_figure(g, f"Bản đồ mặt bằng — {run.name}"), use_container_width=True, key=f"img_ground_{idx}")
                        counts = {z: int((g["zone"] == z).sum()) for z in ZONE_ORDER}
                        dc = st.columns(3)
                        for col, dec in zip(dc, _decisions(counts)):
                            with col:
                                _status_badge(dec)
                dl1, dl2 = st.columns(2)
                dl1.download_button("Tải ảnh đã đánh dấu", run.annotated_jpeg, f"annotated_{Path(run.name).stem}.jpg", "image/jpeg", key=f"img_jpg_{idx}")
                dl2.download_button("Tải CSV phát hiện", run.detections.to_csv(index=False).encode("utf-8"), f"detections_{Path(run.name).stem}.csv", "text/csv", key=f"img_csv_{idx}")

elif page == "Bản đồ mặt bằng":
    st.title("Bản đồ mặt bằng và hợp nhất đa camera")
    idx = _frame_controls("map_frame")
    current = PRED_TEST[PRED_TEST["gt_index"] == idx]
    gt_cur = GT_TEST[GT_TEST["gt_index"] == idx]

    a, b, c = st.columns(3)
    a.metric("Quan sát chỉ từ Camera 1", int((current["source"] == "cam1").sum()))
    b.metric("Quan sát chỉ từ Camera 2", int((current["source"] == "cam2").sum()))
    c.metric("Quan sát hợp nhất hai camera", int((current["source"] == "cam1+cam2").sum()))

    tab1, tab2 = st.tabs(["Dự đoán", "Nhãn thật GT"])
    with tab1:
        st.plotly_chart(ground_figure(current, "Dự đoán: quan sát mặt bằng đã hợp nhất"), use_container_width=True)
    with tab2:
        gt_view = gt_cur.assign(source="GT", cam1_track_id=np.nan, cam2_track_id=np.nan)
        st.plotly_chart(ground_figure(gt_view, "Nhãn thật tham chiếu của tập TEST tách riêng"), use_container_width=True)
    st.caption("Khu A/B/C là các vùng không gian tổng quát của WILDTRACK, không được tự gán thành khu làm thủ tục, an ninh hoặc cửa ra máy bay.")

elif page == "Phân tích khu vực":
    st.title("Phân tích số người theo khu vực")
    st.caption("Phân vùng PCA chỉ được học từ dữ liệu nhãn thật của tập phát triển và tập xác thực; tập TEST không được dùng để học phân vùng.")

    pred_long = PRED_OCC.reset_index().melt(id_vars="gt_index", var_name="zone", value_name="Pred")
    gt_long = GT_OCC.reset_index().melt(id_vars="gt_index", var_name="zone", value_name="GT")
    both = pred_long.merge(gt_long, on=["gt_index", "zone"])
    both["time_sec"] = both["gt_index"] * 0.5

    zone = st.selectbox("Chọn khu vực", ZONE_ORDER, format_func=lambda x: ZONE_VI.get(x, x))
    z = both[both["zone"] == zone]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z["time_sec"], y=z["GT"], name="Nhãn thật GT", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=z["time_sec"], y=z["Pred"], name="Dự đoán", mode="lines+markers"))
    fig.update_layout(height=430, xaxis_title="Thời gian (giây)", yaxis_title="Số người hiện diện", title=f"{ZONE_VI.get(zone, zone)}: số người dự đoán so với nhãn thật GT")
    st.plotly_chart(fig, use_container_width=True)

    summary = D["zone_results"][(D["zone_results"]["pair"] == "Pair12") & (D["zone_results"]["system"] == "Hungarian Equal")].copy()
    st.dataframe(viet_df(summary[["zone", "timestamps", "GT_mean_occupancy", "Pred_mean_occupancy", "MAE", "RMSE", "Bias", "WAPE"]]), use_container_width=True, hide_index=True)

elif page == "Dự báo & cảnh báo sớm":
    st.title("Dự báo ùn tắc và cảnh báo sớm 5–10–15 phút")
    st.caption("V6 dùng lịch sử realtime đã lưu trong SQLite để dự báo tải tương lai. Đây là baseline dự báo minh bạch phục vụ nguyên mẫu; chưa phải mô hình dự báo sân bay đã được huấn luyện và kiểm định thực địa.")

    sources = list_sources()
    if not sources:
        st.info("Cơ sở dữ liệu chưa có lịch sử CCTV. Hãy chạy camera realtime có hiệu chuẩn để ghi Khu A/B/C, hoặc tạo dữ liệu mô phỏng chỉ để kiểm thử giao diện V6.")
        if st.button("Tạo 20 phút dữ liệu MÔ PHỎNG để thử V6", type="primary"):
            n = seed_demo_history(CAP)
            st.success(f"Đã tạo {n} bản ghi mô phỏng. Đây KHÔNG phải dữ liệu thực nghiệm.")
            st.rerun()
    else:
        source = st.selectbox("Nguồn lịch sử", sources, index=len(sources)-1)
        a, b = st.columns(2)
        window_min = a.slider("Cửa sổ lịch sử dùng để ước lượng xu hướng (phút)", 2, 30, 10, 1)
        show_min = b.slider("Hiển thị lịch sử gần nhất (phút)", 10, 180, 60, 10)
        hist = read_metrics(source=source, since_minutes=float(show_min))
        zone_hist = hist[hist["zone"].isin(ZONE_ORDER)].copy() if not hist.empty else hist

        if zone_hist.empty:
            st.warning("Nguồn này hiện chỉ có tổng số người (camera chưa hiệu chuẩn), vì vậy chưa thể dự báo theo Khu A/B/C để điều hướng. Hãy chọn camera có hiệu chuẩn mặt bằng.")
            if not hist.empty:
                chart = hist.copy()
                chart["Thời gian"] = pd.to_datetime(chart["recorded_at"], unit="s", utc=True).dt.tz_convert("Asia/Ho_Chi_Minh")
                fig = px.line(chart, x="Thời gian", y="occupancy", title="Tổng số người theo thời gian", labels={"occupancy": "Số người"})
                st.plotly_chart(fig, use_container_width=True)
        else:
            forecasts, meta = forecast_all(
                zone_hist, CAP, ZONE_ORDER, horizons=(5, 10, 15), window_minutes=float(window_min),
                busy_threshold=busy, congested_threshold=congested,
            )
            latest = zone_hist.sort_values("recorded_at").groupby("zone", as_index=False).tail(1)
            mcols = st.columns(3)
            for col, zone in zip(mcols, ZONE_ORDER):
                r = latest[latest["zone"] == zone]
                cur = int(r["occupancy"].iloc[-1]) if not r.empty else 0
                with col:
                    st.metric(ZONE_VI[zone], f"{cur} người", f"{cur / max(1, CAP[zone]):.0%} sức chứa")

            chart = zone_hist.copy()
            chart["Thời gian"] = pd.to_datetime(chart["recorded_at"], unit="s", utc=True).dt.tz_convert("Asia/Ho_Chi_Minh")
            chart["Khu vực"] = chart["zone"].map(ZONE_VI)
            fig = px.line(chart, x="Thời gian", y="occupancy", color="Khu vực", markers=True, title="Lịch sử số người theo khu vực", labels={"occupancy": "Số người"})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Cảnh báo sớm")
            _forecast_badges(forecasts)

            if not forecasts.empty:
                view = forecasts.copy()
                view["zone"] = view["zone"].map(lambda x: ZONE_VI.get(x, x))
                view["status"] = view["status"].map(lambda x: STATUS_VI.get(x, x))
                view["action"] = view["action"].map(lambda x: ACTION_VI.get(x, x))
                view["alternative"] = view["alternative"].map(lambda x: ZONE_VI.get(x, x) if x else "—")
                st.subheader("Bảng dự báo +5 / +10 / +15 phút")
                st.dataframe(viet_df(view[["zone", "horizon_min", "current", "forecast", "lower", "upper", "capacity", "predicted_utilization", "status", "action", "alternative", "slope_people_per_min", "reliability"]]), use_container_width=True, hide_index=True)

                fig2 = go.Figure()
                for zone in ZONE_ORDER:
                    fz = forecasts[forecasts["zone"] == zone]
                    if fz.empty:
                        continue
                    fig2.add_trace(go.Scatter(x=fz["horizon_min"], y=fz["forecast"], mode="lines+markers", name=ZONE_VI[zone]))
                fig2.update_layout(title="Số người dự báo theo tầm thời gian", xaxis_title="Phút kể từ hiện tại", yaxis_title="Số người dự báo", height=430)
                st.plotly_chart(fig2, use_container_width=True)

                st.subheader("Đánh giá hồi cứu tự động của baseline dự báo")
                backtest = backtest_forecasts(
                    zone_hist, CAP, ZONE_ORDER, horizons=(5, 10, 15), window_minutes=float(window_min),
                    busy_threshold=busy, congested_threshold=congested,
                )
                if backtest.empty:
                    st.caption("Chưa đủ lịch sử liên tục để backtest các tầm +5/+10/+15 phút. Thường cần hơn 15–20 phút dữ liệu.")
                else:
                    bv = backtest.copy()
                    bv["zone"] = bv["zone"].map(lambda x: ZONE_VI.get(x, x))
                    st.dataframe(viet_df(bv), use_container_width=True, hide_index=True)
                    st.caption("MAE/RMSE/WAPE ở đây so dự báo với occupancy tương lai do chính pipeline realtime ghi nhận; đây chưa phải đánh giá forecast trên ground truth người thật.")

                st.markdown("**Chất lượng lịch sử đầu vào**")
                quality = []
                for zone in ZONE_ORDER:
                    m = meta.get(zone)
                    quality.append({"Khu vực": ZONE_VI[zone], "Số điểm": m.points if m else 0, "Độ dài lịch sử (phút)": round(m.duration_minutes, 2) if m else 0, "Mức tin cậy": m.reliability if m else "THẤP", "Phương pháp": m.method if m else "—"})
                st.dataframe(pd.DataFrame(quality), use_container_width=True, hide_index=True)

                alert_rows = forecasts[forecasts["status"].isin(["BUSY", "CONGESTED"])].copy()
                if st.button("Lưu các cảnh báo dự báo hiện tại vào cơ sở dữ liệu", disabled=alert_rows.empty):
                    n = log_alerts(alert_rows, source)
                    st.success(f"Đã lưu {n} cảnh báo dự báo.")

            st.warning("Khoảng cận dưới/cận trên là bất định heuristic của prototype, KHÔNG phải khoảng tin cậy thống kê đã được hiệu chuẩn. Khi viết báo cáo, hãy gọi đây là baseline dự báo V6 hoặc mô-đun cảnh báo sớm nguyên mẫu.")


elif page == "Điều hướng":
    st.title("Nguyên mẫu hỗ trợ quyết định và điều hướng dòng người")
    st.caption("Mô-đun này chuyển số người hiện diện theo từng khu thành khuyến nghị GIỮ NGUYÊN / HẠN CHẾ / ĐIỀU HƯỚNG LẠI bằng chính sách minh bạch; có thể thay bằng mô hình tối ưu hóa ở giai đoạn sau.")
    idx = _frame_controls("route_frame")
    counts = _current_counts(idx)
    decisions = _decisions(counts)

    cols = st.columns(3)
    for col, dec in zip(cols, decisions):
        with col:
            _status_badge(dec)

    util = pd.DataFrame({
        "zone": ZONE_ORDER,
        "occupancy": [counts[z] for z in ZONE_ORDER],
        "capacity": [CAP[z] for z in ZONE_ORDER],
    })
    util["utilization"] = util["occupancy"] / util["capacity"]
    util["khu_vuc"] = util["zone"].map(lambda x: ZONE_VI.get(x, x))
    fig = px.bar(util, x="khu_vuc", y="utilization", text=util["utilization"].map(lambda x: f"{x:.0%}"), title="Mức sử dụng sức chứa theo chính sách đang cấu hình", labels={"khu_vuc": "Khu vực", "utilization": "Mức sử dụng"})
    fig.add_hline(y=busy, line_dash="dash", annotation_text="ĐÔNG")
    fig.add_hline(y=congested, line_dash="dash", annotation_text="ÙN TẮC")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Độ tin cậy của luồng trên tập TEST tách riêng")
    flow = D["flow_results"].copy()
    st.dataframe(viet_df(flow[["pair", "GT_transition_events", "Pred_transition_events", "Transition_F1", "Transition_Count_WAPE", "Transition_Distribution_TV", "Temporal_Transition_Count_WAPE"]]), use_container_width=True, hide_index=True)
    st.warning("ĐIỀU HƯỚNG LẠI ở đây là quyết định hỗ trợ dựa trên số người hiện diện và sức chứa cấu hình. F1/WAPE của luồng phía trên là kết quả thực nghiệm; không được suy diễn rằng chính sách điều hướng đã được kiểm chứng ngoài thực địa.")

elif page == "Bất thường & cảnh báo":
    st.title("Bất thường & cảnh báo — V6")
    st.caption("Hợp nhất bất thường cấp đối tượng/quỹ đạo, bất thường cấp đám đông và khuyến nghị điều phối thành một hàng đợi sự cố ưu tiên.")
    st.info("Phạm vi hiện tại: loitering, tốc độ mặt bằng cao, vùng hạn chế, hướng cấm, đổi hướng đột ngột (experimental), quá tải, tăng nhanh và ùn tắc kéo dài. Hệ thống không nhận dạng danh tính và không tự kết luận hành vi nguy hiểm.")

    tab_now, tab_history, tab_scope = st.tabs(["Phân tích phiên hiện tại", "Lịch sử cảnh báo", "Phạm vi & giới hạn"])
    with tab_now:
        cfg_global = _anomaly_controls("global_anom")
        pair_now = st.session_state.get("pair_run")
        if pair_now is not None and pair_now.get("obs") is not None and not pair_now["obs"].empty:
            obs_now = pair_now["obs"].copy()
            occ_now = pair_now.get("occupancy", pd.DataFrame()).copy()
            if not occ_now.empty:
                sid_now = int(occ_now["sample_index"].max())
                rr = occ_now[occ_now["sample_index"] == sid_now]
                counts_now = {z: int(rr.iloc[0].get(z, 0)) if not rr.empty else 0 for z in ZONE_ORDER}
            else:
                counts_now = {z: 0 for z in ZONE_ORDER}
            dec_now = _decisions(counts_now)
            p_alert, c_alert, queue = _analyze_anomalies(obs_now, occ_now, dec_now, cfg_global)
            _render_incidents(p_alert, c_alert, queue, "global_pair_session")
            if st.button("Lưu cảnh báo phiên video hiện tại", disabled=queue.empty, key="save_global_pair_alerts"):
                key = f"offline-pair-{int(time.time())}"
                n = log_anomaly_alerts(p_alert, "Video Pair12 offline", key, "PERSON") + log_anomaly_alerts(c_alert, "Video Pair12 offline", key, "CROWD")
                st.success(f"Đã lưu {n} cảnh báo mới vào SQLite.")
        else:
            st.warning("Chưa có phiên video Pair12 trong bộ nhớ. Bạn có thể vào **Phân tích video → 2 camera** để chạy một phiên; bên dưới là minh họa crowd-level bằng occupancy TEST WILDTRACK, không có cảnh báo cá nhân vì file TEST tổng hợp không chứa trajectory global ID đầy đủ.")
            occ_demo = PRED_OCC.copy().reset_index()
            if "gt_index" in occ_demo.columns:
                occ_demo = occ_demo.rename(columns={"gt_index": "sample_index"})
            elif "index" in occ_demo.columns:
                occ_demo = occ_demo.rename(columns={"index": "sample_index"})
            c_demo = detect_crowd_anomalies(
                occ_demo, CAP, busy_threshold=busy, congested_threshold=congested,
                surge_people=cfg_global["surge_people"], surge_window_points=cfg_global["surge_window_points"],
                persistent_points=cfg_global["persistent_points"],
            )
            latest_idx = int(max(TEST_IDS))
            demo_dec = _decisions(_current_counts(latest_idx))
            queue_demo = build_incident_table(pd.DataFrame(), c_demo, demo_dec)
            _render_incidents(pd.DataFrame(columns=["global_id","anomaly_type","severity","zone","time_sec","value","threshold","message"]), c_demo, queue_demo, "wildtrack_crowd_demo")

    with tab_history:
        ah = read_anomaly_alerts(limit=10000)
        if ah.empty:
            st.caption("Chưa có cảnh báo bất thường nào được lưu. Cảnh báo realtime 2-camera sẽ được ghi tự động khi phát sinh.")
        else:
            h1, h2, h3 = st.columns(3)
            h1.metric("Tổng bản ghi", len(ah))
            h2.metric("HIGH/CRITICAL", int(ah["severity"].isin(["HIGH", "CRITICAL"]).sum()))
            h3.metric("Nguồn", int(ah["source"].nunique()))
            srcs = ["Tất cả"] + sorted(ah["source"].dropna().astype(str).unique().tolist())
            sel = st.selectbox("Nguồn cảnh báo", srcs, key="anom_history_source")
            view = ah if sel == "Tất cả" else ah[ah["source"] == sel]
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button("Tải toàn bộ lịch sử cảnh báo CSV", ah.to_csv(index=False).encode("utf-8"), "anomaly_alert_history.csv", "text/csv", key="anom_hist_csv")

    with tab_scope:
        st.markdown("""
**Đã có trong V6**
- Cấp cá nhân/quỹ đạo: lưu lại quá lâu, tốc độ bất thường, vào vùng hạn chế, đi sai hướng, đổi hướng đột ngột (tùy chọn).
- Cấp đám đông: mức sử dụng cao, vượt sức chứa cấu hình, tăng nhanh, ùn tắc kéo dài.
- Hỗ trợ quyết định: ghép cảnh báo với GIỮ NGUYÊN / HẠN CHẾ / ĐIỀU HƯỚNG LẠI và tạo hàng đợi sự cố.
- Lưu SQLite + xuất CSV để phục vụ kiểm tra và báo cáo.

**Chưa được tuyên bố là đã giải quyết**
- Nhận diện hành động phức tạp như đánh nhau, té ngã, hành vi phạm pháp hoặc nhận dạng khuôn mặt. Những bài toán này cần dữ liệu gán nhãn và mô hình chuyên biệt riêng.
- Ngưỡng vận hành/an toàn thực tế tại sân bay. Các ngưỡng hiện tại là tham số nghiên cứu cần xác thực bằng VAA-2Cam.
""")

elif page == "Lịch sử & cơ sở dữ liệu":
    st.title("Lịch sử thời gian thực và cơ sở dữ liệu SQLite")
    stats = db_stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Bản ghi số người", stats["metric_rows"])
    c2.metric("Cảnh báo forecast", stats["alert_rows"])
    c3.metric("Cảnh báo bất thường", stats.get("anomaly_alert_rows", 0))
    c4.metric("Dung lượng CSDL", f"{stats['size_bytes'] / 1024:.1f} KB")
    c5.metric("Số nguồn", len(list_sources()))
    st.caption(f"Tệp cơ sở dữ liệu: `{stats['path']}`")

    sources = ["Tất cả"] + list_sources()
    src = st.selectbox("Lọc theo nguồn", sources)
    hours = st.slider("Khoảng thời gian hiển thị gần nhất (giờ)", 1, 72, 24)
    hist = read_metrics(source=None if src == "Tất cả" else src, since_minutes=float(hours * 60))
    if hist.empty:
        st.info("Chưa có dữ liệu lịch sử. Khi CCTV realtime chạy, V6 sẽ tự lưu snapshot vào SQLite.")
    else:
        h = hist.copy()
        h["Thời gian"] = pd.to_datetime(h["recorded_at"], unit="s", utc=True).dt.tz_convert("Asia/Ho_Chi_Minh")
        h["Khu vực"] = h["zone"].map(lambda x: ZONE_VI.get(x, "Tổng" if x == "TOTAL" else x))
        fig = px.line(h, x="Thời gian", y="occupancy", color="Khu vực", facet_row="source" if src == "Tất cả" and h["source"].nunique() <= 3 else None, title="Lịch sử số người đã lưu", labels={"occupancy": "Số người"})
        st.plotly_chart(fig, use_container_width=True)
        display_cols = [c for c in ["recorded_at_iso", "source", "zone", "occupancy", "capacity", "utilization", "status", "action", "sample_key"] if c in hist.columns]
        shown = hist[display_cols].copy()
        if "zone" in shown: shown["zone"] = shown["zone"].map(lambda x: ZONE_VI.get(x, "Tổng" if x == "TOTAL" else x))
        if "status" in shown: shown["status"] = shown["status"].map(lambda x: STATUS_VI.get(x, x))
        if "action" in shown: shown["action"] = shown["action"].map(lambda x: ACTION_VI.get(x, x))
        st.dataframe(viet_df(shown.tail(1000)), use_container_width=True, hide_index=True, height=420)
        st.download_button("Tải lịch sử CSV", hist.to_csv(index=False).encode("utf-8"), "crowdflow_v6_history.csv", "text/csv")

    alerts = read_alerts(500)
    with st.expander("Cảnh báo dự báo đã lưu"):
        if alerts.empty:
            st.caption("Chưa lưu cảnh báo dự báo nào.")
        else:
            av = alerts.copy()
            av["zone"] = av["zone"].map(lambda x: ZONE_VI.get(x, x))
            av["status"] = av["status"].map(lambda x: STATUS_VI.get(x, x))
            av["action"] = av["action"].map(lambda x: ACTION_VI.get(x, x))
            st.dataframe(viet_df(av), use_container_width=True, hide_index=True)
            st.download_button("Tải cảnh báo CSV", alerts.to_csv(index=False).encode("utf-8"), "crowdflow_v6_forecast_alerts.csv", "text/csv")

    anomaly_hist = read_anomaly_alerts(source=None if src == "Tất cả" else src, limit=2000)
    with st.expander("Cảnh báo bất thường đã lưu"):
        if anomaly_hist.empty:
            st.caption("Chưa có cảnh báo bất thường nào trong bộ lọc hiện tại.")
        else:
            st.dataframe(anomaly_hist, use_container_width=True, hide_index=True)
            st.download_button("Tải cảnh báo bất thường CSV", anomaly_hist.to_csv(index=False).encode("utf-8"), "crowdflow_v6_anomaly_alerts.csv", "text/csv", key="history_anom_csv")

    db_file = Path(DEFAULT_DB_PATH)
    if db_file.exists():
        st.download_button("Tải toàn bộ cơ sở dữ liệu SQLite", db_file.read_bytes(), "crowdflow_v6.sqlite", "application/vnd.sqlite3")

    st.divider()
    st.subheader("Công cụ kiểm thử")
    x1, x2 = st.columns(2)
    if x1.button("Tạo dữ liệu mô phỏng 20 phút"):
        n = seed_demo_history(CAP)
        st.success(f"Đã tạo {n} bản ghi mô phỏng để thử forecast. Nguồn được gắn nhãn rõ là KHÔNG PHẢI DỮ LIỆU THỰC.")
        st.rerun()
    confirm_clear = x2.checkbox("Tôi hiểu thao tác xóa sẽ xóa lịch sử V6 trên máy này")
    if x2.button("Xóa toàn bộ lịch sử V6", disabled=not confirm_clear):
        clear_database()
        st.success("Đã xóa dữ liệu lịch sử và cảnh báo trong SQLite.")
        st.rerun()


elif page == "Đánh giá khoa học":
    st.title("Đánh giá khoa học")
    score = D["scorecard"].copy()
    st.subheader("Bảng tổng hợp cấu hình")
    st.dataframe(viet_df(score), use_container_width=True, hide_index=True)

    m1, m2, m3, m4 = st.columns(4)
    p12 = score[score["pair"] == "Pair12"].iloc[0]
    m1.metric("Pair12 F1 mặt bằng", f"{p12['Baseline_Ground_F1']:.4f}")
    m2.metric("Pair12 IDF1 bám vết", f"{p12['Tracking_IDF1']:.4f}")
    m3.metric("Pair12 F1 Vào/Ra", f"{p12['EntryExit_F1']:.4f}")
    m4.metric("Pair12 F1 sự kiện luồng", f"{p12['Flow_Event_F1']:.4f}")

    st.subheader("Bám vết")
    st.dataframe(viet_df(D["tracking_results"][["pair", "detection_precision", "detection_recall", "detection_f1", "matched_ground_MAE_cm", "IDF1", "MOTA", "ID_switches", "predicted_trajectories"]]), use_container_width=True, hide_index=True)

    st.subheader("Vào/Ra, thời gian lưu lại và luồng")
    t1, t2, t3 = st.tabs(["Vào/Ra", "Thời gian lưu lại", "Luồng"])
    with t1:
        st.dataframe(viet_df(D["entry_exit_results"]), use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(viet_df(D["dwell_results"]), use_container_width=True, hide_index=True)
    with t3:
        st.dataframe(viet_df(D["flow_results"]), use_container_width=True, hide_index=True)

    st.subheader("Lưu ý khoa học")
    CAUTION_VI = {
        "Raw occupancy-error correlation must not be interpreted without crowd-size control.": "Không được diễn giải tương quan thô giữa sai số số người và quy mô đám đông nếu chưa kiểm soát quy mô đám đông.",
        "Ground MAE must be interpreted with Recall/F1.": "MAE mặt bằng phải được diễn giải cùng Recall/F1.",
        "Dwell MAE is conditional on matched complete visits.": "MAE thời gian lưu lại chỉ được tính có điều kiện trên các lượt hoàn chỉnh đã ghép khớp.",
        "Total flow count alone is insufficient; direction composition must also be evaluated.": "Chỉ tổng số luồng là chưa đủ; cần đánh giá cả thành phần theo hướng di chuyển.",
        "Generic WILDTRACK zones must not be assigned airport semantics.": "Không được gán ý nghĩa nghiệp vụ sân bay cho các khu vực tổng quát của WILDTRACK.",
    }
    cautions = D["protocol"].get("important_interpretation_rules", [])
    for i, text in enumerate(cautions, 1):
        st.markdown(f"**{i}.** {CAUTION_VI.get(text, text)}")
    st.error("Điểm yếu cần trình bày rõ khi bảo vệ: tính liên tục danh tính và khả năng khôi phục lượt lưu lại còn thấp; Pair34 yếu hơn Pair12/Pair56. Ứng dụng là nguyên mẫu nghiên cứu, chưa phải hệ thống an toàn vận hành thực tế.")

elif page == "Dữ liệu":
    st.title("Khám phá dữ liệu nghiên cứu")
    table_name = st.selectbox("Bộ dữ liệu", ["Pair12 TEST đã hợp nhất", "Pair12 TEST nhãn thật GT", "Kết quả khu vực", "Kết quả bám vết", "Kết quả Vào/Ra", "Kết quả thời gian lưu lại", "Kết quả luồng", "Bảng tổng hợp"])
    mapping = {
        "Pair12 TEST đã hợp nhất": PRED_TEST,
        "Pair12 TEST nhãn thật GT": GT_TEST,
        "Kết quả khu vực": D["zone_results"],
        "Kết quả bám vết": D["tracking_results"],
        "Kết quả Vào/Ra": D["entry_exit_results"],
        "Kết quả thời gian lưu lại": D["dwell_results"],
        "Kết quả luồng": D["flow_results"],
        "Bảng tổng hợp": D["scorecard"],
    }
    view = mapping[table_name]
    st.dataframe(viet_df(view), use_container_width=True, hide_index=True, height=520)
    st.download_button("Tải CSV", view.to_csv(index=False).encode("utf-8"), file_name=table_name.lower().replace(" ", "_") + ".csv", mime="text/csv")

st.divider()
st.caption("VAA FlowVision CrowdFlow V6 • Nhánh thí nghiệm WILDTRACK • Suy luận trực tiếp từ ảnh/video/CCTV và kiểm tra kết quả có thể tái lập • Không phải hệ thống an toàn sân bay dùng trong vận hành thực tế.")
