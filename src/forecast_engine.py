from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

from src.decision_engine import classify, DEFAULT_ADJACENCY


@dataclass(frozen=True)
class ForecastMeta:
    points: int
    duration_minutes: float
    reliability: str
    method: str


def _reliability(points: int, duration_min: float) -> str:
    if points >= 30 and duration_min >= 5:
        return "KHÁ"
    if points >= 12 and duration_min >= 2:
        return "TRUNG BÌNH"
    return "THẤP"


def _prepare_zone(df: pd.DataFrame, zone: str, window_minutes: float) -> pd.DataFrame:
    z = df[df["zone"] == zone].copy()
    if z.empty:
        return z
    z = z.sort_values("recorded_at").drop_duplicates(subset=["recorded_at"], keep="last")
    cutoff = float(z["recorded_at"].max()) - float(window_minutes) * 60.0
    z = z[z["recorded_at"] >= cutoff].copy()
    return z


def forecast_zone(
    history: pd.DataFrame,
    zone: str,
    capacity: int,
    horizons: tuple[int, ...] = (5, 10, 15),
    window_minutes: float = 10.0,
    busy_threshold: float = 0.65,
    congested_threshold: float = 0.85,
) -> tuple[pd.DataFrame, ForecastMeta]:
    """Baseline minh bạch: mức hiện tại + xu hướng tuyến tính co về persistence.

    Đây là mô-đun forecast prototype, không phải model đã train/validate trên dữ liệu sân bay.
    """
    z = _prepare_zone(history, zone, window_minutes)
    if z.empty:
        return pd.DataFrame(), ForecastMeta(0, 0.0, "THẤP", "Không đủ dữ liệu")

    y = z["occupancy"].to_numpy(float)
    t = (z["recorded_at"].to_numpy(float) - float(z["recorded_at"].iloc[-1])) / 60.0
    duration = float((z["recorded_at"].max() - z["recorded_at"].min()) / 60.0) if len(z) > 1 else 0.0
    points = int(len(z))
    last = float(y[-1])

    # EMA giúp giảm nhảy do detector; trend chỉ fit khi đủ điểm/thời gian.
    ema = pd.Series(y).ewm(span=min(max(3, points // 3), 15), adjust=False).mean().to_numpy(float)
    slope = 0.0
    intercept = float(ema[-1])
    residual_std = max(0.5, float(np.std(y - ema))) if points > 1 else 1.0
    if points >= 4 and duration >= 0.25 and np.ptp(t) > 1e-6:
        slope_raw, intercept_raw = np.polyfit(t, ema, 1)
        # Chặn xu hướng cực đoan: tối đa ±50% sức chứa/phút.
        slope_cap = max(1.0, float(capacity) * 0.5)
        slope = float(np.clip(slope_raw, -slope_cap, slope_cap))
        intercept = float(intercept_raw)
        fit = slope * t + intercept
        residual_std = max(0.5, float(np.std(y - fit)))

    # Shrink trend về persistence để horizon xa không bùng nổ quá nhanh.
    shrink = min(0.70, max(0.20, duration / max(1.0, window_minutes)))
    now_level = 0.65 * last + 0.35 * float(ema[-1])
    reliability = _reliability(points, duration)

    rows = []
    for h in horizons:
        pred = max(0.0, now_level + shrink * slope * float(h))
        # Khoảng bất định heuristic cho prototype; không gọi là CI thống kê đã hiệu chuẩn.
        uncertainty = residual_std * math.sqrt(1.0 + float(h) / max(1.0, duration + 1.0))
        low = max(0.0, pred - 1.64 * uncertainty)
        high = max(low, pred + 1.64 * uncertainty)
        util = pred / max(1, int(capacity))
        status, action = classify(util, busy_threshold, congested_threshold)
        rows.append({
            "zone": zone,
            "horizon_min": int(h),
            "current": last,
            "forecast": float(pred),
            "lower": float(low),
            "upper": float(high),
            "capacity": int(capacity),
            "predicted_utilization": float(util),
            "status": status,
            "action": action,
            "slope_people_per_min": float(slope),
            "reliability": reliability,
        })
    return pd.DataFrame(rows), ForecastMeta(points, duration, reliability, "EMA + xu hướng tuyến tính co về persistence")


def forecast_all(
    history: pd.DataFrame,
    capacities: dict[str, int],
    zones: list[str] | tuple[str, ...],
    horizons: tuple[int, ...] = (5, 10, 15),
    window_minutes: float = 10.0,
    busy_threshold: float = 0.65,
    congested_threshold: float = 0.85,
    adjacency: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, ForecastMeta]]:
    adjacency = adjacency or DEFAULT_ADJACENCY
    chunks, meta = [], {}
    for zone in zones:
        f, m = forecast_zone(
            history, zone, capacities.get(zone, 1), horizons, window_minutes,
            busy_threshold, congested_threshold,
        )
        meta[zone] = m
        if not f.empty:
            chunks.append(f)
    if not chunks:
        return pd.DataFrame(), meta
    out = pd.concat(chunks, ignore_index=True)

    # Chọn khu thay thế tại từng horizon dựa trên tải dự báo thấp nhất trong các khu kề.
    out["alternative"] = ""
    out["message"] = ""
    for h in horizons:
        hmask = out["horizon_min"] == h
        util_map = dict(zip(out.loc[hmask, "zone"], out.loc[hmask, "predicted_utilization"]))
        for idx in out.index[hmask]:
            r = out.loc[idx]
            zone = str(r["zone"])
            action = str(r["action"])
            alternative = ""
            if action == "REDIRECT":
                candidates = [z for z in adjacency.get(zone, []) if z in util_map]
                if candidates:
                    alternative = min(candidates, key=lambda z: util_map.get(z, 999.0))
            out.at[idx, "alternative"] = alternative
            if action == "KEEP":
                msg = f"{zone}: dự báo +{h} phút vẫn dưới ngưỡng đông; giữ nguyên hướng di chuyển."
            elif action == "RESTRICT":
                msg = f"{zone}: dự báo +{h} phút tiến gần sức chứa; nên hạn chế dòng đi vào và theo dõi sát."
            elif alternative:
                msg = f"{zone}: dự báo +{h} phút vượt ngưỡng ùn tắc; ưu tiên điều hướng dòng mới sang {alternative}."
            else:
                msg = f"{zone}: dự báo +{h} phút vượt ngưỡng ùn tắc; chưa có khu lân cận thay thế phù hợp."
            out.at[idx, "message"] = msg
    return out, meta


def backtest_forecasts(
    history: pd.DataFrame,
    capacities: dict[str, int],
    zones: list[str] | tuple[str, ...],
    horizons: tuple[int, ...] = (5, 10, 15),
    window_minutes: float = 10.0,
    busy_threshold: float = 0.65,
    congested_threshold: float = 0.85,
    anchor_step_seconds: int = 60,
    target_tolerance_seconds: int = 45,
) -> pd.DataFrame:
    """Backtest forecast với occupancy tương lai đã quan sát trong chính hệ thống.

    Lưu ý: đây là consistency/backtest của forecast trên occupancy do pipeline sinh ra,
    không thay thế đánh giá với ground-truth người thật.
    """
    if history is None or history.empty:
        return pd.DataFrame()
    results = []
    for zone in zones:
        z = history[history["zone"] == zone].sort_values("recorded_at").copy()
        if len(z) < 8:
            continue
        times = z["recorded_at"].to_numpy(float)
        start = float(times.min()) + max(60.0, window_minutes * 60.0 * 0.5)
        end = float(times.max()) - max(horizons) * 60.0
        if end < start:
            continue
        anchors = np.arange(start, end + 1e-6, max(30, int(anchor_step_seconds)), dtype=float)
        for anchor in anchors:
            train = z[z["recorded_at"] <= anchor]
            if len(train) < 4:
                continue
            f, meta = forecast_zone(
                train, zone, capacities.get(zone, 1), horizons=horizons, window_minutes=window_minutes,
                busy_threshold=busy_threshold, congested_threshold=congested_threshold,
            )
            if f.empty:
                continue
            for _, pred in f.iterrows():
                h = int(pred["horizon_min"])
                target = anchor + h * 60.0
                idx = int(np.argmin(np.abs(times - target)))
                gap = abs(float(times[idx]) - target)
                if gap > target_tolerance_seconds:
                    continue
                actual = float(z.iloc[idx]["occupancy"])
                forecast = float(pred["forecast"])
                results.append({
                    "zone": zone, "anchor_time": anchor, "horizon_min": h,
                    "forecast": forecast, "actual": actual,
                    "abs_error": abs(forecast - actual),
                    "sq_error": (forecast - actual) ** 2,
                    "signed_error": forecast - actual,
                    "reliability": meta.reliability,
                })
    if not results:
        return pd.DataFrame()
    raw = pd.DataFrame(results)
    rows = []
    for (zone, h), g in raw.groupby(["zone", "horizon_min"]):
        denom = float(g["actual"].abs().sum())
        rows.append({
            "zone": zone,
            "horizon_min": int(h),
            "samples": int(len(g)),
            "MAE": float(g["abs_error"].mean()),
            "RMSE": float(np.sqrt(g["sq_error"].mean())),
            "Bias": float(g["signed_error"].mean()),
            "WAPE": float(g["abs_error"].sum() / denom) if denom > 1e-9 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["horizon_min", "zone"]).reset_index(drop=True)
