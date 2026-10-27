from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Iterable

import numpy as np
import pandas as pd


PERSON_ALERT_COLUMNS = [
    "global_id", "anomaly_type", "severity", "zone", "time_sec",
    "value", "threshold", "message",
]
CROWD_ALERT_COLUMNS = [
    "anomaly_type", "severity", "zone", "time_sec", "value", "threshold", "message",
]
INCIDENT_COLUMNS = [
    "incident_type", "severity", "zone", "time_sec", "global_id",
    "recommended_action", "message",
]


@dataclass(frozen=True)
class Anomaly:
    global_id: int
    anomaly_type: str
    severity: str
    zone: str
    time_sec: float
    value: float | None
    threshold: float | None
    message: str


@dataclass(frozen=True)
class CrowdAnomaly:
    anomaly_type: str
    severity: str
    zone: str
    time_sec: float
    value: float | None
    threshold: float | None
    message: str


SEVERITY_RANK = {"CRITICAL": 4, "HIGH": 3, "WARNING": 2, "INFO": 1}


def _severity(ratio: float, *, critical: float = 2.0, high: float = 1.35) -> str:
    if ratio >= critical:
        return "CRITICAL"
    if ratio >= high:
        return "HIGH"
    return "WARNING"


def _sort_alerts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_rank"] = out["severity"].map(SEVERITY_RANK).fillna(0)
    out = out.sort_values(["_rank", "time_sec"], ascending=[False, False]).drop(columns="_rank")
    return out.reset_index(drop=True)


def _zone_episodes(g: pd.DataFrame, max_gap_sec: float) -> Iterable[tuple[str, pd.DataFrame]]:
    """Yield contiguous-ish same-zone episodes, splitting on zone change or long tracking gap."""
    if g.empty:
        return
    gg = g.sort_values("time_sec").reset_index(drop=True)
    episode = 0
    zone_prev = None
    time_prev = None
    ids: list[int] = []
    for i, r in gg.iterrows():
        zone = str(r["zone"])
        t = float(r["time_sec"])
        if zone_prev is None or zone != zone_prev or (time_prev is not None and t - time_prev > max_gap_sec):
            if ids:
                part = gg.loc[ids].copy()
                yield str(part.iloc[0]["zone"]), part
            ids = [i]
            episode += 1
        else:
            ids.append(i)
        zone_prev, time_prev = zone, t
    if ids:
        part = gg.loc[ids].copy()
        yield str(part.iloc[0]["zone"]), part


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1)); n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-9 or n2 <= 1e-9:
        return 0.0
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def detect_trajectory_anomalies(
    observations: pd.DataFrame,
    *,
    loitering_sec: float = 120.0,
    speed_threshold_cm_s: float = 250.0,
    restricted_zones: set[str] | None = None,
    forbidden_transitions: set[tuple[str, str]] | None = None,
    min_points: int = 3,
    max_gap_sec: float = 2.0,
    abrupt_turn_deg: float = 150.0,
    abrupt_turn_min_step_cm: float = 50.0,
    detect_abrupt_turn: bool = False,
) -> pd.DataFrame:
    """Rule-based screening on ground-plane trajectories.

    Supported person-level events:
      * LOITERING: continuous-ish presence in the same zone above a duration threshold.
      * ABNORMAL_SPEED: maximum consecutive-point ground speed above threshold.
      * RESTRICTED_ZONE: presence in a user-configured restricted zone.
      * WRONG_WAY: zone transition explicitly configured as forbidden.
      * ABRUPT_DIRECTION_CHANGE: optional large heading reversal with minimum step length.

    This is operational decision support, not identity recognition or a validated security classifier.
    Thresholds must be calibrated on deployment data before operational use.
    """
    columns = PERSON_ALERT_COLUMNS
    if observations is None or observations.empty:
        return pd.DataFrame(columns=columns)
    required = {"global_id", "time_sec", "ground_x", "ground_y", "zone"}
    if not required.issubset(observations.columns):
        return pd.DataFrame(columns=columns)

    restricted_zones = {str(z) for z in (restricted_zones or set())}
    forbidden_transitions = {(str(a), str(b)) for a, b in (forbidden_transitions or set())}
    rows: list[Anomaly] = []
    obs = observations.dropna(subset=["global_id", "time_sec", "ground_x", "ground_y", "zone"]).copy()
    if obs.empty:
        return pd.DataFrame(columns=columns)

    for gid, g in obs.sort_values(["global_id", "time_sec"]).groupby("global_id"):
        g = g.drop_duplicates(subset=["time_sec"], keep="last").reset_index(drop=True)
        if g.empty:
            continue
        gid_i = int(gid)

        # Restricted-zone intrusion: one alert per track/zone in the current analysis window.
        for zone in sorted(set(g["zone"].astype(str)) & restricted_zones):
            rz = g[g["zone"].astype(str) == zone]
            last = rz.iloc[-1]
            rows.append(Anomaly(
                gid_i, "RESTRICTED_ZONE", "HIGH", zone, float(last.time_sec), None, None,
                f"ID {gid_i} xuất hiện trong khu vực hạn chế {zone}."
            ))

        # Loitering must be a contiguous-ish episode, not the total time accumulated across separated visits.
        for zone, episode in _zone_episodes(g, max_gap_sec=max_gap_sec):
            if len(episode) < max(2, int(min_points)):
                continue
            duration = float(episode["time_sec"].max() - episode["time_sec"].min())
            if duration >= float(loitering_sec):
                rows.append(Anomaly(
                    gid_i, "LOITERING", _severity(duration / max(float(loitering_sec), 1e-6)), zone,
                    float(episode["time_sec"].max()), duration, float(loitering_sec),
                    f"ID {gid_i} lưu lại liên tục tại {zone} khoảng {duration:.1f} giây, vượt ngưỡng {loitering_sec:.0f} giây."
                ))

        # Speed anomaly from consecutive valid ground points.
        if len(g) >= 2:
            prev = g.shift(1)
            dt = g["time_sec"] - prev["time_sec"]
            dist = np.sqrt((g["ground_x"] - prev["ground_x"]) ** 2 + (g["ground_y"] - prev["ground_y"]) ** 2)
            speed = dist / dt.where(dt > 0)
            valid = speed.replace([math.inf, -math.inf], np.nan).dropna()
            if not valid.empty:
                idx = valid.idxmax()
                vmax = float(valid.loc[idx])
                if vmax >= float(speed_threshold_cm_s):
                    r = g.loc[idx]
                    rows.append(Anomaly(
                        gid_i, "ABNORMAL_SPEED", _severity(vmax / max(float(speed_threshold_cm_s), 1e-6)),
                        str(r.zone), float(r.time_sec), vmax, float(speed_threshold_cm_s),
                        f"ID {gid_i} có tốc độ mặt bằng ước lượng {vmax:.1f} cm/s, vượt ngưỡng {speed_threshold_cm_s:.0f} cm/s."
                    ))

        # Wrong-way transition: only transitions explicitly configured as forbidden.
        seq = g[["zone", "time_sec"]].copy()
        seq = seq[seq["zone"].astype(str).ne(seq["zone"].astype(str).shift())]
        if len(seq) >= 2:
            prev_zone = seq["zone"].shift()
            for idx, r in seq.iloc[1:].iterrows():
                pair = (str(prev_zone.loc[idx]), str(r.zone))
                if pair in forbidden_transitions:
                    rows.append(Anomaly(
                        gid_i, "WRONG_WAY", "HIGH", str(r.zone), float(r.time_sec), None, None,
                        f"ID {gid_i} di chuyển {pair[0]} → {pair[1]}, hướng này đang được cấu hình là không hợp lệ."
                    ))

        # Optional heading reversal / abrupt turn diagnostic.
        if detect_abrupt_turn and len(g) >= 3:
            xy = g[["ground_x", "ground_y"]].to_numpy(float)
            for i in range(2, len(g)):
                v1 = xy[i - 1] - xy[i - 2]
                v2 = xy[i] - xy[i - 1]
                if np.linalg.norm(v1) < abrupt_turn_min_step_cm or np.linalg.norm(v2) < abrupt_turn_min_step_cm:
                    continue
                ang = _angle_deg(v1, v2)
                if ang >= abrupt_turn_deg:
                    r = g.iloc[i]
                    rows.append(Anomaly(
                        gid_i, "ABRUPT_DIRECTION_CHANGE", "WARNING", str(r.zone), float(r.time_sec),
                        ang, float(abrupt_turn_deg),
                        f"ID {gid_i} đổi hướng khoảng {ang:.0f}°, vượt ngưỡng chẩn đoán {abrupt_turn_deg:.0f}°."
                    ))
                    break

    if not rows:
        return pd.DataFrame(columns=columns)
    return _sort_alerts(pd.DataFrame([asdict(x) for x in rows], columns=columns))


def _normalize_occupancy_history(history: pd.DataFrame) -> pd.DataFrame:
    """Accept long history (zone, occupancy, time) or wide occupancy (Zone A/B/C per sample)."""
    if history is None or history.empty:
        return pd.DataFrame(columns=["zone", "time_sec", "occupancy"])
    df = history.copy()
    if {"zone", "occupancy"}.issubset(df.columns):
        if "time_sec" not in df.columns:
            if "recorded_at" in df.columns:
                df["time_sec"] = pd.to_numeric(df["recorded_at"], errors="coerce")
            elif "sample_index" in df.columns:
                df["time_sec"] = pd.to_numeric(df["sample_index"], errors="coerce")
            else:
                df["time_sec"] = np.arange(len(df), dtype=float)
        return df[["zone", "time_sec", "occupancy"]].dropna().copy()

    id_col = "time_sec" if "time_sec" in df.columns else ("sample_index" if "sample_index" in df.columns else None)
    zone_cols = [c for c in df.columns if str(c).startswith("Zone ")]
    if not zone_cols:
        return pd.DataFrame(columns=["zone", "time_sec", "occupancy"])
    if id_col is None:
        df["_time"] = np.arange(len(df), dtype=float)
        id_col = "_time"
    out = df.melt(id_vars=[id_col], value_vars=zone_cols, var_name="zone", value_name="occupancy")
    out = out.rename(columns={id_col: "time_sec"})
    return out[["zone", "time_sec", "occupancy"]].dropna().copy()


def detect_crowd_anomalies(
    occupancy_history: pd.DataFrame,
    capacities: dict[str, int],
    *,
    busy_threshold: float = 0.65,
    congested_threshold: float = 0.85,
    surge_people: float = 4.0,
    surge_window_points: int = 5,
    persistent_points: int = 4,
) -> pd.DataFrame:
    """Screen crowd-level occupancy anomalies from recent zone history.

    Events:
      * HIGH_OCCUPANCY: latest utilization >= busy threshold.
      * OVER_CAPACITY: latest occupancy exceeds configured capacity.
      * RAPID_SURGE: increase over a recent point window exceeds surge_people.
      * PERSISTENT_CONGESTION: latest N samples all exceed congested threshold.

    Capacity and thresholds are operational configuration, not externally validated safety limits.
    """
    columns = CROWD_ALERT_COLUMNS
    h = _normalize_occupancy_history(occupancy_history)
    if h.empty:
        return pd.DataFrame(columns=columns)

    rows: list[CrowdAnomaly] = []
    for zone, z in h.groupby("zone"):
        z = z.sort_values("time_sec").drop_duplicates("time_sec", keep="last")
        if z.empty:
            continue
        cap = max(1, int(capacities.get(str(zone), 1)))
        last = z.iloc[-1]
        occ = float(last["occupancy"]); t = float(last["time_sec"]); util = occ / cap

        if occ > cap:
            rows.append(CrowdAnomaly(
                "OVER_CAPACITY", "CRITICAL", str(zone), t, occ, float(cap),
                f"{zone} có {occ:.0f} người, vượt sức chứa cấu hình {cap}."
            ))
        elif util >= congested_threshold:
            rows.append(CrowdAnomaly(
                "HIGH_OCCUPANCY", "HIGH", str(zone), t, util, float(congested_threshold),
                f"{zone} đang ở {util:.0%} sức chứa, vượt ngưỡng ùn tắc {congested_threshold:.0%}."
            ))
        elif util >= busy_threshold:
            rows.append(CrowdAnomaly(
                "HIGH_OCCUPANCY", "WARNING", str(zone), t, util, float(busy_threshold),
                f"{zone} đang ở {util:.0%} sức chứa, vượt ngưỡng đông {busy_threshold:.0%}."
            ))

        win = max(2, int(surge_window_points))
        if len(z) >= win:
            recent = z.iloc[-win:]
            increase = float(recent["occupancy"].iloc[-1] - recent["occupancy"].iloc[0])
            if increase >= float(surge_people):
                rows.append(CrowdAnomaly(
                    "RAPID_SURGE", _severity(increase / max(float(surge_people), 1e-6), critical=2.0, high=1.5),
                    str(zone), t, increase, float(surge_people),
                    f"{zone} tăng {increase:.0f} người trong {win} mẫu gần nhất, vượt ngưỡng tăng nhanh {surge_people:.0f}."
                ))

        n = max(2, int(persistent_points))
        if len(z) >= n:
            util_tail = z["occupancy"].iloc[-n:].astype(float) / cap
            if bool((util_tail >= congested_threshold).all()):
                rows.append(CrowdAnomaly(
                    "PERSISTENT_CONGESTION", "HIGH", str(zone), t, float(util_tail.mean()), float(congested_threshold),
                    f"{zone} duy trì trên ngưỡng ùn tắc trong {n} mẫu liên tiếp."
                ))

    if not rows:
        return pd.DataFrame(columns=columns)
    return _sort_alerts(pd.DataFrame([asdict(x) for x in rows], columns=columns))


def build_incident_table(
    person_alerts: pd.DataFrame | None,
    crowd_alerts: pd.DataFrame | None,
    decisions: Iterable | None = None,
) -> pd.DataFrame:
    """Merge alerts with operational decisions into one prioritized incident queue."""
    rows: list[dict] = []
    decision_map = {getattr(d, "zone", ""): d for d in (decisions or [])}

    if person_alerts is not None and not person_alerts.empty:
        for _, r in person_alerts.iterrows():
            d = decision_map.get(str(r.get("zone", "")))
            action = getattr(d, "action", "MONITOR") if d is not None else "MONITOR"
            if str(r.get("anomaly_type")) == "RESTRICTED_ZONE":
                action = "VERIFY_RESTRICTED_ZONE"
            elif str(r.get("anomaly_type")) == "WRONG_WAY":
                action = "VERIFY_DIRECTION"
            rows.append({
                "incident_type": str(r.get("anomaly_type", "PERSON_ANOMALY")),
                "severity": str(r.get("severity", "WARNING")),
                "zone": str(r.get("zone", "")),
                "time_sec": float(r.get("time_sec", np.nan)),
                "global_id": int(r["global_id"]) if pd.notna(r.get("global_id")) else np.nan,
                "recommended_action": action,
                "message": str(r.get("message", "")),
            })

    if crowd_alerts is not None and not crowd_alerts.empty:
        for _, r in crowd_alerts.iterrows():
            d = decision_map.get(str(r.get("zone", "")))
            action = getattr(d, "action", "MONITOR") if d is not None else "MONITOR"
            rows.append({
                "incident_type": str(r.get("anomaly_type", "CROWD_ANOMALY")),
                "severity": str(r.get("severity", "WARNING")),
                "zone": str(r.get("zone", "")),
                "time_sec": float(r.get("time_sec", np.nan)),
                "global_id": np.nan,
                "recommended_action": action,
                "message": str(r.get("message", "")),
            })

    if not rows:
        return pd.DataFrame(columns=INCIDENT_COLUMNS)
    return _sort_alerts(pd.DataFrame(rows, columns=INCIDENT_COLUMNS)).reset_index(drop=True)
