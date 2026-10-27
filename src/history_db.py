from __future__ import annotations

from pathlib import Path
import sqlite3
import time
from datetime import datetime
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "runtime" / "crowdflow_v6.sqlite"


def _connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10.0)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    with _connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS realtime_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at REAL NOT NULL,
                recorded_at_iso TEXT NOT NULL,
                source TEXT NOT NULL,
                sample_key TEXT NOT NULL,
                zone TEXT NOT NULL,
                occupancy INTEGER NOT NULL,
                capacity INTEGER NOT NULL,
                utilization REAL NOT NULL,
                status TEXT NOT NULL,
                action TEXT NOT NULL,
                UNIQUE(source, sample_key, zone)
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_time ON realtime_metrics(recorded_at);
            CREATE INDEX IF NOT EXISTS idx_metrics_source_zone_time ON realtime_metrics(source, zone, recorded_at);

            CREATE TABLE IF NOT EXISTS forecast_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at REAL NOT NULL,
                recorded_at_iso TEXT NOT NULL,
                source TEXT NOT NULL,
                zone TEXT NOT NULL,
                horizon_min INTEGER NOT NULL,
                predicted_occupancy REAL NOT NULL,
                predicted_utilization REAL NOT NULL,
                status TEXT NOT NULL,
                action TEXT NOT NULL,
                reliability TEXT NOT NULL,
                message TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_alerts_time ON forecast_alerts(recorded_at);

            CREATE TABLE IF NOT EXISTS anomaly_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at REAL NOT NULL,
                recorded_at_iso TEXT NOT NULL,
                source TEXT NOT NULL,
                sample_key TEXT NOT NULL,
                alert_scope TEXT NOT NULL,
                anomaly_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                zone TEXT,
                global_id INTEGER,
                event_time REAL,
                value REAL,
                threshold REAL,
                message TEXT NOT NULL,
                UNIQUE(source, sample_key, alert_scope, anomaly_type, zone, global_id, event_time)
            );
            CREATE INDEX IF NOT EXISTS idx_anomaly_alerts_time ON anomaly_alerts(recorded_at);
            CREATE INDEX IF NOT EXISTS idx_anomaly_alerts_source ON anomaly_alerts(source, recorded_at);
            """
        )


def log_zone_snapshot(
    source: str,
    sample_key: str,
    counts: dict[str, int],
    capacities: dict[str, int],
    decisions: Iterable,
    recorded_at: float | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Lưu một snapshot theo khu. UNIQUE chống ghi trùng khi Streamlit fragment render lại."""
    init_db(db_path)
    ts = float(time.time() if recorded_at is None else recorded_at)
    iso = datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")
    decision_map = {getattr(d, "zone", ""): d for d in decisions}
    rows = []
    for zone, occ in counts.items():
        cap = max(1, int(capacities.get(zone, 1)))
        dec = decision_map.get(zone)
        util = float(occ) / cap
        status = getattr(dec, "status", "UNKNOWN")
        action = getattr(dec, "action", "UNKNOWN")
        rows.append((ts, iso, str(source), str(sample_key), str(zone), int(occ), cap, util, status, action))
    with _connect(db_path) as con:
        before = con.total_changes
        con.executemany(
            """
            INSERT OR IGNORE INTO realtime_metrics
            (recorded_at, recorded_at_iso, source, sample_key, zone, occupancy, capacity, utilization, status, action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return con.total_changes - before


def log_total_snapshot(
    source: str,
    sample_key: str,
    occupancy: int,
    recorded_at: float | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Lưu tổng số người khi camera chưa có hiệu chuẩn mặt bằng."""
    init_db(db_path)
    ts = float(time.time() if recorded_at is None else recorded_at)
    iso = datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")
    with _connect(db_path) as con:
        before = con.total_changes
        con.execute(
            """
            INSERT OR IGNORE INTO realtime_metrics
            (recorded_at, recorded_at_iso, source, sample_key, zone, occupancy, capacity, utilization, status, action)
            VALUES (?, ?, ?, ?, 'TOTAL', ?, 1, ?, 'INFO', 'MONITOR')
            """,
            (ts, iso, str(source), str(sample_key), int(occupancy), float(occupancy)),
        )
        return con.total_changes - before


def read_metrics(
    source: str | None = None,
    since_minutes: float | None = None,
    limit: int = 100000,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    init_db(db_path)
    where, args = [], []
    if source and source != "Tất cả":
        where.append("source = ?")
        args.append(source)
    if since_minutes is not None:
        where.append("recorded_at >= ?")
        args.append(time.time() - float(since_minutes) * 60.0)
    clause = " WHERE " + " AND ".join(where) if where else ""
    sql = f"SELECT * FROM realtime_metrics{clause} ORDER BY recorded_at ASC, id ASC LIMIT ?"
    args.append(int(limit))
    with _connect(db_path) as con:
        return pd.read_sql_query(sql, con, params=args)


def list_sources(db_path: str | Path = DEFAULT_DB_PATH) -> list[str]:
    init_db(db_path)
    with _connect(db_path) as con:
        rows = con.execute("SELECT DISTINCT source FROM realtime_metrics ORDER BY source").fetchall()
    return [str(r[0]) for r in rows]


def db_stats(db_path: str | Path = DEFAULT_DB_PATH) -> dict:
    init_db(db_path)
    with _connect(db_path) as con:
        n = int(con.execute("SELECT COUNT(*) FROM realtime_metrics").fetchone()[0])
        a = int(con.execute("SELECT COUNT(*) FROM forecast_alerts").fetchone()[0])
        aa = int(con.execute("SELECT COUNT(*) FROM anomaly_alerts").fetchone()[0])
        first = con.execute("SELECT MIN(recorded_at), MAX(recorded_at) FROM realtime_metrics").fetchone()
    path = Path(db_path)
    return {
        "metric_rows": n,
        "alert_rows": a,
        "anomaly_alert_rows": aa,
        "first_ts": first[0],
        "last_ts": first[1],
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def log_alerts(alerts: pd.DataFrame, source: str, db_path: str | Path = DEFAULT_DB_PATH) -> int:
    if alerts is None or alerts.empty:
        return 0
    init_db(db_path)
    ts = time.time()
    iso = datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")
    rows = []
    for _, r in alerts.iterrows():
        rows.append((
            ts, iso, source, str(r["zone"]), int(r["horizon_min"]), float(r["forecast"]),
            float(r["predicted_utilization"]), str(r["status"]), str(r["action"]),
            str(r["reliability"]), str(r["message"]),
        ))
    with _connect(db_path) as con:
        before = con.total_changes
        con.executemany(
            """
            INSERT INTO forecast_alerts
            (recorded_at, recorded_at_iso, source, zone, horizon_min, predicted_occupancy,
             predicted_utilization, status, action, reliability, message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return con.total_changes - before


def read_alerts(limit: int = 5000, db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as con:
        return pd.read_sql_query(
            "SELECT * FROM forecast_alerts ORDER BY recorded_at DESC, id DESC LIMIT ?", con, params=[int(limit)]
        )


def clear_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    init_db(db_path)
    with _connect(db_path) as con:
        con.execute("DELETE FROM realtime_metrics")
        con.execute("DELETE FROM forecast_alerts")
        con.execute("DELETE FROM anomaly_alerts")
    with _connect(db_path) as con:
        con.execute("VACUUM")


def seed_demo_history(
    capacities: dict[str, int],
    source: str = "MÔ PHỎNG V4 - KHÔNG PHẢI DỮ LIỆU THỰC",
    minutes: int = 20,
    step_seconds: int = 30,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Tạo chuỗi deterministic chỉ để thử giao diện forecast khi chưa có CCTV."""
    import math
    init_db(db_path)
    now = time.time()
    n = max(8, int(minutes * 60 / step_seconds) + 1)
    with _connect(db_path) as con:
        con.execute("DELETE FROM realtime_metrics WHERE source = ?", (source,))
    total = 0
    for i in range(n):
        frac = i / max(1, n - 1)
        ts = now - (n - 1 - i) * step_seconds
        # Khu B tăng rõ để minh họa cảnh báo sớm; A/C biến động nhẹ.
        counts = {
            "Zone A": max(0, int(round(3 + 2.0 * frac + 0.5 * math.sin(i / 3)))),
            "Zone B": max(0, int(round(4 + 7.0 * frac + 0.7 * math.sin(i / 4)))),
            "Zone C": max(0, int(round(1 + 1.0 * frac + 0.3 * math.sin(i / 5)))),
        }
        # Tránh import Decision để module DB độc lập; tính status/action cùng quy tắc mặc định.
        fake = []
        for z, occ in counts.items():
            cap = max(1, int(capacities.get(z, 1)))
            u = occ / cap
            if u >= 0.85:
                status, action = "CONGESTED", "REDIRECT"
            elif u >= 0.65:
                status, action = "BUSY", "RESTRICT"
            else:
                status, action = "NORMAL", "KEEP"
            class Obj: pass
            o = Obj(); o.zone = z; o.status = status; o.action = action
            fake.append(o)
        total += log_zone_snapshot(source, f"demo-{i}", counts, capacities, fake, recorded_at=ts, db_path=db_path)
    return total


def log_anomaly_alerts(
    alerts: pd.DataFrame,
    source: str,
    sample_key: str,
    alert_scope: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Persist anomaly/crowd alerts with de-duplication across Streamlit refreshes."""
    if alerts is None or alerts.empty:
        return 0
    init_db(db_path)
    ts = time.time()
    iso = datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")
    rows = []
    for _, r in alerts.iterrows():
        gid = r.get("global_id")
        gid = int(gid) if pd.notna(gid) else None
        et = r.get("time_sec")
        et = float(et) if pd.notna(et) else None
        value = r.get("value")
        value = float(value) if pd.notna(value) else None
        threshold = r.get("threshold")
        threshold = float(threshold) if pd.notna(threshold) else None
        rows.append((
            ts, iso, str(source), str(sample_key), str(alert_scope),
            str(r.get("anomaly_type", "UNKNOWN")), str(r.get("severity", "WARNING")),
            str(r.get("zone", "")), gid, et, value, threshold, str(r.get("message", "")),
        ))
    with _connect(db_path) as con:
        before = con.total_changes
        con.executemany(
            """
            INSERT OR IGNORE INTO anomaly_alerts
            (recorded_at, recorded_at_iso, source, sample_key, alert_scope, anomaly_type, severity,
             zone, global_id, event_time, value, threshold, message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return con.total_changes - before


def read_anomaly_alerts(
    source: str | None = None,
    limit: int = 5000,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    init_db(db_path)
    where, args = [], []
    if source and source != "Tất cả":
        where.append("source = ?")
        args.append(source)
    clause = " WHERE " + " AND ".join(where) if where else ""
    sql = f"SELECT * FROM anomaly_alerts{clause} ORDER BY recorded_at DESC, id DESC LIMIT ?"
    args.append(int(limit))
    with _connect(db_path) as con:
        return pd.read_sql_query(sql, con, params=args)
