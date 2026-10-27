from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd

from src.anomaly_engine import detect_trajectory_anomalies, detect_crowd_anomalies, build_incident_table
from src.decision_engine import evaluate_zone
from src.history_db import init_db, log_anomaly_alerts, read_anomaly_alerts


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[OK] {message}")


def main() -> None:
    # Synthetic ground-plane trajectories.
    rows = []
    for t in [0, 30, 60, 90, 120, 130]:
        rows.append({"global_id": 1, "time_sec": t, "ground_x": 100 + t * 0.2, "ground_y": 100, "zone": "Zone A"})
    rows += [
        {"global_id": 2, "time_sec": 0, "ground_x": 0, "ground_y": 0, "zone": "Zone B"},
        {"global_id": 2, "time_sec": 1, "ground_x": 10, "ground_y": 0, "zone": "Zone C"},
        {"global_id": 3, "time_sec": 0, "ground_x": 0, "ground_y": 0, "zone": "Zone B"},
        {"global_id": 3, "time_sec": 1, "ground_x": 400, "ground_y": 0, "zone": "Zone A"},
    ]
    obs = pd.DataFrame(rows)
    person = detect_trajectory_anomalies(
        obs,
        loitering_sec=120,
        speed_threshold_cm_s=250,
        restricted_zones={"Zone C"},
        forbidden_transitions={("Zone B", "Zone A")},
        max_gap_sec=35,
    )
    kinds = set(person["anomaly_type"].tolist())
    check("LOITERING" in kinds, "Detect loitering episode")
    check("RESTRICTED_ZONE" in kinds, "Detect restricted-zone presence")
    check("WRONG_WAY" in kinds, "Detect configured wrong-way transition")
    check("ABNORMAL_SPEED" in kinds, "Detect abnormal ground-plane speed")

    occ = pd.DataFrame({
        "sample_index": [0, 1, 2, 3, 4],
        "Zone A": [2, 2, 2, 2, 2],
        "Zone B": [4, 6, 8, 11, 14],
        "Zone C": [1, 1, 1, 1, 1],
    })
    capacities = {"Zone A": 10, "Zone B": 12, "Zone C": 4}
    crowd = detect_crowd_anomalies(
        occ, capacities, busy_threshold=0.65, congested_threshold=0.85,
        surge_people=4, surge_window_points=5, persistent_points=2,
    )
    ckinds = set(crowd["anomaly_type"].tolist())
    check("OVER_CAPACITY" in ckinds, "Detect configured over-capacity")
    check("RAPID_SURGE" in ckinds, "Detect rapid crowd surge")

    counts = {"Zone A": 2, "Zone B": 14, "Zone C": 1}
    decisions = [evaluate_zone(z, counts[z], capacities, counts) for z in capacities]
    incidents = build_incident_table(person, crowd, decisions)
    check(not incidents.empty and "recommended_action" in incidents.columns, "Build prioritized incident queue")

    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.sqlite"
        init_db(db)
        n1 = log_anomaly_alerts(person, "synthetic", "s1", "PERSON", db_path=db)
        n2 = log_anomaly_alerts(person, "synthetic", "s1", "PERSON", db_path=db)
        saved = read_anomaly_alerts(source="synthetic", db_path=db)
        check(n1 > 0, "Persist anomaly alerts")
        check(n2 == 0, "De-duplicate identical anomaly alerts")
        check(len(saved) == n1, "Read persisted anomaly alerts")

    print("\nV6 validation passed.")


if __name__ == "__main__":
    main()
