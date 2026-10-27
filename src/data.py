from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def csv(path: str) -> pd.DataFrame:
    return pd.read_csv(DATA / path)


def json_file(path: str) -> dict:
    with open(DATA / path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all() -> dict:
    return {
        "gt": csv("pair12/official_ground_truth.csv"),
        "fusion": csv("pair12/fusion_cam1_cam2.csv"),
        "ground_cam1": csv("pair12/ground_cam1_valid.csv"),
        "ground_cam2": csv("pair12/ground_cam2_valid.csv"),
        "zone_results": csv("final_summary/07_zone_results.csv"),
        "tracking_results": csv("final_summary/08_tracking_results.csv"),
        "entry_exit_results": csv("final_summary/09_entry_exit_results.csv"),
        "dwell_results": csv("final_summary/10_dwell_results.csv"),
        "flow_results": csv("final_summary/11_flow_results.csv"),
        "scorecard": csv("final_summary/13_configuration_scorecard.csv"),
        "fusion_comparison": csv("final_summary/06_fusion_comparison.csv"),
        "robustness": csv("final_summary/03_robustness_primary.csv"),
        "protocol": json_file("final_summary/FINAL_PROTOCOL.json"),
    }
