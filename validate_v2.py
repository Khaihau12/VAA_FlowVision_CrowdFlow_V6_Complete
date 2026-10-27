from pathlib import Path
import numpy as np
import pandas as pd

from src.data import load_all
from src.zones import fit_pca_zones
from src.live_pipeline import MODEL_PATH, CAMERA_CALIBRATION, load_calibration, fuse_pair, summarize_fusion


def main():
    assert MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000, "Missing bundled yolo11n.pt"
    for name in CAMERA_CALIBRATION:
        K, D, rvec, tvec = load_calibration(name)
        assert K.shape == (3, 3)
        assert D.size >= 4
        assert rvec.shape == (3,) and tvec.shape == (3,)

    c1 = pd.DataFrame([
        {"sample_index": 0, "time_sec": 0.0, "track_id": 1, "ground_x": 0.0, "ground_y": 0.0},
        {"sample_index": 0, "time_sec": 0.0, "track_id": 2, "ground_x": 500.0, "ground_y": 500.0},
    ])
    c2 = pd.DataFrame([
        {"sample_index": 0, "time_sec": 0.0, "track_id": 7, "ground_x": 30.0, "ground_y": 40.0},
        {"sample_index": 0, "time_sec": 0.0, "track_id": 8, "ground_x": 900.0, "ground_y": 900.0},
    ])
    obs, matches = fuse_pair(c1, c2, 100.0)
    assert len(matches) == 1, matches
    assert abs(float(matches.iloc[0]["cross_camera_distance_cm"]) - 50.0) < 1e-9
    assert len(obs) == 3, obs
    assert obs["global_id"].nunique() == 3

    data = load_all()
    zmodel = fit_pca_zones(data["gt"])
    fusion = summarize_fusion(obs, zmodel, 0.2)
    assert not fusion.occupancy.empty
    print("V2 VALIDATION: PASS")
    print(f"Bundled model: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size/1024/1024:.2f} MB)")
    print(f"Calibrations: {len(CAMERA_CALIBRATION)} cameras")
    print("Synthetic Hungarian match distance: 50.0 cm")


if __name__ == "__main__":
    main()
