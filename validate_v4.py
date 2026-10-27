from __future__ import annotations

from pathlib import Path
import tempfile

from src.history_db import init_db, seed_demo_history, read_metrics, db_stats, clear_database
from src.forecast_engine import forecast_all, backtest_forecasts

ROOT = Path(__file__).resolve().parent
CAP = {"Zone A": 10, "Zone B": 12, "Zone C": 4}
ZONES = ["Zone A", "Zone B", "Zone C"]

# 1) Các file V4 bắt buộc.
required = [
    ROOT / "app.py",
    ROOT / "src" / "history_db.py",
    ROOT / "src" / "forecast_engine.py",
    ROOT / "src" / "realtime_pipeline.py",
    ROOT / "src" / "image_pipeline.py",
    ROOT / "assets" / "model" / "yolo11n.pt",
]
missing = [str(p) for p in required if not p.exists()]
assert not missing, f"Thiếu file V4: {missing}"

# 2) SQLite + dữ liệu mô phỏng được gắn nhãn rõ.
with tempfile.TemporaryDirectory() as td:
    db = Path(td) / "v4_test.sqlite"
    init_db(db)
    inserted = seed_demo_history(CAP, minutes=20, step_seconds=30, db_path=db)
    hist = read_metrics(source="MÔ PHỎNG V4 - KHÔNG PHẢI DỮ LIỆU THỰC", since_minutes=60, db_path=db)
    assert inserted > 0 and len(hist) > 0
    assert set(hist["zone"].unique()) == set(ZONES)

    # 3) Forecast phải có đủ 3 khu x 3 horizon.
    forecasts, meta = forecast_all(hist, CAP, ZONES, horizons=(5, 10, 15))
    assert len(forecasts) == 9
    assert set(forecasts["horizon_min"].tolist()) == {5, 10, 15}
    assert forecasts["forecast"].ge(0).all()
    assert forecasts["status"].isin(["NORMAL", "BUSY", "CONGESTED"]).all()
    assert forecasts["action"].isin(["KEEP", "RESTRICT", "REDIRECT"]).all()

    # 4) Backtest tự động hoạt động khi lịch sử đủ dài.
    backtest = backtest_forecasts(hist, CAP, ZONES, horizons=(5, 10, 15))
    assert not backtest.empty
    assert set(backtest["horizon_min"].tolist()) == {5, 10, 15}

    # 5) Xóa CSDL.
    clear_database(db)
    assert db_stats(db)["metric_rows"] == 0

print("KIỂM TRA V4: ĐẠT")
print(f"- File bắt buộc: {len(required)}")
print(f"- Bản ghi mô phỏng kiểm thử: {inserted}")
print("- Dự báo: 3 khu × 3 tầm 5/10/15 phút = đạt")
print("- Backtest MAE/RMSE/WAPE: đạt")
print("- SQLite ghi/đọc/xóa: đạt")
