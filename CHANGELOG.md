# Changelog

## V6.0 Complete — 2026-09-30

- Giữ toàn bộ single-camera/multi-camera/forecast/SQLite của V5.
- Nâng anomaly từ V5.1 thành hai tầng: person trajectory + crowd occupancy.
- Sửa loitering theo episode liên tục thay vì cộng gộp các lần ghé Zone tách rời.
- Thêm restricted-zone, wrong-way, abnormal speed và optional abrupt direction change.
- Thêm high occupancy, over-capacity, rapid surge và persistent congestion.
- Thêm incident queue ghép cảnh báo với decision engine.
- Thêm bảng `anomaly_alerts` vào SQLite.
- Tích hợp anomaly vào video 1 camera có calibration, video 2 camera và CCTV realtime 2 camera.
- Thêm màn hình `Bất thường & cảnh báo` và lịch sử cảnh báo.
- Thêm `config/operational_policy.json`.
- Thêm `validate_v6.py` và tài liệu kiến trúc/checklist.
- Gói kèm tài liệu nguồn nghiên cứu và WILDTRACK export trong `research_sources/`.
