from pathlib import Path
import py_compile
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent

# 1) Mã nguồn chính phải biên dịch được.
for rel in [
    "app.py",
    "src/image_pipeline.py",
    "src/realtime_pipeline.py",
    "src/live_pipeline.py",
    "src/decision_engine.py",
]:
    py_compile.compile(str(ROOT / rel), doraise=True)

# 2) Model và hiệu chuẩn phải tồn tại.
model = ROOT / "assets/model/yolo11n.pt"
assert model.exists() and model.stat().st_size > 1_000_000, "Thiếu checkpoint YOLO11n"
assert len(list((ROOT / "assets/calibration/intrinsic_original").glob("*.xml"))) == 7
assert len(list((ROOT / "assets/calibration/extrinsic").glob("*.xml"))) == 7

# 3) Mô-đun ảnh phải đọc được ảnh hợp lệ mà chưa cần chạy YOLO.
from src.image_pipeline import decode_image
img = np.zeros((64, 96, 3), dtype=np.uint8)
ok, encoded = cv2.imencode(".jpg", img)
assert ok
restored = decode_image(encoded.tobytes())
assert restored.shape[:2] == (64, 96)

# 4) Phiên realtime phải khởi tạo được mà không mở camera.
from src.realtime_pipeline import RealtimeConfig, RealtimeStream
session = RealtimeStream(0, "Camera kiểm tra", RealtimeConfig(target_fps=1.0), None)
info = session.status
assert info["status"] == "STOPPED"
assert info["source"]

print("KIỂM TRA V3: ĐẠT")
print("- Mã nguồn: hợp lệ")
print("- YOLO11n: có sẵn")
print("- 7 bộ hiệu chuẩn: có sẵn")
print("- Đọc ảnh: đạt")
print("- Khởi tạo phiên CCTV/RTSP: đạt")
