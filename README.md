# VAA FlowVision — CrowdFlow Research App V6 Complete

## 1. Mục tiêu

V6 là gói nguyên mẫu nghiên cứu đầy đủ cho bài toán **phân tích luồng người đa camera và hỗ trợ bộ phận quản lý ra quyết định điều phối đám đông**, đồng thời bổ sung **sàng lọc bất thường từ đối tượng/quỹ đạo và bất thường cấp đám đông**.

Chuỗi xử lý chính:

`Camera/Video/Ảnh → YOLO11n/person detector → ByteTrack → điểm chân → hiệu chuẩn mặt bằng → multi-camera Hungarian fusion → global trajectory → Zone/Occupancy → Entry/Exit → Dwell → Directed Flow → Forecast → Anomaly Detection → Decision Support → Dashboard/SQLite/CSV`

V6 kế thừa toàn bộ V5/V5.1 và không thay đổi các kết quả TEST WILDTRACK đã khóa.

## 2. Chức năng đã tích hợp

### Detection / Tracking
- Upload ảnh, video và CCTV/RTSP.
- YOLO11n đóng gói sẵn; có thể upload model `.pt` fine-tune riêng.
- Single-camera Lab để quét confidence, resolution, ROI và tiled inference.
- ByteTrack cho video/realtime.
- Capture worker và AI worker độc lập để giảm nghẽn realtime.

### Multi-camera / Flow analytics
- Hiệu chuẩn WILDTRACK 7 camera.
- Pixel → ground-plane.
- Hungarian fusion giữa hai camera.
- Global trajectory trong cửa sổ phân tích.
- Zone occupancy.
- Entry/Exit nội bộ.
- Dwell time.
- Directed flow transition.
- Dashboard map + bảng + xuất CSV.

### Forecast / Decision Support
- Lịch sử occupancy lưu bằng SQLite.
- Forecast baseline +5/+10/+15 phút.
- Backtest MAE/RMSE/WAPE trên lịch sử do pipeline tạo.
- Chính sách minh bạch: `KEEP / RESTRICT / REDIRECT`.
- Sức chứa và ngưỡng đông/ùn tắc điều chỉnh trên UI.

### V6 Anomaly Detection
**Cấp đối tượng/quỹ đạo**
- `LOITERING`: lưu lại liên tục tại một Zone quá ngưỡng.
- `ABNORMAL_SPEED`: tốc độ mặt bằng ước lượng vượt ngưỡng.
- `RESTRICTED_ZONE`: xuất hiện trong Zone được cấu hình là hạn chế.
- `WRONG_WAY`: chuyển Zone theo hướng cấm được cấu hình.
- `ABRUPT_DIRECTION_CHANGE`: đổi hướng lớn, tùy chọn experimental.

**Cấp đám đông**
- `HIGH_OCCUPANCY`: vượt ngưỡng đông/ùn tắc.
- `OVER_CAPACITY`: vượt sức chứa cấu hình.
- `RAPID_SURGE`: số người tăng nhanh trong cửa sổ gần nhất.
- `PERSISTENT_CONGESTION`: ùn tắc kéo dài nhiều mẫu liên tiếp.

**Incident queue**
- Gộp cảnh báo cá nhân + đám đông.
- Xếp ưu tiên theo severity: `CRITICAL / HIGH / WARNING / INFO`.
- Ghép với quyết định điều phối hiện tại.
- Lưu cảnh báo realtime vào SQLite và tải CSV.

## 3. Phạm vi khoa học bắt buộc

- Module anomaly là **decision support theo quy tắc**, không phải nhận dạng danh tính, không phải hệ thống an ninh tự động và chưa được xác thực là ngưỡng an toàn sân bay.
- WILDTRACK Zone A/B/C là **vùng không gian tổng quát**. Không tự đổi tên thành Security/Gate/Check-in/Baggage.
- Các ngưỡng capacity/loitering/speed/restricted/wrong-way phải được hiệu chỉnh và đánh giá trên dữ liệu VAA-2Cam tự thu trước khi triển khai thực tế.
- Hệ thống hiện **không tuyên bố** nhận diện đánh nhau, té ngã, hành vi phạm pháp hoặc khuôn mặt. Các bài toán đó cần dataset gán nhãn và model chuyên biệt.

## 4. Cách chạy Windows

Mở PowerShell/CMD trong thư mục project:

```powershell
python -m pip install -r requirements.txt
python check_gpu.py
python validate_v6.py
python -m streamlit run app.py
```

Sau đó mở `http://localhost:8501`.

Hoặc double-click `run_app.bat`.

## 5. Thứ tự kiểm thử khuyến nghị

1. `Phòng thí nghiệm 1 camera`: kiểm tra detector trên ảnh TEST riêng.
2. `Phân tích video → 1 camera`: kiểm tra track ID và ROI.
3. `Phân tích video → 2 camera`: kiểm tra calibration/fusion/zone/flow.
4. `CCTV / RTSP thời gian thực`: kiểm tra FPS camera, FPS AI, sync và fusion.
5. `Bất thường & cảnh báo`: cấu hình restricted zone, wrong-way, loitering, speed, surge.
6. `Dự báo & cảnh báo sớm`: cần lịch sử occupancy đủ dài.
7. `Lịch sử & cơ sở dữ liệu`: kiểm tra metric, forecast alert và anomaly alert.
8. Chỉ sau khi single-camera đạt tiêu chí TEST mới dùng multi-camera để kết luận nghiên cứu.

## 6. Validation

```powershell
python validate_research_data.py
python validate_v2.py
python validate_v3.py
python validate_v4.py
python validate_v5.py
python validate_v6.py
```

`validate_v6.py` kiểm tra synthetic cho loitering, restricted-zone, wrong-way, abnormal speed, over-capacity, rapid surge, incident queue và SQLite anomaly logging/de-duplication.

## 7. File quan trọng

- `app.py`: Streamlit dashboard toàn hệ thống.
- `src/anomaly_engine.py`: person/crowd anomaly + incident queue.
- `src/decision_engine.py`: chính sách KEEP/RESTRICT/REDIRECT.
- `src/forecast_engine.py`: forecast + backtest.
- `src/history_db.py`: SQLite metrics/forecast alerts/anomaly alerts.
- `src/live_pipeline.py`: video tracking + calibration + fusion.
- `src/realtime_pipeline.py`: realtime capture/AI workers.
- `config/operational_policy.json`: policy mặc định nghiên cứu.
- `docs/SYSTEM_ARCHITECTURE.md`: kiến trúc chi tiết.
- `docs/COMPLETENESS_CHECKLIST.md`: checklist chức năng.
- `METHODS.md`: phương pháp và giới hạn nghiên cứu.

## 8. Model

Project có `assets/model/yolo11n.pt` để chạy ngay. Đây là pretrained general-purpose, không phải model đã fine-tune riêng camera VAA. Với cảnh góc cao/người nhỏ/che khuất, nên annotation + fine-tune và giữ Train/Validation/Test tách riêng.
