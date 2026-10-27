# Phương pháp V5 — Single-camera-first

## 1. Lý do thay đổi quy trình

Multi-camera fusion chỉ hoạt động trên các quan sát đã được detector/tracker tạo ra. Nếu detector ở Camera 1 và Camera 2 cùng bỏ sót người nhỏ thì Hungarian không thể suy ra người không tồn tại trong đầu vào. Vì vậy V5 đặt cổng đánh giá single-camera trước khi sử dụng multi-camera để kết luận.

## 2. Detection

Model mặc định là YOLO11n pretrained, class person. V5 tăng khả năng quan sát người nhỏ bằng bốn cơ chế:

- confidence có thể giảm xuống 0.03;
- imgsz đến 1536 cho ảnh và 1280 cho realtime/video;
- max_det mặc định 1000;
- ROI và tiled inference.

### ROI

Với camera cố định, ROI loại bỏ các vùng không liên quan. Bounding box sau suy luận được ánh xạ ngược về tọa độ ảnh gốc nên điểm chân vẫn có thể dùng với calibration nếu hình học camera gốc phù hợp.

### Tiled inference

ROI được chia thành các tile chồng lấn. Detector chạy trên từng tile; tất cả box được chuyển về hệ tọa độ ảnh gốc và hợp nhất bằng NMS toàn cục. Tiled inference hướng đến ảnh/offline dense crowd, không mặc định cho realtime vì chi phí tính toán cao.

## 3. Tracking

Video/realtime dùng ByteTrack. V5 giữ điểm chân:

`foot_x = (x1 + x2) / 2`

`foot_y = y2`

ROI chỉ thay đổi vùng đưa vào detector; box được cộng offset trước khi lưu/trình chiếu/ground projection.

## 4. Realtime architecture

V4: UI gọi `analyze_latest()` theo chu kỳ refresh, nên inference bị giới hạn bởi Streamlit.

V5:

```text
RTSP/Webcam
    ↓
Capture worker ── luôn giữ frame mới nhất
    ↓
Latest-frame buffer
    ↓
AI worker ── YOLO + ByteTrack theo target FPS
    ↓
Latest annotated packet
    ↓
Streamlit UI chỉ poll kết quả
```

Không xây queue frame dài. Nếu AI chậm, frame cũ được bỏ qua để hạn chế latency tích lũy.

## 5. FPS và latency

V5 tách các đại lượng:

- `stream_fps`: FPS metadata camera/OpenCV báo;
- `measured_capture_fps`: FPS đọc frame thực đo;
- `measured_ai_fps`: số inference thực tế/giây;
- `inference_ms`: thời gian YOLO+ByteTrack;
- frame age: thời gian từ khi frame được capture đến lúc UI hiển thị packet gần nhất.

## 6. Chẩn đoán dense crowd

### Confidence sweep

Giữ ảnh/model/imgsz cố định và thay confidence. Nếu count tăng mạnh khi threshold giảm, nguyên nhân chủ yếu là filtering confidence. Nếu count vẫn rất thấp ở conf 0.05, model/scale mới là giới hạn chính.

### Resolution sweep

Chạy 640/960/1280. Nếu count tăng mạnh theo imgsz thì small-object/downsampling là nguyên nhân quan trọng.

### Ground-truth count sơ bộ

Người dùng có thể nhập số người thật đã đếm thủ công để tính absolute count error và percent error. Chỉ số này không thay thế Precision/Recall/mAP vì chưa có annotation box đầy đủ.

## 7. Fine-tuning

V5 hỗ trợ model `.pt` tùy chỉnh. Để chứng minh model học đặc trưng ảnh camera, cần dataset riêng có annotation, chia Train/Validation/Test. Nên bao phủ:

- người gần/xa;
- thưa/vừa/đông/rất đông;
- occlusion;
- motion blur;
- sáng/tối/ngược sáng;
- góc nhìn cao;
- xe máy/vật cản.

Có thể chạy ablation/pretrained comparison giữa YOLO pretrained, person fine-tuned và head detector fine-tuned.

## 8. Điều kiện chuyển sang multi-camera

Không đặt một con số tùy ý cứng trong code. Nhóm nghiên cứu phải định nghĩa trước tiêu chí PASS dựa trên yêu cầu bài toán. Tối thiểu nên báo cáo detection/counting/tracking trên TEST single-camera trước khi báo cáo fusion.

Sau đó multi-camera mới dùng:

- time synchronization;
- camera calibration;
- ground-plane projection;
- Hungarian matching;
- global ID;
- occupancy/flow/dwell.

## 9. Giới hạn còn lại

- YOLO11n bundled vẫn là model pretrained tổng quát.
- Head detection chưa được đóng gói vì chưa có model head detector đã train/đánh giá của đề tài.
- Tiled inference cải thiện scale nhưng có thể tăng false positives và latency; phải benchmark.
- Streamlit phù hợp prototype nghiên cứu; production realtime FPS cao nên tách inference server/frontend chuyên dụng.

---

# Bổ sung phương pháp V6 — Anomaly & Decision Support

## 1. Nguyên tắc

V6 không suy luận danh tính và không gán nhãn hành vi nguy hiểm. Module anomaly là lớp **screening/decision support** dựa trên global trajectory, Zone và occupancy. Mọi ngưỡng đều là tham số vận hành nghiên cứu cần hiệu chỉnh trên VAA-2Cam.

## 2. Bất thường cấp quỹ đạo

- **Loitering**: tách trajectory thành các episode liên tục theo Zone; episode bị cắt khi đổi Zone hoặc mất track quá `max_gap_sec`. Cảnh báo nếu thời lượng episode vượt `loitering_sec`.
- **Abnormal speed**: tính vận tốc giữa hai điểm ground-plane liên tiếp và cảnh báo khi vận tốc lớn nhất vượt `speed_threshold_cm_s`.
- **Restricted zone**: cảnh báo khi trajectory xuất hiện trong Zone do người dùng cấu hình là hạn chế.
- **Wrong-way**: chỉ cảnh báo transition nằm trong tập hướng cấm do người dùng cấu hình; không tự suy đoán hướng hợp lệ.
- **Abrupt direction change**: diagnostic tùy chọn dựa trên góc giữa hai vector dịch chuyển liên tiếp và minimum step length.

## 3. Bất thường cấp đám đông

Từ lịch sử occupancy theo Zone:

- `HIGH_OCCUPANCY`: utilization vượt busy/congested threshold.
- `OVER_CAPACITY`: occupancy vượt capacity cấu hình.
- `RAPID_SURGE`: chênh lệch occupancy trong cửa sổ mẫu gần nhất vượt ngưỡng.
- `PERSISTENT_CONGESTION`: N mẫu cuối cùng đều vượt congested threshold.

Các capacity/threshold này không được gọi là chuẩn an toàn sân bay nếu chưa có xác nhận nghiệp vụ.

## 4. Hàng đợi sự cố

Cảnh báo person + crowd được chuẩn hóa severity và gộp với `Decision` hiện tại. Incident queue ưu tiên severity cao hơn và đưa ra action có thể kiểm tra được (`MONITOR`, `RESTRICT`, `REDIRECT`, `VERIFY_RESTRICTED_ZONE`, `VERIFY_DIRECTION`).

## 5. Lưu trữ

SQLite V6 có ba nhóm dữ liệu:

1. `realtime_metrics`: occupancy/capacity/utilization/status/action.
2. `forecast_alerts`: cảnh báo dự báo +5/+10/+15 phút.
3. `anomaly_alerts`: cảnh báo quỹ đạo/đám đông có source, sample, type, severity, zone, global_id và message.

## 6. Những phần không được suy diễn

V6 chưa có model action-recognition chuyên biệt cho fight/fall/crime và không có face recognition. Không được viết rằng hệ thống “phát hiện hành vi nguy hiểm” nếu chưa có dataset, annotation, protocol và đánh giá riêng cho bài toán đó.
