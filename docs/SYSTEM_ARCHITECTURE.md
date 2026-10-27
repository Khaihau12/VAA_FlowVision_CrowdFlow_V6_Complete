# Kiến trúc VAA FlowVision CrowdFlow V6

## Pipeline

1. **Input** — ảnh, video, CCTV/RTSP.
2. **Detection** — YOLO person detector.
3. **Tracking** — ByteTrack per camera.
4. **Geometry** — footpoint + camera calibration → ground-plane x/y.
5. **Fusion** — Hungarian association trong không gian ground-plane.
6. **Global trajectory** — nối observation theo thời gian.
7. **Zone analytics** — occupancy, transition, Entry/Exit, dwell.
8. **Forecast** — occupancy +5/+10/+15 phút từ SQLite history.
9. **Anomaly** — trajectory rules + crowd rules.
10. **Decision support** — KEEP/RESTRICT/REDIRECT + incident queue.
11. **Output** — dashboard, SQLite, CSV, video/ảnh annotated.

## Hai tầng cảnh báo

### Person/trajectory
`global_id → trajectory → LOITERING / SPEED / RESTRICTED / WRONG_WAY / optional ABRUPT_TURN`

### Crowd
`occupancy history → utilization/surge/persistence → HIGH_OCCUPANCY / OVER_CAPACITY / RAPID_SURGE / PERSISTENT_CONGESTION`

Hai tầng được gộp trong `build_incident_table()`.

## Deployment note

WILDTRACK dùng để baseline và kiểm chứng pipeline/reliability. VAA-2Cam tự thu mới là dữ liệu cần dùng để hiệu chỉnh semantic Zone, capacity, forbidden direction và threshold vận hành theo bối cảnh thực tế.
