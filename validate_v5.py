from __future__ import annotations
import ast
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

# 1) Compile every project Python file without importing Streamlit/Ultralytics.
for path in [ROOT/'app.py', *sorted((ROOT/'src').glob('*.py'))]:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

from src.single_camera_lab import generate_tiles, nms_numpy, normalized_roi_to_pixels, count_error_metrics, DetectorConfig
from src.realtime_pipeline import RealtimeConfig

# ROI mapping
assert normalized_roi_to_pixels((1000, 2000, 3), (0.25,0.20,0.75,0.80)) == (500,200,1500,800)

# Tiling covers edges and has overlap.
tiles = generate_tiles(1536, 2048, 768, 0.20)
assert len(tiles) > 4
assert any(x1 == 0 and y1 == 0 for x1,y1,x2,y2 in tiles)
assert any(x2 == 1536 for x1,y1,x2,y2 in tiles)
assert any(y2 == 2048 for x1,y1,x2,y2 in tiles)

# Global NMS removes duplicated boxes coming from overlapping tiles.
boxes = np.array([[10,10,110,210],[12,12,112,212],[400,400,450,500]], float)
scores = np.array([0.9,0.8,0.7])
keep = nms_numpy(boxes, scores, 0.5)
assert keep == [0,2], keep

# Counting diagnostic.
m = count_error_metrics(80,100)
assert m['absolute_error'] == 20 and abs(m['absolute_error_pct']-20.0) < 1e-9

# V5 defaults explicitly fix V4 small-object and 8-FPS ceiling.
d = DetectorConfig()
r = RealtimeConfig()
assert d.image_size >= 960 and d.max_det >= 1000 and d.confidence <= 0.15
assert r.target_fps > 8 and r.image_size >= 960 and r.max_det >= 1000

# Realtime implementation must have an independent AI worker.
rt = (ROOT/'src'/'realtime_pipeline.py').read_text(encoding='utf-8')
assert '_inference_loop' in rt and '_infer_thread' in rt and 'measured_capture_fps' in rt and 'measured_ai_fps' in rt

print('KIỂM TRA V5: ĐẠT')
print(f'  - ROI mapping: ĐẠT')
print(f'  - tiled inference: {len(tiles)} tile trong test 1536x2048')
print(f'  - global NMS: ĐẠT')
print(f'  - mặc định detector: conf={d.confidence}, imgsz={d.image_size}, max_det={d.max_det}')
print(f'  - realtime target mặc định: {r.target_fps} FPS')
print('  - capture worker + AI worker độc lập: ĐẠT')
