from __future__ import annotations
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Fine-tune YOLO cho camera CrowdFlow")
    ap.add_argument("--data", required=True, help="Đường dẫn data.yaml của dataset YOLO")
    ap.add_argument("--model", default="assets/model/yolo11n.pt", help="Checkpoint khởi tạo")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0", help="0 cho GPU đầu tiên; cpu nếu không có CUDA")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="crowdflow_person_v5")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        patience=20,
        plots=True,
        save=True,
    )
    print("Hoàn tất. Hãy lấy best.pt trong thư mục runs/detect/<name>/weights/ và nạp vào V6.")
    return result


if __name__ == "__main__":
    main()
