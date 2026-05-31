from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a checkpoint with the paper protocol.")
    parser.add_argument("--weights", default=str(ROOT / "weights" / "best.pt"))
    parser.add_argument("--data", default=str(ROOT / "data" / "data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--project", default=str(ROOT / "runs" / "val"))
    parser.add_argument("--name", default="validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)

    model = YOLO(args.weights)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        iou=args.iou,
        max_det=args.max_det,
        agnostic_nms=False,
        half=False,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
