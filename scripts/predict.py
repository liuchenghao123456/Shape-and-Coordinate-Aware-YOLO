from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image inference.")
    parser.add_argument("--weights", default=str(ROOT / "weights" / "best.pt"))
    parser.add_argument("--source", default=str(ROOT / "assets" / "bus.jpg"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "detection_result.jpg"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)

    model = YOLO(args.weights)
    results = model(
        args.source,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        agnostic_nms=False,
        half=False,
        save=False,
    )
    result = results[0]
    annotated = result.plot(line_width=2, font_size=1, conf=True, labels=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
