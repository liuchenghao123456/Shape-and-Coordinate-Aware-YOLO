from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference on a folder of images.")
    parser.add_argument("--weights", default=str(ROOT / "weights" / "best.pt"))
    parser.add_argument("--source", default=str(ROOT / "data" / "raw" / "test" / "images"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "batch"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def iter_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)

    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    for img_path in iter_images(source):
        results = model(
            str(img_path),
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            agnostic_nms=False,
            half=False,
            save=False,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot(line_width=2, font_size=1, conf=True, labels=True)
        out_path = output_dir / f"{img_path.stem}_detected.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
