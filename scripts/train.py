from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the full SCA-YOLO model.")
    parser.add_argument("--model", default=str(ROOT / "configs" / "sca-yolo.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "data.yaml"))
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Run multiple seeds, e.g. 0 1 2.")
    parser.add_argument("--project", default=str(ROOT / "runs"))
    parser.add_argument("--name", default="sca-yolo")
    return parser.parse_args()


def train_once(args: argparse.Namespace, seed: int) -> None:
    run_name = f"{args.name}_seed{seed}"
    os.environ["PYTHONHASHSEED"] = str(seed)

    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=seed,
        optimizer="auto",
        amp=True,
        momentum=0.937,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        project=args.project,
        name=run_name,
        exist_ok=True,
    )


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    seeds = args.seeds if args.seeds else [args.seed]
    for seed in seeds:
        train_once(args, seed)


if __name__ == "__main__":
    main()
