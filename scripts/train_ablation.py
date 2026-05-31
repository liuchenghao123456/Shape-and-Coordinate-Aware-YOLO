from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

TASKS = [
    {"yaml": "sca-yolo.yaml", "name": "Exp_Full_SCA", "wiou": True},
    {"yaml": "sca-no-ak.yaml", "name": "Abl_No_AK", "wiou": True},
    {"yaml": "sca-no-ca.yaml", "name": "Abl_No_CA", "wiou": True},
    {"yaml": "sca-no-wiou.yaml", "name": "Abl_No_WIoU", "wiou": False},
    {"yaml": "sca-only-ak.yaml", "name": "Ctr_Only_AK", "wiou": False},
    {"yaml": "sca-only-ca.yaml", "name": "Ctr_Only_CA", "wiou": False},
    {"yaml": "sca-only-wiou.yaml", "name": "Ctr_Only_WIoU", "wiou": True},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ablation study.")
    parser.add_argument("--data", default=str(ROOT / "data" / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Run multiple seeds, e.g. 0 1 2.")
    parser.add_argument("--project", default=str(ROOT / "runs"))
    parser.add_argument("--task", choices=[t["yaml"] for t in TASKS], help="Run one configuration only.")
    return parser.parse_args()


def run_one(task: dict[str, object], args: argparse.Namespace, seed: int) -> None:
    os.environ["USE_WIOU"] = "True" if task["wiou"] else "False"
    os.environ["PYTHONHASHSEED"] = str(seed)
    model = YOLO(str(ROOT / "configs" / str(task["yaml"])))
    model.train(
        data=args.data,
        name=f"{task['name']}_seed{seed}",
        project=args.project,
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
        exist_ok=True,
    )
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    seeds = args.seeds if args.seeds else [args.seed]

    if args.task:
        task = next(t for t in TASKS if t["yaml"] == args.task)
        for seed in seeds:
            run_one(task, args, seed)
        return

    for seed in seeds:
        for task in TASKS:
            run_one(task, args, seed)


if __name__ == "__main__":
    main()
