from __future__ import annotations

import argparse
import random
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(images_dirs: list[Path]) -> list[Path]:
    images: list[Path] = []
    for images_dir in images_dirs:
        images.extend(sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS))
    return images


def split_entry(path: Path, source_root: Path, path_prefix: str | None) -> str:
    rel = path.resolve().relative_to(source_root.resolve()).as_posix()
    if path_prefix:
        return f"{path_prefix.rstrip('/')}/{rel}"
    return rel


def write_split(paths: list[Path], out_file: Path, source_root: Path, path_prefix: str | None) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8", newline="\n") as f:
        for p in paths:
            f.write(split_entry(p, source_root, path_prefix))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic YOLO split files.")
    parser.add_argument("--images-dir", nargs="+", type=Path, required=True, help="One or more directories with source images.")
    parser.add_argument("--source-root", type=Path, default=None, help="Root used to make split entries relative.")
    parser.add_argument("--path-prefix", default=None, help="Optional prefix prepended to each split entry, e.g. data/raw.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write train/val/test txt files.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise SystemExit("Split ratios must sum to 1.0.")

    source_root = args.source_root or Path.cwd()
    images = collect_images(args.images_dir)
    if not images:
        raise SystemExit(f"No images found under {args.images_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(images)

    n = len(images)
    explicit_counts = [args.train_count, args.val_count, args.test_count]
    if any(count is not None for count in explicit_counts):
        if not all(count is not None for count in explicit_counts):
            raise SystemExit("Specify all of --train-count, --val-count, and --test-count, or none of them.")
        n_train = int(args.train_count)
        n_val = int(args.val_count)
        n_test = int(args.test_count)
        if n_train + n_val + n_test != n:
            raise SystemExit(f"Requested split counts sum to {n_train + n_val + n_test}, but found {n} images.")
    else:
        n_train = int(n * args.train_ratio)
        n_val = round(n * args.val_ratio)
        n_test = n - n_train - n_val

    train = images[:n_train]
    val = images[n_train : n_train + n_val]
    test = images[n_train + n_val : n_train + n_val + n_test]

    write_split(train, args.out_dir / "train.txt", source_root, args.path_prefix)
    write_split(val, args.out_dir / "val.txt", source_root, args.path_prefix)
    write_split(test, args.out_dir / "test.txt", source_root, args.path_prefix)

    print(f"Wrote {len(train)} train, {len(val)} val, {len(test)} test images to {args.out_dir}")


if __name__ == "__main__":
    main()
