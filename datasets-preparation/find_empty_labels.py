from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_empty_labels(root: Path) -> list[tuple[Path, Path | None]]:
    empty: list[tuple[Path, Path | None]] = []
    for split in ["train", "valid", "val", "test"]:
        labels_dir = root / split / "labels"
        images_dir = root / split / "images"
        if not labels_dir.exists():
            continue
        for label_path in sorted(labels_dir.glob("*.txt")):
            if label_path.read_text(encoding="utf-8").strip():
                continue
            image_path = find_image(images_dir, label_path.stem) if images_dir.exists() else None
            empty.append((label_path, image_path))
    return empty


def split_empty_labels(items: list[tuple[Path, Path | None]], seed: int) -> dict[str, list[tuple[Path, Path | None]]]:
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.7)
    n_val = round(n * 0.2)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def write_audit_bundle(
    splits: dict[str, list[tuple[Path, Path | None]]],
    output_dir: Path,
    copy_images: bool,
    move: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        split_dir = output_dir / split
        label_dir = split_dir / "labels"
        image_dir = split_dir / "images"
        label_dir.mkdir(parents=True, exist_ok=True)
        if copy_images:
            image_dir.mkdir(parents=True, exist_ok=True)

        manifest_lines: list[str] = []
        for label_path, image_path in items:
            dst_label = label_dir / label_path.name
            if move:
                shutil.move(str(label_path), dst_label)
            else:
                shutil.copy2(label_path, dst_label)

            image_name = ""
            if copy_images and image_path and image_path.exists():
                dst_image = image_dir / image_path.name
                if move:
                    shutil.move(str(image_path), dst_image)
                else:
                    shutil.copy2(image_path, dst_image)
                image_name = image_path.name
            manifest_lines.append(f"{label_path.name},{image_name}")

        (split_dir / "manifest.csv").write_text("label_file,image_file\n" + "\n".join(manifest_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export empty YOLO label files into a reproducible audit bundle.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root containing train/valid/test folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to receive the empty-label audit bundle.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used to split audit files into train/val/test folders.")
    parser.add_argument("--copy-images", action="store_true", help="Also copy matching images into the audit bundle.")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    args = parser.parse_args()

    items = collect_empty_labels(args.root)
    splits = split_empty_labels(items, args.seed)
    print(f"Found {len(items)} empty label files.")
    for split, split_items in splits.items():
        print(f"{split}: {len(split_items)}")

    if not args.yes:
        action = "move" if args.move else "copy"
        answer = input(f"Proceed to {action} files into {args.output_dir}? (y/n): ").strip().lower()
        if answer != "y":
            print("Skipped.")
            return

    write_audit_bundle(splits, args.output_dir, args.copy_images, args.move)
    print(f"Audit bundle written to {args.output_dir}")


if __name__ == "__main__":
    main()
