from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def clean_empty_annotations(root: Path, delete_images: bool, yes: bool) -> None:
    for split in ["train", "valid", "val", "test"]:
        labels_dir = root / split / "labels"
        images_dir = root / split / "images"
        if not labels_dir.exists() or not images_dir.exists():
            continue

        empty_pairs: list[tuple[Path, Path | None]] = []
        for label_path in labels_dir.glob("*.txt"):
            if label_path.read_text(encoding="utf-8").strip():
                continue
            image_path = find_image(images_dir, label_path.stem)
            empty_pairs.append((label_path, image_path))

        if not empty_pairs:
            print(f"{split}: no empty labels found")
            continue

        print(f"{split}: found {len(empty_pairs)} empty label files")
        for label_path, image_path in empty_pairs:
            print(f"  {label_path.name}")
            if image_path:
                print(f"  {image_path.name}")

        if not yes:
            answer = input("Delete these files? (y/n): ").strip().lower()
            if answer != "y":
                print("Skipped")
                continue

        for label_path, image_path in empty_pairs:
            label_path.unlink(missing_ok=True)
            if delete_images and image_path and image_path.exists():
                image_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove empty YOLO label files and optionally images.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root.")
    parser.add_argument("--delete-images", action="store_true", help="Also delete matching images.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    args = parser.parse_args()
    clean_empty_annotations(args.root, args.delete_images, args.yes)


if __name__ == "__main__":
    main()
