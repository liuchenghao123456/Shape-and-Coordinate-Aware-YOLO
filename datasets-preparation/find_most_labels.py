from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the label file with the most instances.")
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    max_count = -1
    max_label = None
    for label_path in sorted(args.labels_dir.glob("*.txt")):
        count = sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if count > max_count:
            max_count = count
            max_label = label_path

    if max_label is None:
        print("No label files found.")
        return

    image_path = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        candidate = args.images_dir / f"{max_label.stem}{ext}"
        if candidate.exists():
            image_path = candidate
            break

    print(f"Max labels file: {max_label.name} ({max_count} instances)")
    if image_path:
        print(f"Matching image: {image_path.name}")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(max_label, args.output_dir / max_label.name)
        if image_path:
            shutil.copy2(image_path, args.output_dir / image_path.name)


if __name__ == "__main__":
    main()
