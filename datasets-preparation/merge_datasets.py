from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_class_names(dataset_root: Path) -> list[str]:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml in {dataset_root}")
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"No class names found in {data_yaml}")
    return [str(name) for name in names]


def resolve_split_dir(root: Path, split: str) -> Path | None:
    candidates = [split]
    if split == "val":
        candidates.append("valid")
    elif split == "valid":
        candidates.append("val")
    for candidate in candidates:
        folder = root / candidate
        if folder.exists():
            return folder
    return None


def iter_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def parse_line(line: str, class_names: list[str], class_map: dict[str, int]) -> str | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
    except ValueError:
        return None
    if class_id < 0 or class_id >= len(class_names):
        return None

    class_name = class_names[class_id]
    if class_name not in class_map:
        return None

    coords = parts[1:]
    if len(coords) > 4:
        values = [float(v) for v in coords]
        if len(values) % 2 != 0:
            return None
        xs = values[0::2]
        ys = values[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min
        coords_out = [cx, cy, w, h]
    else:
        try:
            coords_out = [float(v) for v in coords[:4]]
        except ValueError:
            return None

    new_id = class_map[class_name]
    return f"{new_id} " + " ".join(f"{v:.6f}" for v in coords_out)


def merge_dataset(
    sources: list[Path],
    output: Path,
    prefixes: list[str],
    force: bool,
) -> None:
    if output.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    class_lists = [load_class_names(src) for src in sources]
    merged_names: list[str] = []
    for names in class_lists:
        for name in names:
            if name not in merged_names:
                merged_names.append(name)
    class_map = {name: idx for idx, name in enumerate(merged_names)}

    for split in ["train", "val", "test"]:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "labels": 0, "empty_labels": 0}

    for src_idx, src in enumerate(sources):
        prefix = prefixes[src_idx]
        class_names = class_lists[src_idx]
        for split in ["train", "val", "test"]:
            split_dir = resolve_split_dir(src, split)
            if split_dir is None:
                continue
            img_dir = split_dir / "images"
            label_dir = split_dir / "labels"
            if not img_dir.exists() or not label_dir.exists():
                continue

            for img_path in iter_images(img_dir):
                label_path = label_dir / f"{img_path.stem}.txt"
                out_img = output / split / "images" / f"{prefix}{img_path.name}"
                out_label = output / split / "labels" / f"{prefix}{img_path.stem}.txt"

                shutil.copy2(img_path, out_img)
                stats["images"] += 1

                out_lines: list[str] = []
                if label_path.exists():
                    for line in label_path.read_text(encoding="utf-8").splitlines():
                        parsed = parse_line(line, class_names, class_map)
                        if parsed:
                            out_lines.append(parsed)
                else:
                    stats["empty_labels"] += 1

                out_label.write_text("\n".join(out_lines), encoding="utf-8")
                stats["labels"] += 1

    merged_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(merged_names),
        "names": merged_names,
    }
    (output / "data.yaml").write_text(yaml.safe_dump(merged_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"Merged datasets saved to: {output}")
    print(f"Images copied: {stats['images']}")
    print(f"Label files written: {stats['labels']}")
    print(f"Missing labels converted to empty files: {stats['empty_labels']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple YOLO datasets by class name.")
    parser.add_argument("--sources", nargs="+", required=True, type=Path, help="Source dataset roots.")
    parser.add_argument("--prefixes", nargs="+", required=True, help="Prefix for each source dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Output merged dataset root.")
    parser.add_argument("--force", action="store_true", help="Overwrite the output directory if it exists.")
    args = parser.parse_args()

    if len(args.sources) != len(args.prefixes):
        raise SystemExit("--sources and --prefixes must have the same length.")

    merge_dataset(args.sources, args.output, args.prefixes, args.force)


if __name__ == "__main__":
    main()
