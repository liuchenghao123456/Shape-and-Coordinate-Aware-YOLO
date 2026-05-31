from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def count_labels(root: Path) -> tuple[Counter, int]:
    counter: Counter[int] = Counter()
    total = 0
    for split in ["train", "valid", "val", "test"]:
        labels_dir = root / split / "labels"
        if not labels_dir.exists():
            continue
        for label_path in labels_dir.glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue
                counter[class_id] += 1
                total += 1
    return counter, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Print merged dataset statistics.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--names", nargs="*", default=None, help="Optional class names in order.")
    args = parser.parse_args()

    counts, total = count_labels(args.root)
    names = args.names or [str(i) for i in sorted(counts.keys())]
    print(f"Dataset: {args.root}")
    print(f"Total instances: {total}")
    for idx, name in enumerate(names):
        print(f"{idx:>2}  {name:<20} {counts[idx]}")


if __name__ == "__main__":
    main()
