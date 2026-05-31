from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_NAMES = ["pole", "signboard", "stairs", "tree", "two-wheeler", "vehicle", "objects", "people", "pothole", "road"]


def count_root(root: Path) -> Counter:
    counter: Counter[int] = Counter()
    for split in ["train", "valid", "val", "test"]:
        labels_dir = root / split / "labels"
        if not labels_dir.exists():
            continue
        for label_path in labels_dir.glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        counter[int(parts[0])] += 1
                    except ValueError:
                        continue
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot class distribution for one or more datasets.")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--names", nargs="*", default=DEFAULT_NAMES)
    args = parser.parse_args()

    for root in args.roots:
        counts = count_root(root)
        values = [counts[i] for i in range(len(args.names))]
        print(f"\nDataset: {root}")
        for idx, name in enumerate(args.names):
            print(f"{idx:>2}  {name:<15} {values[idx]}")

        plt.figure(figsize=(10, 4))
        plt.bar(args.names, values)
        plt.title(f"Class distribution - {root.name}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
