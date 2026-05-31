# Dataset splits

This folder contains the deterministic split files used by the released benchmark.

If the benchmark data cannot be redistributed, reconstruct it from the original sources and regenerate the split files with:

```bash
python tools/make_splits.py \
  --images-dir data/raw/train/images data/raw/valid/images data/raw/test/images \
  --out-dir data/splits \
  --train-count 4156 \
  --val-count 1188 \
  --test-count 594 \
  --seed 42
```

Expected outputs:

- `train.txt`
- `val.txt`
- `test.txt`

Each line should point to one image file relative to the repository root.
