# Release manifest

## Public

- `README.md`
- `reproduction.md`
- `requirements.txt`
- `environment.yml`
- `data/data.yaml`
- `data/BENCHMARK.md`
- `data/splits/README.md`
- `data/splits/train.txt`
- `data/splits/val.txt`
- `data/splits/test.txt`
- `cleaned-labels/README.md`
- `cleaned-labels/*/labels/*.txt`
- `cleaned-labels/*/manifest.csv`
- `datasets-preparation/*.py`
- `datasets-preparation/*.yaml`
- `datasets-preparation/README.md`
- `configs/*.yaml`
- `scripts/*.py`
- `settings/paper_reproducibility.yaml`
- `tools/*.py`
- `tools/*.ps1`
- `patches/ultralytics/nn/modules/block.py`
- `patches/ultralytics/nn/modules/__init__.py`
- `patches/ultralytics/nn/tasks.py`
- `patches/ultralytics/utils/loss.py`
- `patches/ultralytics/utils/metrics.py`
- `assets/bus.jpg`
- `weights/README.md`

## Do not publish

- any absolute local dataset path from the author machine
- any path containing a personal username
- `.cache` files and directories
- temporary experiment files
- files containing account, token, or secret values
- large intermediate caches
- local source dataset images and labels under `data/raw/`
- failed experiments not used in the paper
- the full local conda environment under `sca-yolo/`

## Notes

- The modified Ultralytics files are source-level overrides, not a full vendored copy.
- Dataset preparation scripts were parameterized from the local experimental scripts and do not contain local paths.
- `cleaned-labels/` is an audit bundle of empty label files removed during curation; it is not part of the final 5,938-image benchmark split.
- If you release weights, place them in `weights/` and update the README with the exact filename.
