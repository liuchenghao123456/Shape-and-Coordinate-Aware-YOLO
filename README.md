# SCA-YOLO

This repository contains the code, configuration files, benchmark definition, and reproduction instructions for SCA-YOLO.

## Citation note

This repository is the official release associated with the manuscript submitted to *The Visual Computer*:

> SCA-YOLO: an improved YOLO-based method for road scene object detection.

If this repository is archived on Zenodo, cite the archived version by its DOI. The DOI should be added here after repository deposition.

## Contents

- `configs/`: model definitions for the full method and ablations
- `patches/ultralytics/`: source changes required to reproduce SCA-YOLO
- `scripts/`: training and inference entry points
- `data/data.yaml`: dataset definition and split-file entry points
- `data/BENCHMARK.md`: benchmark class mapping, statistics, and split protocol
- `data/splits/`: deterministic train/val/test split files
- `cleaned-labels/`: empty-label audit bundle released with the benchmark
- `datasets-preparation/`: scripts for reconstructing and cleaning the benchmark from source datasets
- `tools/`: helper scripts for split generation and patch application
- `datasets-preparation/find_empty_labels.py`: parameterized empty-label extraction utility
- `weights/`: place released weights here when redistribution is allowed
- `reproduction.md`: reproduction protocol for the submitted experiments

## Install

```bash
pip install -r requirements.txt
```

If you prefer Conda:

```bash
conda env create -f environment.yml
```

## Apply the Ultralytics patch

The custom modules and loss changes live under `patches/ultralytics/`.
After installing `ultralytics==8.3.235`, apply the patch files with:

```powershell
powershell -ExecutionPolicy Bypass -File tools/apply_ultralytics_patch.ps1
```

## Dataset layout

Put the reconstructed dataset under `data/raw/` with this layout:

```text
data/raw/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

The released `data/data.yaml` reads the checked-in split files under `data/splits/`.
The released benchmark uses a fixed image-level split of 4,156 training images, 1,188 validation images, and 594 test images.

If redistribution is restricted, reconstruct the benchmark from the original sources and regenerate the split files with `tools/make_splits.py`.

Dataset preparation scripts are provided in `datasets-preparation/`. For example:

```bash
python datasets-preparation/merge_datasets.py --sources /path/to/source1 /path/to/source2 /path/to/source3 --prefixes od_ ur_ tr_ --output data/raw --force
python datasets-preparation/clean_empty_labels.py --root data/raw --yes
python datasets-preparation/find_empty_labels.py --root data/raw --output-dir cleaned-labels --seed 42
python datasets-preparation/merged_dataset_statistics.py --root data/raw
```

For the paper-reported settings, use three independent runs with seeds `0`, `1`, and `2`.

## Train

```bash
python scripts/train.py
python scripts/train_ablation.py
```

## Inference

```bash
python scripts/predict.py --weights weights/best.pt --source assets/bus.jpg
python scripts/val.py --weights weights/best.pt --data data/data.yaml
python scripts/predict_batch.py --weights weights/best.pt --source data/raw/test/images
```

## Weights

Trained weights are not bundled in this package. If you have permission to release them, place them in `weights/`.

## DOI

Add the Zenodo DOI here after archiving the repository.
