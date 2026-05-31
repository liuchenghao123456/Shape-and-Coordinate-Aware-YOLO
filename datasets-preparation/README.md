# Dataset Preparation

This folder contains the scripts used to merge and clean the source datasets before training SCA-YOLO.

The scripts are designed to work on local source data that is not redistributed here.

## Expected source layout

Each source dataset should already contain:

```text
dataset_root/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
  data.yaml
```

If your source dataset uses `val/` instead of `valid/`, adjust the script arguments accordingly.

## Merge datasets

```bash
python merge_datasets.py --sources /path/to/dataset1 /path/to/dataset2 --prefixes od_ ur_ --output data/raw --force
```

The script can merge two or more datasets and will remap classes by name.

## Clean empty labels

```bash
python clean_empty_labels.py --root data/raw --yes
```

## Export empty-label audit files

```bash
python find_empty_labels.py --root data/raw --output-dir cleaned-labels --seed 42 --yes
```

## Statistics

```bash
python merged_dataset_statistics.py --root data/raw
python class_distribution.py --roots data/raw data/raw_secondary
```

## Find the densest label file

```bash
python find_most_labels.py --labels-dir data/raw/train/labels --images-dir data/raw/train/images --output-dir outputs/most_labels
```
