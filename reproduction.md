# Reproduction Appendix

This appendix records the dataset curation, split protocol, environment, and commands used for the SCA-YOLO experiments submitted to *The Visual Computer*.

## A. Source Datasets

The fused benchmark was reconstructed from two public Roboflow datasets. The original image files are not bundled in this repository; users should download them from the source links and reconstruct the benchmark with the scripts in `datasets-preparation/`.

| Dataset | Link | License | Export/download record | Native annotation format | Source classes retained |
| --- | --- | --- | --- | --- | --- |
| Obstacle Detection | `https://universe.roboflow.com/mtechcv/obstacle-detection-bgabd` | CC BY 4.0 | Roboflow version 16, exported January 30, 2024 at 10:52 AM GMT; local metadata reports 5,376 images | YOLOv8 boxes | `pole`, `signboard`, `stairs`, `tree`, `two-wheeler`, `vehicle` |
| Unstructured Road | `https://universe.roboflow.com/galuh-dataset/unstructured-road` | CC BY 4.0 | Roboflow version 5, exported October 12, 2025 at 9:59 PM GMT; local metadata reports 2,015 images | YOLOv8 oriented/object annotations | `objects`, `people`, `pothole`, `road`, `vehicle` |

The metadata above is taken from the downloaded Roboflow `README.dataset.txt`, `README.roboflow.txt`, and `data.yaml` files.

## B. Annotation Normalization Rules

The source datasets use different annotation tasks and class definitions. The release normalizes them into one YOLO-format object-detection benchmark with ten classes:

```text
pole, signboard, stairs, tree, two-wheeler, vehicle, objects, people, pothole, road
```

The preparation code applies the following rules:

1. Each source image and label file is copied into a merged YOLO directory.
2. A source-specific filename prefix is added, for example `od_` for Obstacle Detection and `ur_` for Unstructured Road, to avoid filename collisions.
3. Class IDs are remapped by class name into the unified class list.
4. Annotation rows whose class names are not in the unified target set are skipped.
5. Annotation rows with polygon-like coordinates are converted to axis-aligned bounding boxes using the minimum and maximum x/y coordinates.
6. Malformed rows, invalid class IDs, and unsupported annotations are skipped.

The only explicit semantic merge supported by the released code is `vehicle`, because both source datasets contain a source class with that exact name. The release does not contain code evidence for finer-grained merges such as `car`, `auto-rickshaw`, or `truck` into `vehicle`, so those merges should not be claimed in the paper unless a separate annotation record is added.

Empty label files were removed from the final benchmark during curation. The released `cleaned-labels/` directory is an audit bundle of the 1,454 empty `.txt` label files detected during cleanup; these files are not included in the final train/validation/test split lists.

## C. Final Benchmark and Split Protocol

After filtering invalid labels, removing empty-label samples, and normalizing class IDs, the final fused benchmark contains 5,938 images and 8,890 annotated instances.

Images were split at the image level with fixed seed `42` and exact counts:

| Split | File | Images |
| --- | --- | ---: |
| Train | `data/splits/train.txt` | 4,156 |
| Validation | `data/splits/val.txt` | 1,188 |
| Test | `data/splits/test.txt` | 594 |

This corresponds approximately to a 7:2:1 split. The checked-in split files contain image paths relative to the repository root and are referenced by `data/data.yaml`.

To regenerate the split files from a reconstructed `data/raw/` directory:

```bash
python tools/make_splits.py \
  --images-dir data/raw/train/images data/raw/valid/images data/raw/test/images \
  --out-dir data/splits \
  --train-count 4156 \
  --val-count 1188 \
  --test-count 594 \
  --seed 42
```

## D. Dataset Reconstruction Commands

Arrange each downloaded source dataset in YOLO layout:

```text
source_dataset/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  test/images/
  test/labels/
  data.yaml
```

Merge the two source datasets:

```bash
python datasets-preparation/merge_datasets.py \
  --sources /path/to/Obstacle_Detection /path/to/Unstructured_Road \
  --prefixes od_ ur_ \
  --output data/raw \
  --force
```

Detect and export empty label files for audit:

```bash
python datasets-preparation/find_empty_labels.py \
  --root data/raw \
  --output-dir cleaned-labels \
  --seed 42 \
  --yes
```

Remove empty label files and their corresponding images from the benchmark:

```bash
python datasets-preparation/clean_empty_labels.py --root data/raw --delete-images --yes
```

Generate basic statistics:

```bash
python datasets-preparation/merged_dataset_statistics.py --root data/raw
python datasets-preparation/class_distribution.py --roots data/raw
```

## E. Environment

The experiments were organized around the following environment:

| Type | Configuration | Version / Parameters |
| --- | --- | --- |
| Hardware | System | Windows 11 25H2 |
| Hardware | GPU | NVIDIA RTX 5060 Ti, 16GB GDDR7 |
| Hardware | CPU | Intel Core i7-14700, 20 cores |
| Software | Python | 3.11.14 |
| Software | PyTorch | 2.10.0.dev20251207+cu130 |
| Software | Torchvision | 0.25.0.dev20251206+cu130 |
| Software | Ultralytics | 8.3.235 |

The same settings are recorded in `settings/paper_reproducibility.yaml`.

Install dependencies:

```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate sca-yolo
```

## F. Implementation Patch

SCA-YOLO adds or modifies the following Ultralytics components:

- `C2f_AK`
- `AKConv`
- `CoordAtt`
- WIoU-v3 loss support
- model parsing support for the new modules

Apply the source overrides after installing `ultralytics==8.3.235`:

```powershell
powershell -ExecutionPolicy Bypass -File tools/apply_ultralytics_patch.ps1
```

## G. Training and Validation

Train the full model with three independent seeds:

```bash
python scripts/train.py \
  --model configs/sca-yolo.yaml \
  --data data/data.yaml \
  --epochs 300 \
  --imgsz 640 \
  --batch 16 \
  --workers 6 \
  --device cuda:0 \
  --seeds 0 1 2
```

Run the ablation experiments:

```bash
python scripts/train_ablation.py \
  --data data/data.yaml \
  --epochs 300 \
  --imgsz 640 \
  --batch 16 \
  --workers 6 \
  --device 0 \
  --seeds 0 1 2
```

Validate a checkpoint:

```bash
python scripts/val.py --weights weights/best.pt --data data/data.yaml --iou 0.7 --max-det 300
```

Training uses `imgsz=640`, `epochs=300`, `batch=16`, `workers=6`, `optimizer=auto`, `lr0=0.01`, `lrf=0.01`, `momentum=0.937`, `weight_decay=0.0005`, and AMP enabled. Mosaic is enabled with `mosaic=1.0` and closed during the last 10 epochs. Rotation, shear, perspective, MixUp, CutMix, and copy-paste are disabled.

Validation and inference use FP32 (`half=False`), `iou=0.7`, `max_det=300`, and no TensorRT or INT8 acceleration.
