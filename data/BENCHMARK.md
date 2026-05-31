# Benchmark Definition

The fused benchmark is reconstructed from two Roboflow YOLO-format source datasets:

| Source dataset | Source URL | License | Export record | Source classes used |
| --- | --- | --- | --- | --- |
| `Obstacle Detection` | `https://universe.roboflow.com/mtechcv/obstacle-detection-bgabd` | CC BY 4.0 | exported on January 30, 2024 at 10:52 AM GMT; Roboflow version 16; 5,376 images | `pole`, `signboard`, `stairs`, `tree`, `two-wheeler`, `vehicle` |
| `Unstructured Road` | `https://universe.roboflow.com/galuh-dataset/unstructured-road` | CC BY 4.0 | exported on October 12, 2025 at 9:59 PM GMT; Roboflow version 5; 2,015 images | `objects`, `people`, `pothole`, `road`, `vehicle` |

During fusion, class IDs are remapped by class name. Polygon-style annotations with more than four coordinate values are converted to axis-aligned YOLO boxes by taking the minimum and maximum polygon coordinates. Images are copied with source-specific filename prefixes to avoid filename collisions.

The final unified label set contains ten classes:

1. `pole`
2. `signboard`
3. `stairs`
4. `tree`
5. `two-wheeler`
6. `vehicle`
7. `objects`
8. `people`
9. `pothole`
10. `road`

The explicit cross-source semantic merge in the released preparation code is the shared `vehicle` class, which appears in both source datasets and is mapped to one unified `vehicle` category. No code-supported merge such as `car`, `auto-rickshaw`, or `truck` into `vehicle` is present in this release. Classes outside the retained source class lists are discarded by the class-name filter in `datasets-preparation/merge_datasets.py`; invalid or unsupported annotation lines are skipped.

Empty label files were treated as non-annotated samples and removed from the training benchmark during curation. The released `cleaned-labels/` directory is an audit bundle containing the 1,454 empty label files detected during this cleanup step. These files are not part of the final training, validation, or test image lists.

After filtering invalid labels, removing empty-label samples, and normalizing category IDs, the benchmark contains 5,938 images and 8,890 annotated instances. The released split protocol uses fixed seed `42`, image-level assignment, and exact split counts corresponding approximately to a 7:2:1 division:

- `train.txt`: 4,156 images
- `val.txt`: 1,188 images
- `test.txt`: 594 images

The split files are stored in `data/splits/` and are referenced by `data/data.yaml`.
