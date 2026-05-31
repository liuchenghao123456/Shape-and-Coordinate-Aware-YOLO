# Empty-label audit bundle

This directory contains empty YOLO label files detected during dataset curation. It is provided as an audit record only and is not part of the final 5,938-image benchmark split files in `data/splits/`.

The files were redistributed into this audit bundle with fixed seed `42` and an approximate 7:2:1 division:

- `train/labels`: 1017 files
- `val/labels`: 291 files
- `test/labels`: 146 files

Each split includes a `manifest.csv` file with the original source split folder recorded before redistribution.
