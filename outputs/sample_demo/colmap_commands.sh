#!/usr/bin/env bash
set -euo pipefail

# Generated template. Run from this repo after installing COLMAP.
WORKDIR="outputs/sample_demo"
IMAGE_DIR="$WORKDIR/frames"
DATABASE="$WORKDIR/colmap/database.db"
SPARSE="$WORKDIR/colmap/sparse"
DENSE="$WORKDIR/colmap/dense"

mkdir -p "$SPARSE" "$DENSE"
colmap feature_extractor \
  --database_path "$DATABASE" \
  --image_path "$IMAGE_DIR" \
  --ImageReader.camera_model OPENCV_FISHEYE \
  --SiftExtraction.use_gpu 0

colmap exhaustive_matcher \
  --database_path "$DATABASE" \
  --SiftMatching.use_gpu 0

colmap mapper \
  --database_path "$DATABASE" \
  --image_path "$IMAGE_DIR" \
  --output_path "$SPARSE"

colmap image_undistorter \
  --image_path "$IMAGE_DIR" \
  --input_path "$SPARSE/0" \
  --output_path "$DENSE" \
  --output_type COLMAP
