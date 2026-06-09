# Reconstruction Readiness Report

## Motivation

Egocentric reconstruction is difficult because the camera and many foreground
objects move together. Before running heavy reconstruction tools, it is useful
to inspect whether frames, calibration, pose records, and command templates are
coherent.

## Method

The preparation script extracts sampled frames, exports calibration, writes a
TUM-like SLAM pose file, creates a point-cloud preview, and generates COLMAP,
NeRFStudio, and 3D Gaussian Splatting command templates. If a COLMAP sparse text
model is available, the parser summarizes cameras, registered images, sparse
points, reprojection error, and point-cloud bounds.

## Artifacts

- `frames_manifest.json`: frame-to-pose alignment.
- `calibration.json`: camera intrinsics and distortion.
- `slam_summary.json`: available pose and point-cloud counts.
- `colmap_summary.json`: optional sparse model summary.
- `reconstruction_manifest.svg`: quick trajectory visual.

## Interpretation

A ready episode has enough sharp frames, valid calibration, a non-empty pose
track, and visual overlap across sampled frames. The prepared artifacts are a
diagnostic stage, not a guarantee of high-quality novel-view synthesis.

## Failure Modes

- Motion blur reduces feature matching reliability.
- Dynamic hands and tools violate static-scene assumptions.
- Fisheye distortion can hurt COLMAP unless modeled correctly.
- Sparse overlap can fragment the reconstruction.

## Next Work

- Compare COLMAP camera poses against SLAM poses.
- Add masks for hands and moving objects before feature extraction.
- Add image-quality diagnostics for blur, overlap, and exposure.
