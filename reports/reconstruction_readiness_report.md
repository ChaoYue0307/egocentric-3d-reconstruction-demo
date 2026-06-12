# Reconstruction Readiness Report

## Motivation

Egocentric reconstruction is difficult because the camera and many foreground
objects move together. Before running heavy reconstruction tools, it is useful
to inspect whether frames, calibration, pose records, and command templates are
coherent — and to quantify frame quality and mask dynamic regions instead of
guessing.

## Method

The preparation script extracts sampled frames, exports calibration, writes a
TUM-like SLAM pose file, creates a point-cloud preview, and generates COLMAP,
NeRFStudio, and 3D Gaussian Splatting command templates.

Three diagnostic stages were added:

- **Frame quality:** per-frame Laplacian-variance blur (computed on a central
  crop to exclude the fisheye border), exposure statistics, and ORB feature
  matches between consecutive sampled frames, with the weakest pairs surfaced.
- **Hand masks:** MANO hand joints are projected through the rig extrinsic
  chain and SLAM pose into COLMAP-convention masks. The extrinsic composition
  is selected empirically from four candidates, scored by in-bounds fraction
  and arm's-length depth plausibility; the selection evidence and a visual
  preview ship with every run. On the sample episode the selected chain is
  `inv(cam_body) @ world_body` with ~12% of pixels masked on cam1.
- **Trajectory metrics:** COLMAP poses are converted to camera centers
  (`C = -R^T t`), Sim(3)-aligned to SLAM positions with the Umeyama method,
  and scored with ATE RMSE and RPE RMSE plus the recovered scale.

On the sample episode, 24 sampled cam0 frames show 0–1 blurry frames and a
mean of ~719 ORB matches between consecutive frames — comfortable overlap for
COLMAP at stride 180.

## Artifacts

- `frames_manifest.json`: frame-to-pose alignment.
- `calibration.json`: camera intrinsics and distortion.
- `frame_quality.json`: blur, exposure, and overlap diagnostics.
- `hand_mask_report.json` + `hand_masks/`: dynamic-region masks with
  convention-selection evidence and preview images.
- `slam_summary.json`: available pose and point-cloud counts.
- `colmap_summary.json`: optional sparse model summary.
- `colmap_vs_slam.json`: optional Sim(3)-aligned ATE/RPE comparison.
- `reconstruction_manifest.svg`: quick trajectory visual.

## Interpretation

A ready episode has enough sharp frames, valid calibration, a non-empty pose
track, and visual overlap across sampled frames. The prepared artifacts are a
diagnostic stage, not a guarantee of high-quality novel-view synthesis.

Two findings worth noting on this rig:

- `fisheye_cam0` faces away from the tabletop; `cam1` and the stereo pair see
  the workspace. Camera selection is therefore a reconstruction decision:
  cameras that do not see hands observe more static scene.
- The mocap-to-camera registration carries a visible offset of tens of pixels.
  The masks compensate with generous dilation and are explicitly approximate;
  a learned hand segmenter or refined hand-eye calibration is the production
  path. The repo treats this as a teaching artifact: extrinsic conventions
  must be verified empirically, never assumed from field names.

## Failure Modes

- Motion blur reduces feature matching reliability.
- Dynamic hands and tools violate static-scene assumptions.
- Fisheye distortion can hurt COLMAP unless modeled correctly.
- Sparse overlap can fragment the reconstruction.
- Extrinsic-convention mistakes silently misplace projections.
- Comparing COLMAP tvecs directly to SLAM positions (without camera centers
  and Sim(3) alignment) produces meaningless deltas.

## Next Work

- Run COLMAP with and without hand masks and compare registered-image counts
  and reprojection error — the masks' value, measured.
- Replace projected-joint masks with a learned hand segmenter.
- Extend quality diagnostics with per-pair geometric verification (inlier
  ratios after RANSAC essential-matrix fitting).
