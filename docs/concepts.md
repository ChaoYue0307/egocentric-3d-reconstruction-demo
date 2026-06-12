# Concepts: Egocentric 3D Reconstruction

## Frame Extraction

Most reconstruction tools operate on images. Frame extraction samples still
images from the video so feature matchers and neural renderers can process them.

Dense sampling gives more overlap but costs more time. Sparse sampling is faster
but can fail if adjacent images no longer share enough visual content.

## Camera Intrinsics

Intrinsics describe how a camera maps 3D rays to pixels. Common values include:

- focal length,
- principal point,
- distortion parameters.

Fisheye cameras are useful for egocentric capture because they see a wide field
of view. They also require careful distortion handling.

## Camera Extrinsics

Extrinsics describe where a camera is in 3D space and how it is oriented. In a
multi-camera rig, extrinsics explain how cameras are positioned relative to each
other.

## SLAM

SLAM means simultaneous localization and mapping. It estimates the camera path
while building a sparse map of the environment. The sample annotation already
contains SLAM camera poses and a point cloud preview.

## COLMAP

COLMAP is a structure-from-motion and multi-view stereo system. It usually runs:

1. feature extraction,
2. feature matching,
3. mapping and bundle adjustment,
4. optional image undistortion and dense reconstruction.

Bundle adjustment optimizes camera poses and 3D points so projected points align
with matched image features.

## NeRF

NeRF represents a scene as a neural function. Given a 3D point and viewing
direction, it predicts density and color. Rendering a new view means sampling
many camera rays through this learned field.

## 3D Gaussian Splatting

3D Gaussian Splatting represents a scene with many colored 3D Gaussians. It is
often faster to render than classic NeRF and can produce high-quality novel
views when camera poses are accurate.

## Why Egocentric Reconstruction Is Hard

First-person footage violates many assumptions of clean multi-view
reconstruction:

- the camera moves quickly,
- hands and objects move independently,
- motion blur reduces feature matches,
- fisheye distortion needs the right model,
- dynamic objects can confuse bundle adjustment,
- the camera may not revisit the same area from enough viewpoints.

Good reconstruction starts with clean frame sampling, calibration awareness, and
explicit failure analysis.

## Frame Quality Diagnostics

Reconstruction fails quietly when inputs are weak. Three cheap numbers predict
most failures before COLMAP runs: Laplacian variance (blur — low variance
means few sharp edges), exposure clipping fractions, and ORB feature matches
between consecutive frames (overlap — the matches COLMAP will actually rely
on). Measuring them on a central crop keeps the black fisheye border from
skewing the statistics.

## Extrinsic Convention Verification

A transform stored as `T_c_b` might map body→camera or camera→body, and the
SLAM pose might be world→body or body→world. Names do not disambiguate;
geometry does. Projecting known 3D points (here, mocap hand joints) under each
candidate composition and scoring in-bounds fraction plus depth plausibility
selects the right chain empirically — and the wide fisheye field of view makes
in-bounds alone insufficient, which is why the depth prior matters.

## Dynamic-Object Masking

COLMAP assumes a static scene; hands and manipulated objects violate it.
A mask (white = use, black = ignore) removes those pixels from feature
extraction. Masks from projected mocap are approximate — registration offsets
of tens of pixels are normal — so dilation absorbs error and a visual preview
is mandatory before trusting them.

## ATE, RPE, and Sim(3) Alignment

COLMAP trajectories live in an arbitrary frame and scale, so comparing raw
translations to SLAM is meaningless. First convert COLMAP world-to-camera
poses to camera centers (`C = -R^T t`), then align with a similarity
transform (Umeyama). ATE RMSE measures global consistency after alignment;
RPE RMSE measures local drift between consecutive poses; the recovered scale
is itself diagnostic — a scale far from any physical expectation signals a
degenerate reconstruction.
