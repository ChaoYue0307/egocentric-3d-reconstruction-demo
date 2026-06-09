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
