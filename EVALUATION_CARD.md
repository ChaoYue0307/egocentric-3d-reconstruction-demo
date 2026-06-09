# Evaluation Card

## Task

Prepare an egocentric video episode for 3D reconstruction and inspect whether
the camera trajectory, calibration, and frames are coherent enough for mapping.

## Artifacts To Inspect

- Extracted frames: coverage and visual sharpness.
- `calibration.json`: camera intrinsics and distortion model.
- `slam_summary.json`: available camera poses and point-cloud statistics.
- `frames_manifest.json`: frame-to-pose alignment.
- `reconstruction_manifest.svg`: quick visual check of trajectory coverage.

## Success Criteria

A good preparation run has temporally ordered frames, valid calibration, a
non-empty pose set, and a frame manifest that links images to nearby poses.

## Known Failure Modes

Fast head motion, motion blur, reflective surfaces, hands near the lens, and
dynamic objects can make feature matching or neural rendering unstable.
