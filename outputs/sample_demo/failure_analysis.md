# Failure Case Analysis

This generated note records the expected reconstruction risks for the Xperience-10M
sample episode.

## Smoke-Test Context

- Extracted frames: 24
- Exported SLAM poses: 200
- SLAM point cloud points: 4089
- COLMAP installed: False
- NeRFStudio installed: False

## Common Failure Modes

- Motion blur from head-mounted or hand-held camera motion reduces feature repeatability.
- Dynamic hands, kettle, dripper, and water stream violate static-scene assumptions.
- Fisheye distortion needs the correct camera model or pre-undistortion.
- Rolling shutter and exposure changes can hurt bundle adjustment.
- Sparse visual overlap across sampled frames may create disconnected components.
- Reflective or texture-poor kitchen objects can cause noisy points or BA outliers.

## Next Improvements

- Extract denser frames around slow camera motion.
- Mask hands and highly dynamic objects before feature extraction.
- Use calibration from `calibration.json` to seed intrinsics.
- Compare SLAM poses against COLMAP camera poses when COLMAP is installed.
