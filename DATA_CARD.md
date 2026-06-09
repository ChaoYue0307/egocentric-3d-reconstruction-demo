# Data Card

## Source

This project uses one local Xperience-10M sample episode of a pour-over coffee
task. Raw videos, `annotation.hdf5`, and `.rrd` files stay outside the
repository.

## Inputs Used

- `fisheye_cam0.mp4` for sampled frames.
- `annotation.hdf5` for calibration, SLAM poses, and point-cloud preview.

## Scope

The sample is used to prepare reconstruction inputs and explain the pipeline
from frames to COLMAP/NeRF/3DGS. The repo does not claim to finish full neural
rendering training without external tools.

## Limitations

- Single scene and task.
- Fisheye distortion requires careful camera handling.
- Dynamic hands and objects can violate static-scene assumptions.
- COLMAP, NeRFStudio, and 3DGS are optional external dependencies.
