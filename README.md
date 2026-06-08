# Egocentric 3D Reconstruction Demo

Practical 3D/4D reconstruction wrapper for one Xperience-10M egocentric sample.

This repo is intentionally lightweight: it validates frame extraction,
calibration export, SLAM pose export, and command generation locally, while
leaving heavy COLMAP, NeRF, and Gaussian Splatting training as external tools.

It maps directly to:

- multi-view and dynamic scene reconstruction,
- camera calibration and fisheye handling,
- bundle adjustment with COLMAP,
- NeRF and 3D Gaussian Splatting command preparation,
- novel-view synthesis demo planning,
- failure analysis for egocentric video.

Raw videos, `annotation.hdf5`, and `.rrd` files stay outside git.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/reconstruction_demo.py \
  --data-root /Users/chaoyue/Library/CloudStorage/Dropbox/Ropedia/data/sample/xperience-10m-sample \
  --output-dir outputs/sample_demo \
  --frame-stride 180 \
  --max-frames 24
```

The smoke test does not require COLMAP, NeRFStudio, or 3DGS to be installed. It
records whether they are available in `dependency_check.json`.

## Outputs

- `frames/`: sampled JPEG frames for reconstruction input.
- `calibration.json`: camera intrinsics, distortion, and extrinsics from HDF5.
- `slam_poses_tum.txt`: SLAM trajectory in a common TUM-like text format.
- `slam_point_cloud_preview.ply`: small point-cloud preview from HDF5 SLAM.
- `colmap_commands.sh`: generated COLMAP feature/matching/mapper template.
- `nerf_3dgs_templates.sh`: generated NeRFStudio and 3DGS command template.
- `failure_analysis.md`: egocentric reconstruction failure-mode checklist.

## Suggested Next Steps

1. Run the generated COLMAP script after installing COLMAP.
2. Undistort fisheye frames using `calibration.json` before matching.
3. Mask dynamic hands/objects before feature extraction.
4. Train a NeRF or Gaussian Splatting model from COLMAP outputs.
5. Render a short novel-view path and compare it with the SLAM trajectory.
