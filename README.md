# Egocentric 3D Reconstruction Demo

Learn the moving parts behind 3D reconstruction from first-person video:
frames, camera calibration, SLAM poses, COLMAP, NeRF, and 3D Gaussian Splatting.

The script prepares the ingredients that reconstruction systems need. It
extracts video frames, exports camera calibration, writes a SLAM trajectory,
generates COLMAP and neural-rendering command templates, and explains common
failure cases for egocentric footage.

## Interactive Tutorial

Open the visual walkthrough:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000/docs/`.

The page explains the pipeline stage by stage and shows why egocentric
reconstruction is harder than static multi-view reconstruction. A glossary lives
in `docs/concepts.md`.

## What You Will Learn

- **Frame extraction:** turning a video into images that reconstruction tools can match.
- **Camera calibration:** intrinsics, distortion, and extrinsics that describe each camera.
- **SLAM pose:** an estimated camera path through 3D space.
- **COLMAP:** feature matching plus bundle adjustment to estimate cameras and sparse points.
- **NeRF:** a neural field that learns density and color for novel-view rendering.
- **3D Gaussian Splatting:** a point-based neural renderer that can produce fast novel views.
- **Failure analysis:** why motion blur, dynamic hands, fisheye distortion, and weak overlap matter.

## Data

Raw videos, `annotation.hdf5`, and `.rrd` files stay outside git. Set
`DATA_ROOT` to your local Xperience-10M sample:

```bash
export DATA_ROOT=/path/to/xperience-10m-sample
```

The expected directory contains `annotation.hdf5` and camera videos such as
`fisheye_cam0.mp4`.
See `DATA_NOTICE.md` for the minimal data contract.

## Run The Preparation Script

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/reconstruction_demo.py \
  --data-root "$DATA_ROOT" \
  --output-dir outputs/sample_demo \
  --frame-stride 180 \
  --max-frames 24
```

This command does not require COLMAP, NeRFStudio, or 3DGS to be installed. It
checks whether those tools are available and still writes the tutorial artifacts.

## Outputs

| Output | Why It Matters |
| --- | --- |
| `frames/` | sampled JPEG images used as reconstruction input |
| `calibration.json` | camera intrinsics, distortion, and rig transforms |
| `slam_poses_tum.txt` | camera trajectory in a common TUM-like text format |
| `slam_point_cloud_preview.ply` | lightweight preview of the existing SLAM point cloud |
| `colmap_commands.sh` | COLMAP feature, matching, mapping, and undistortion commands |
| `nerf_3dgs_templates.sh` | command templates for NeRFStudio and Gaussian Splatting |
| `failure_analysis.md` | checklist for diagnosing egocentric reconstruction failures |

## Reading The Pipeline

1. Extract frames from the egocentric video.
2. Use calibration to understand how pixels map to camera rays.
3. Use COLMAP to match visual features and optimize camera poses.
4. Use NeRF or 3DGS to learn a renderable 3D representation.
5. Render novel views and compare them with the SLAM trajectory.

Egocentric video is difficult because the scene is not fully static: hands,
kettle, water, and coffee tools move while the camera moves too. Treat these
moving regions carefully when moving from this preparation step to full
reconstruction.
