#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np


def as_jsonable(value):
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return item
    return arr.tolist()


def check_dependencies() -> dict:
    names = ["colmap", "ns-process-data", "ns-train", "instant-ngp", "python"]
    return {name: {"available": shutil.which(name) is not None, "path": shutil.which(name)} for name in names}


def extract_frames(video_path: Path, output_dir: Path, stride: int, max_frames: int) -> list[dict]:
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    rows, frame_idx, saved = [], 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                name = f"frame_{saved:05d}_src_{frame_idx:06d}.jpg"
                out = frame_dir / name
                cv2.imwrite(str(out), frame)
                rows.append({"image": str(out.relative_to(output_dir)), "source_frame": frame_idx})
                saved += 1
                if max_frames and saved >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()
    return rows


def export_calibration(annotation: Path, output_dir: Path) -> dict:
    payload = {}
    with h5py.File(annotation, "r") as h5:
        cal = h5["calibration"]
        for cam_name in sorted(cal.keys()):
            cam = cal[cam_name]
            payload[cam_name] = {key: as_jsonable(cam[key][()]) for key in cam.keys()}
    (output_dir / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def export_slam(annotation: Path, output_dir: Path, max_rows: int = 200) -> dict:
    with h5py.File(annotation, "r") as h5:
        names = [np.asarray(x).tobytes().decode("utf-8", errors="replace").strip("\x00") for x in h5["slam/frame_names"][:max_rows]]
        trans = np.asarray(h5["slam/trans_xyz"][:max_rows], dtype=float)
        quat = np.asarray(h5["slam/quat_wxyz"][:max_rows], dtype=float)
        point_cloud = np.asarray(h5["slam/point_cloud"], dtype=float)
    pose_path = output_dir / "slam_poses_tum.txt"
    with pose_path.open("w", encoding="utf-8") as fp:
        fp.write("# timestamp tx ty tz qx qy qz qw\n")
        for name, t, q in zip(names, trans, quat):
            timestamp = name.rsplit(".", 1)[0]
            fp.write(f"{timestamp} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} {q[0]:.8f}\n")
    ply_path = output_dir / "slam_point_cloud_preview.ply"
    preview = point_cloud[: min(len(point_cloud), 4000)]
    with ply_path.open("w", encoding="utf-8") as fp:
        fp.write("ply\nformat ascii 1.0\n")
        fp.write(f"element vertex {len(preview)}\n")
        fp.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in preview:
            fp.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
    summary = {
        "num_exported_poses": len(names),
        "num_point_cloud_points": int(len(point_cloud)),
        "pose_file": str(pose_path.relative_to(output_dir)),
        "point_cloud_preview": str(ply_path.relative_to(output_dir)),
    }
    (output_dir / "slam_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_colmap_commands(output_dir: Path, camera_model: str = "OPENCV_FISHEYE") -> None:
    text = f"""#!/usr/bin/env bash
set -euo pipefail

# Generated template. Run from this repo after installing COLMAP.
WORKDIR="{output_dir}"
IMAGE_DIR="$WORKDIR/frames"
DATABASE="$WORKDIR/colmap/database.db"
SPARSE="$WORKDIR/colmap/sparse"
DENSE="$WORKDIR/colmap/dense"

mkdir -p "$SPARSE" "$DENSE"
colmap feature_extractor \\
  --database_path "$DATABASE" \\
  --image_path "$IMAGE_DIR" \\
  --ImageReader.camera_model {camera_model} \\
  --SiftExtraction.use_gpu 0

colmap exhaustive_matcher \\
  --database_path "$DATABASE" \\
  --SiftMatching.use_gpu 0

colmap mapper \\
  --database_path "$DATABASE" \\
  --image_path "$IMAGE_DIR" \\
  --output_path "$SPARSE"

colmap image_undistorter \\
  --image_path "$IMAGE_DIR" \\
  --input_path "$SPARSE/0" \\
  --output_path "$DENSE" \\
  --output_type COLMAP
"""
    (output_dir / "colmap_commands.sh").write_text(text, encoding="utf-8")


def write_nerf_3dgs_templates(output_dir: Path) -> None:
    text = f"""#!/usr/bin/env bash
set -euo pipefail

# NeRFStudio path after COLMAP or ns-process-data is available.
ns-process-data images \\
  --data {output_dir}/frames \\
  --output-dir {output_dir}/nerfstudio

ns-train nerfacto \\
  --data {output_dir}/nerfstudio

# 3D Gaussian Splatting-style training placeholder.
# Replace TRAIN_3DGS with your local 3DGS training entry point.
TRAIN_3DGS \\
  --source_path {output_dir}/frames \\
  --model_path {output_dir}/gaussian_splatting
"""
    (output_dir / "nerf_3dgs_templates.sh").write_text(text, encoding="utf-8")


def write_failure_analysis(output_dir: Path, frames: list[dict], deps: dict, slam: dict) -> None:
    text = f"""# Failure Case Analysis

This generated note records the expected reconstruction risks for the Xperience-10M
sample episode.

## Smoke-Test Context

- Extracted frames: {len(frames)}
- Exported SLAM poses: {slam['num_exported_poses']}
- SLAM point cloud points: {slam['num_point_cloud_points']}
- COLMAP installed: {deps['colmap']['available']}
- NeRFStudio installed: {deps['ns-train']['available']}

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
"""
    (output_dir / "failure_analysis.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root_default = Path(__file__).resolve().parents[2] / "data/sample/xperience-10m-sample"
    parser = argparse.ArgumentParser(description="Prepare an egocentric 3D reconstruction demo from the Xperience sample.")
    parser.add_argument("--data-root", type=Path, default=root_default)
    parser.add_argument("--video", default="fisheye_cam0.mp4")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sample_demo"))
    parser.add_argument("--frame-stride", type=int, default=180)
    parser.add_argument("--max-frames", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotation = args.data_root / "annotation.hdf5"
    deps = check_dependencies()
    frames = extract_frames(args.data_root / args.video, args.output_dir, args.frame_stride, args.max_frames)
    export_calibration(annotation, args.output_dir)
    slam = export_slam(annotation, args.output_dir)
    write_colmap_commands(args.output_dir)
    write_nerf_3dgs_templates(args.output_dir)
    write_failure_analysis(args.output_dir, frames, deps, slam)
    (args.output_dir / "dependency_check.json").write_text(json.dumps(deps, indent=2), encoding="utf-8")
    print(f"extracted_frames={len(frames)} colmap_available={deps['colmap']['available']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
