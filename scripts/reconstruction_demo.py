#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def as_jsonable(value):
    import numpy as np

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
    import cv2

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


def read_slam_records(annotation: Path, max_rows: int | None = None) -> list[dict]:
    import h5py
    import numpy as np

    with h5py.File(annotation, "r") as h5:
        stop = None if max_rows is None or max_rows == 0 else max_rows
        if stop is None:
            raw_names = h5["slam/frame_names"][...]
            trans = np.asarray(h5["slam/trans_xyz"][...], dtype=float)
            quat = np.asarray(h5["slam/quat_wxyz"][...], dtype=float)
        else:
            raw_names = h5["slam/frame_names"][:stop]
            trans = np.asarray(h5["slam/trans_xyz"][:stop], dtype=float)
            quat = np.asarray(h5["slam/quat_wxyz"][:stop], dtype=float)
        names = [np.asarray(x).tobytes().decode("utf-8", errors="replace").strip("\x00") for x in raw_names]
    records = []
    for idx, (name, t, q) in enumerate(zip(names, trans, quat, strict=True)):
        timestamp = name.rsplit(".", 1)[0]
        records.append({
            "index": idx,
            "timestamp": timestamp,
            "position_xyz": [round(float(x), 8) for x in t],
            "quaternion_wxyz": [round(float(x), 8) for x in q],
        })
    return records


def export_calibration(annotation: Path, output_dir: Path) -> dict:
    import h5py

    payload = {}
    with h5py.File(annotation, "r") as h5:
        cal = h5["calibration"]
        for cam_name in sorted(cal.keys()):
            cam = cal[cam_name]
            payload[cam_name] = {key: as_jsonable(cam[key][()]) for key in cam.keys()}
    (output_dir / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def export_slam(annotation: Path, output_dir: Path, max_rows: int = 200) -> dict:
    import h5py
    import numpy as np

    records = read_slam_records(annotation, max_rows)
    with h5py.File(annotation, "r") as h5:
        point_cloud = np.asarray(h5["slam/point_cloud"], dtype=float)
    pose_path = output_dir / "slam_poses_tum.txt"
    with pose_path.open("w", encoding="utf-8") as fp:
        fp.write("# timestamp tx ty tz qx qy qz qw\n")
        for record in records:
            t = record["position_xyz"]
            q = record["quaternion_wxyz"]
            fp.write(f"{record['timestamp']} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} {q[0]:.8f}\n")
    ply_path = output_dir / "slam_point_cloud_preview.ply"
    preview = point_cloud[: min(len(point_cloud), 4000)]
    with ply_path.open("w", encoding="utf-8") as fp:
        fp.write("ply\nformat ascii 1.0\n")
        fp.write(f"element vertex {len(preview)}\n")
        fp.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in preview:
            fp.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
    summary = {
        "num_exported_poses": len(records),
        "num_point_cloud_points": int(len(point_cloud)),
        "pose_file": str(pose_path.relative_to(output_dir)),
        "point_cloud_preview": str(ply_path.relative_to(output_dir)),
    }
    (output_dir / "slam_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_frames_manifest(frames: list[dict], calibration: dict, slam_records: list[dict], camera_name: str) -> list[dict]:
    camera = calibration.get(camera_name, {})
    manifest = []
    for frame in frames:
        source_frame = int(frame["source_frame"])
        nearest = min(slam_records, key=lambda row: abs(int(row["index"]) - source_frame)) if slam_records else None
        manifest.append({
            **frame,
            "camera": camera_name,
            "camera_intrinsics": camera.get("K"),
            "camera_distortion": camera.get("D"),
            "nearest_slam_pose": nearest,
            "pose_source": "slam/frame_names + slam/trans_xyz + slam/quat_wxyz",
        })
    return manifest


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


def _data_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def parse_colmap_model(model_dir: Path) -> dict:
    cameras = []
    for line in _data_lines(model_dir / "cameras.txt"):
        parts = line.split()
        if len(parts) >= 5:
            cameras.append({
                "camera_id": int(parts[0]),
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(x) for x in parts[4:]],
            })

    images = []
    image_lines = _data_lines(model_dir / "images.txt")
    for idx in range(0, len(image_lines), 2):
        parts = image_lines[idx].split()
        if len(parts) >= 10:
            images.append({
                "image_id": int(parts[0]),
                "qvec_wxyz": [float(x) for x in parts[1:5]],
                "tvec_xyz": [float(x) for x in parts[5:8]],
                "camera_id": int(parts[8]),
                "name": parts[9],
            })

    points = []
    errors = []
    for line in _data_lines(model_dir / "points3D.txt"):
        parts = line.split()
        if len(parts) >= 8:
            xyz = [float(x) for x in parts[1:4]]
            points.append(xyz)
            errors.append(float(parts[7]))

    bbox = {
        "min_xyz": [round(min(point[i] for point in points), 6) for i in range(3)] if points else None,
        "max_xyz": [round(max(point[i] for point in points), 6) for i in range(3)] if points else None,
    }
    return {
        "model_dir": str(model_dir),
        "num_cameras": len(cameras),
        "num_registered_images": len(images),
        "num_sparse_points": len(points),
        "camera_models": sorted({row["model"] for row in cameras}),
        "registered_images": [row["name"] for row in images[:20]],
        "mean_reprojection_error": round(sum(errors) / len(errors), 6) if errors else None,
        "point_bbox": bbox,
    }


def write_colmap_summary(output_dir: Path, summary: dict) -> None:
    (output_dir / "colmap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    width, height = 720, 320
    bars = [
        ("cameras", summary["num_cameras"], "#38bdf8"),
        ("images", summary["num_registered_images"], "#a78bfa"),
        ("points", summary["num_sparse_points"], "#34d399"),
    ]
    max_value = max([value for _, value, _ in bars] + [1])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="COLMAP sparse model summary">',
        '<rect width="720" height="320" rx="24" fill="#0b1220"/>',
        '<text x="32" y="48" fill="#edf2fb" font-family="Inter,Arial,sans-serif" font-size="24" font-weight="800">COLMAP Sparse Model Summary</text>',
    ]
    for i, (label, value, color) in enumerate(bars):
        y = 92 + i * 64
        bar_width = int(520 * value / max_value)
        parts.append(f'<text x="32" y="{y + 22}" fill="#aab6ca" font-family="Inter,Arial,sans-serif" font-size="16">{label}</text>')
        parts.append(f'<rect x="150" y="{y}" width="{max(bar_width, 4)}" height="32" rx="10" fill="{color}"/>')
        parts.append(f'<text x="{164 + max(bar_width, 4)}" y="{y + 22}" fill="#edf2fb" font-family="Inter,Arial,sans-serif" font-size="16">{value}</text>')
    error = summary["mean_reprojection_error"]
    parts.append(f'<text x="32" y="292" fill="#aab6ca" font-family="Inter,Arial,sans-serif" font-size="15">mean reprojection error: {error if error is not None else "n/a"}</text>')
    parts.append("</svg>")
    (output_dir / "colmap_summary.svg").write_text("\n".join(parts), encoding="utf-8")


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
    parser.add_argument("--camera-name", default="cam0")
    parser.add_argument("--max-slam-poses", type=int, default=200)
    parser.add_argument("--colmap-model", type=Path, help="Optional COLMAP sparse text model directory containing cameras.txt, images.txt, and points3D.txt.")
    parser.add_argument("--parse-colmap-only", action="store_true", help="Only parse --colmap-model and write colmap_summary artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.parse_colmap_only:
        if not args.colmap_model:
            raise SystemExit("--parse-colmap-only requires --colmap-model")
        summary = parse_colmap_model(args.colmap_model)
        write_colmap_summary(args.output_dir, summary)
        print(f"colmap_images={summary['num_registered_images']} sparse_points={summary['num_sparse_points']}")
        return 0
    annotation = args.data_root / "annotation.hdf5"
    deps = check_dependencies()
    frames = extract_frames(args.data_root / args.video, args.output_dir, args.frame_stride, args.max_frames)
    calibration = export_calibration(annotation, args.output_dir)
    slam_records = read_slam_records(annotation, None)
    manifest = build_frames_manifest(frames, calibration, slam_records, args.camera_name)
    (args.output_dir / "frames_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    slam = export_slam(annotation, args.output_dir, args.max_slam_poses)
    write_colmap_commands(args.output_dir)
    write_nerf_3dgs_templates(args.output_dir)
    write_failure_analysis(args.output_dir, frames, deps, slam)
    colmap_summary = parse_colmap_model(args.colmap_model) if args.colmap_model else None
    if colmap_summary:
        write_colmap_summary(args.output_dir, colmap_summary)
    (args.output_dir / "dependency_check.json").write_text(json.dumps(deps, indent=2), encoding="utf-8")
    (args.output_dir / "run_summary.json").write_text(json.dumps({
        "video": args.video,
        "camera_name": args.camera_name,
        "num_extracted_frames": len(frames),
        "num_manifest_rows": len(manifest),
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "colmap_summary": "colmap_summary.json" if colmap_summary else None,
    }, indent=2), encoding="utf-8")
    print(f"extracted_frames={len(frames)} colmap_available={deps['colmap']['available']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
