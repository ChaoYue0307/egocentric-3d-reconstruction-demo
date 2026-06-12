#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
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
    names = ["colmap", "ns-process-data", "ns-train", "instant-ngp", "python3"]
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


def frame_quality_report(output_dir: Path, frames: list[dict], blur_threshold: float = 60.0, overlap_threshold: int = 80) -> dict:
    """Blur, exposure, and consecutive-frame ORB-overlap diagnostics.

    Metrics are computed on a central crop so the black fisheye border does not
    dominate exposure statistics.
    """
    import cv2

    rows = []
    orb = cv2.ORB_create(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_des = None
    for frame in frames:
        img = cv2.imread(str(output_dir / frame["image"]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        crop = gray[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
        blur = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        keypoints, des = orb.detectAndCompute(gray, None)
        matches = None
        if prev_des is not None and des is not None and len(prev_des) and len(des):
            matches = len(matcher.match(prev_des, des))
        rows.append({
            "image": frame["image"],
            "source_frame": frame["source_frame"],
            "blur_laplacian_var": round(blur, 2),
            "blurry": blur < blur_threshold,
            "mean_intensity": round(float(crop.mean()), 2),
            "underexposed_fraction": round(float((crop < 10).mean()), 4),
            "overexposed_fraction": round(float((crop > 245).mean()), 4),
            "num_keypoints": len(keypoints or []),
            "matches_to_previous": matches,
            "low_overlap": matches is not None and matches < overlap_threshold,
        })
        prev_des = des
    match_values = [row["matches_to_previous"] for row in rows if row["matches_to_previous"] is not None]
    summary = {
        "num_frames": len(rows),
        "blur_threshold": blur_threshold,
        "overlap_threshold": overlap_threshold,
        "num_blurry": sum(row["blurry"] for row in rows),
        "num_low_overlap": sum(row["low_overlap"] for row in rows),
        "mean_matches_to_previous": round(sum(match_values) / len(match_values), 1) if match_values else None,
        "weakest_pairs": sorted(
            [{"image": row["image"], "matches_to_previous": row["matches_to_previous"]} for row in rows if row["matches_to_previous"] is not None],
            key=lambda row: row["matches_to_previous"],
        )[:5],
        "frames": rows,
    }
    (output_dir / "frame_quality.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


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


def quat_wxyz_to_rotation(q) -> object:
    import numpy as np

    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def camera_from_body_chain(annotation: Path, camera_name: str):
    """Resolve the rig extrinsic chain: cam0 stores T_c_b, cam1..3 chain via T_cn_cnm1."""
    import numpy as np

    with h5py_file(annotation) as h5:
        T = np.asarray(h5["calibration/cam0/T_c_b"][...], dtype=float)
        if camera_name == "cam0":
            return T
        for cam in ("cam1", "cam2", "cam3"):
            T = np.asarray(h5[f"calibration/{cam}/T_cn_cnm1"][...], dtype=float) @ T
            if cam == camera_name:
                return T
    raise ValueError(f"Unsupported camera for hand masks: {camera_name}")


def h5py_file(path: Path):
    import h5py

    return h5py.File(path, "r")


def hand_chain_candidates(T_cam_body, T_world_body):
    """The four plausible compositions of rig extrinsics and SLAM pose.

    Conventions in shipped rigs are inconsistent, so the projection chain is
    selected empirically per episode instead of trusted from field names.
    """
    import numpy as np

    inv = np.linalg.inv
    return {
        "cam_body @ inv(world_body)": T_cam_body @ inv(T_world_body),
        "inv(cam_body) @ inv(world_body)": inv(T_cam_body) @ inv(T_world_body),
        "cam_body @ world_body": T_cam_body @ T_world_body,
        "inv(cam_body) @ world_body": inv(T_cam_body) @ T_world_body,
    }


def project_fisheye(points_cam, K4, D, min_depth: float = 0.02):
    """Project camera-frame points through an equidistant fisheye model."""
    import cv2
    import numpy as np

    points_cam = np.asarray(points_cam, dtype=float)
    front = points_cam[points_cam[:, 2] > min_depth]
    if not len(front):
        return np.zeros((0, 2)), np.zeros((0,))
    K = np.array([[K4[0], 0.0, K4[2]], [0.0, K4[1], K4[3]], [0.0, 0.0, 1.0]])
    uv, _ = cv2.fisheye.projectPoints(front.reshape(1, -1, 3), np.zeros(3), np.zeros(3), K, np.asarray(D, dtype=float).reshape(4, 1))
    return uv[0], front[:, 2]


def select_hand_projection_chain(annotation: Path, camera_name: str, image_size, sample_stride: int = 240) -> dict:
    """Score each candidate chain by in-front and in-bounds fractions and pick the best."""
    import numpy as np

    T_cam_body = camera_from_body_chain(annotation, camera_name)
    with h5py_file(annotation) as h5:
        K4 = np.asarray(h5[f"calibration/{camera_name}/K"][...], dtype=float)
        D = np.asarray(h5[f"calibration/{camera_name}/D"][...], dtype=float)
        left = np.asarray(h5["hand_mocap/left_joints_3d"][...], dtype=float)
        right = np.asarray(h5["hand_mocap/right_joints_3d"][...], dtype=float)
        trans = np.asarray(h5["slam/trans_xyz"][...], dtype=float)
        quat = np.asarray(h5["slam/quat_wxyz"][...], dtype=float)
    width, height = image_size
    scores = {}
    for label in hand_chain_candidates(np.eye(4), np.eye(4)):
        total, in_front, in_bounds, depths = 0, 0, 0, []
        for idx in range(0, len(left), sample_stride):
            joints = np.concatenate([left[idx], right[idx]])
            if not np.isfinite(joints).all():
                continue
            T_world_body = np.eye(4)
            T_world_body[:3, :3] = quat_wxyz_to_rotation(quat[idx])
            T_world_body[:3, 3] = trans[idx]
            T = hand_chain_candidates(T_cam_body, T_world_body)[label]
            points_cam = (T @ np.c_[joints, np.ones(len(joints))].T)[:3].T
            uv, z = project_fisheye(points_cam, K4, D)
            total += len(joints)
            in_front += len(z)
            depths.extend(z.tolist())
            if len(uv):
                ok = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
                in_bounds += int(ok.sum())
        median_depth = float(np.median(depths)) if depths else None
        # Hands on a tabletop sit roughly an arm's length from a head rig; a
        # median outside this band signals a wrong frame composition even when
        # the wide fisheye keeps the projection in bounds.
        plausible_depth = median_depth is not None and 0.2 <= median_depth <= 1.5
        scores[label] = {
            "in_front_fraction": round(in_front / total, 4) if total else 0.0,
            "in_bounds_fraction": round(in_bounds / total, 4) if total else 0.0,
            "median_depth_m": round(median_depth, 4) if median_depth is not None else None,
            "score": (in_bounds / total if total else 0.0) + (1.0 if plausible_depth else 0.0),
        }
    best = max(scores, key=lambda label: scores[label]["score"])
    return {"camera": camera_name, "selected_chain": best, "candidates": scores}


def draw_hand_mask(image_shape, uv, depths, focal_px: float, hand_radius_m: float = 0.10, dilation_px: int = 25):
    """COLMAP-convention mask: white = usable pixels, black = masked hand region."""
    import cv2
    import numpy as np

    height, width = image_shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    for (u, v), z in zip(uv, depths, strict=True):
        radius = int(np.clip(hand_radius_m * focal_px / max(float(z), 0.05), 12, 220))
        cv2.circle(mask, (int(round(u)), int(round(v))), radius, 0, -1)
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1))
        mask = cv2.erode(mask, kernel)
    return mask


def write_hand_masks(annotation: Path, output_dir: Path, frames: list[dict], camera_name: str, dilation_px: int = 25) -> dict:
    """Project MANO hand joints into each extracted frame and write COLMAP masks.

    The mocap-to-camera registration carries a visible systematic offset of tens
    of pixels on this rig, so masks are generously dilated and an overlay
    preview is always written for human verification.
    """
    import cv2
    import numpy as np

    first = cv2.imread(str(output_dir / frames[0]["image"]))
    height, width = first.shape[:2]
    selection = select_hand_projection_chain(annotation, camera_name, (width, height))
    T_cam_body = camera_from_body_chain(annotation, camera_name)
    with h5py_file(annotation) as h5:
        K4 = np.asarray(h5[f"calibration/{camera_name}/K"][...], dtype=float)
        D = np.asarray(h5[f"calibration/{camera_name}/D"][...], dtype=float)
        left = np.asarray(h5["hand_mocap/left_joints_3d"][...], dtype=float)
        right = np.asarray(h5["hand_mocap/right_joints_3d"][...], dtype=float)
        trans = np.asarray(h5["slam/trans_xyz"][...], dtype=float)
        quat = np.asarray(h5["slam/quat_wxyz"][...], dtype=float)
    mask_dir = output_dir / "hand_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    previews = 0
    for frame in frames:
        idx = min(int(frame["source_frame"]), len(left) - 1)
        joints = np.concatenate([left[idx], right[idx]])
        masked_fraction = 0.0
        mask = np.full((height, width), 255, dtype=np.uint8)
        if np.isfinite(joints).all():
            T_world_body = np.eye(4)
            T_world_body[:3, :3] = quat_wxyz_to_rotation(quat[idx])
            T_world_body[:3, 3] = trans[idx]
            T = hand_chain_candidates(T_cam_body, T_world_body)[selection["selected_chain"]]
            points_cam = (T @ np.c_[joints, np.ones(len(joints))].T)[:3].T
            uv, z = project_fisheye(points_cam, K4, D)
            if len(uv):
                mask = draw_hand_mask((height, width), uv, z, float(K4[0]), dilation_px=dilation_px)
                masked_fraction = float((mask == 0).mean())
        mask_name = Path(frame["image"]).name + ".png"
        cv2.imwrite(str(mask_dir / mask_name), mask)
        if previews < 2 and masked_fraction > 0.0:
            image = cv2.imread(str(output_dir / frame["image"]))
            overlay = image.copy()
            overlay[mask == 0] = (0, 0, 255)
            preview = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
            cv2.imwrite(str(output_dir / f"hand_mask_preview_{previews}.jpg"), preview)
            previews += 1
        rows.append({"image": frame["image"], "mask": f"hand_masks/{mask_name}", "masked_fraction": round(masked_fraction, 4)})
    report = {
        "camera": camera_name,
        "convention_selection": selection,
        "dilation_px": dilation_px,
        "mask_convention": "COLMAP: white pixels are used for feature extraction, black pixels are ignored.",
        "registration_caveat": "Mocap-to-camera registration shows a systematic offset of tens of pixels on this rig; masks are dilated to compensate. Always inspect hand_mask_preview_*.jpg.",
        "num_masks": len(rows),
        "mean_masked_fraction": round(sum(row["masked_fraction"] for row in rows) / len(rows), 4) if rows else 0.0,
        "masks": rows,
    }
    (output_dir / "hand_mask_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


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
        "image_poses": images[:100],
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


def colmap_camera_center(qvec_wxyz, tvec_xyz):
    """COLMAP stores world-to-camera poses; the camera center is C = -R^T t."""
    import numpy as np

    R = quat_wxyz_to_rotation(qvec_wxyz)
    return (-R.T @ np.asarray(tvec_xyz, dtype=float)).tolist()


def umeyama_alignment(src, dst):
    """Similarity transform (s, R, t) minimizing ||dst - (s R src + t)||^2 (Umeyama 1991)."""
    import numpy as np

    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = dst_c.T @ src_c / len(src)
    U, S, Vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(U @ Vt)) or 1.0
    D = np.diag([1.0, 1.0, sign])
    R = U @ D @ Vt
    var_src = float((src_c ** 2).sum() / len(src))
    scale = float(np.trace(np.diag(S) @ D) / var_src) if var_src > 0 else 1.0
    t = mu_dst - scale * R @ mu_src
    return scale, R, t


def trajectory_metrics(estimated, reference) -> dict:
    """ATE and RPE after Sim(3) alignment of an estimated trajectory to a reference."""
    import numpy as np

    estimated = np.asarray(estimated, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(estimated) < 3:
        return {"num_poses": int(len(estimated)), "ate_rmse": None, "rpe_rmse": None, "scale": None}
    scale, R, t = umeyama_alignment(estimated, reference)
    aligned = (scale * (R @ estimated.T)).T + t
    errors = np.linalg.norm(aligned - reference, axis=1)
    rel_est = np.diff(aligned, axis=0)
    rel_ref = np.diff(reference, axis=0)
    rpe = np.linalg.norm(rel_est - rel_ref, axis=1)
    return {
        "num_poses": int(len(estimated)),
        "scale": round(scale, 6),
        "ate_rmse": round(float(np.sqrt((errors ** 2).mean())), 6),
        "ate_max": round(float(errors.max()), 6),
        "rpe_rmse": round(float(np.sqrt((rpe ** 2).mean())), 6),
        "per_pose_error": [round(float(x), 6) for x in errors],
    }


def compare_colmap_to_slam(colmap_summary: dict, frames_manifest_path: Path) -> dict:
    if not frames_manifest_path.exists():
        return {"num_matched_frames": 0, "mean_translation_delta": None, "rows": []}
    manifest = json.loads(frames_manifest_path.read_text(encoding="utf-8"))
    by_source = {int(row["source_frame"]): row for row in manifest}
    matched = []
    for image in colmap_summary.get("image_poses", []):
        match = re.search(r"_src_(\d+)", image["name"])
        if not match:
            continue
        source_frame = int(match.group(1))
        frame = by_source.get(source_frame)
        slam_pose = frame.get("nearest_slam_pose") if frame else None
        if not slam_pose:
            continue
        matched.append({
            "image": image["name"],
            "source_frame": source_frame,
            "slam_timestamp": slam_pose["timestamp"],
            "colmap_center": colmap_camera_center(image["qvec_wxyz"], image["tvec_xyz"]),
            "slam_position": [float(x) for x in slam_pose["position_xyz"]],
        })
    if not matched:
        return {"num_matched_frames": 0, "mean_translation_delta": None, "rows": []}
    metrics = trajectory_metrics([row["colmap_center"] for row in matched], [row["slam_position"] for row in matched])
    per_pose = metrics.pop("per_pose_error", None)
    rows = []
    for i, row in enumerate(matched):
        delta = per_pose[i] if per_pose else math.dist(row["colmap_center"], row["slam_position"])
        rows.append({
            "image": row["image"],
            "source_frame": row["source_frame"],
            "slam_timestamp": row["slam_timestamp"],
            "translation_delta": round(delta, 6),
        })
    mean_delta = round(sum(row["translation_delta"] for row in rows) / len(rows), 6)
    return {
        "num_matched_frames": len(rows),
        "alignment": "Sim(3) Umeyama on COLMAP camera centers (C = -R^T t) versus SLAM positions" if len(rows) >= 3 else "unaligned (fewer than 3 matched poses)",
        "metrics": metrics,
        "mean_translation_delta": mean_delta,
        "rows": rows,
    }


def write_pose_comparison(output_dir: Path, comparison: dict) -> None:
    (output_dir / "colmap_vs_slam.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    rows = comparison.get("rows", [])[:8]
    max_delta = max([row["translation_delta"] for row in rows] + [1.0])
    bars = []
    y = 86
    for row in rows:
        width = int(420 * row["translation_delta"] / max_delta)
        bars.append(f'<text x="40" y="{y + 16}" fill="#aab6ca" font-family="Inter,Arial" font-size="13">{row["source_frame"]}</text>')
        bars.append(f'<rect x="118" y="{y}" width="{max(width, 4)}" height="22" rx="8" fill="#f59e0b"/>')
        bars.append(f'<text x="{132 + max(width, 4)}" y="{y + 16}" fill="#edf2fb" font-family="Inter,Arial" font-size="13">{row["translation_delta"]}</text>')
        y += 34
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 360" role="img" aria-label="COLMAP versus SLAM pose comparison">
  <rect width="720" height="360" rx="24" fill="#0b1220"/>
  <text x="32" y="44" fill="#edf2fb" font-family="Inter,Arial" font-size="24" font-weight="800">COLMAP vs SLAM Pose Check</text>
  <text x="32" y="70" fill="#aab6ca" font-family="Inter,Arial" font-size="14">{comparison["num_matched_frames"]} matched frames · mean translation delta: {comparison["mean_translation_delta"]}</text>
  {''.join(bars)}
</svg>
"""
    (output_dir / "colmap_vs_slam.svg").write_text(svg, encoding="utf-8")


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
    parser.add_argument("--skip-quality", action="store_true", help="Skip blur, exposure, and ORB-overlap frame diagnostics.")
    parser.add_argument("--write-hand-masks", action="store_true", help="Project MANO hand joints through rig calibration and SLAM poses into COLMAP-convention masks.")
    parser.add_argument("--mask-dilation-px", type=int, default=25, help="Extra dilation applied to hand masks to absorb mocap-to-camera registration offset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.parse_colmap_only:
        if not args.colmap_model:
            raise SystemExit("--parse-colmap-only requires --colmap-model")
        summary = parse_colmap_model(args.colmap_model)
        write_colmap_summary(args.output_dir, summary)
        comparison = compare_colmap_to_slam(summary, args.output_dir / "frames_manifest.json")
        write_pose_comparison(args.output_dir, comparison)
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
    quality = None if args.skip_quality else frame_quality_report(args.output_dir, frames)
    hand_masks = write_hand_masks(annotation, args.output_dir, frames, args.camera_name, args.mask_dilation_px) if args.write_hand_masks else None
    write_colmap_commands(args.output_dir)
    write_nerf_3dgs_templates(args.output_dir)
    write_failure_analysis(args.output_dir, frames, deps, slam)
    colmap_summary = parse_colmap_model(args.colmap_model) if args.colmap_model else None
    if colmap_summary:
        write_colmap_summary(args.output_dir, colmap_summary)
        write_pose_comparison(args.output_dir, compare_colmap_to_slam(colmap_summary, args.output_dir / "frames_manifest.json"))
    (args.output_dir / "dependency_check.json").write_text(json.dumps(deps, indent=2), encoding="utf-8")
    (args.output_dir / "run_summary.json").write_text(json.dumps({
        "video": args.video,
        "camera_name": args.camera_name,
        "num_extracted_frames": len(frames),
        "num_manifest_rows": len(manifest),
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "colmap_summary": "colmap_summary.json" if colmap_summary else None,
        "frame_quality": {
            "num_blurry": quality["num_blurry"],
            "num_low_overlap": quality["num_low_overlap"],
            "mean_matches_to_previous": quality["mean_matches_to_previous"],
        } if quality else None,
        "hand_masks": {
            "selected_chain": hand_masks["convention_selection"]["selected_chain"],
            "mean_masked_fraction": hand_masks["mean_masked_fraction"],
        } if hand_masks else None,
    }, indent=2), encoding="utf-8")
    quality_note = f" blurry={quality['num_blurry']} low_overlap={quality['num_low_overlap']}" if quality else ""
    print(f"extracted_frames={len(frames)} colmap_available={deps['colmap']['available']}{quality_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
