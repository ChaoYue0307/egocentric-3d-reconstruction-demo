from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from adapters import XperienceReconstructionAdapter  # noqa: E402
from reconstruction_demo import (  # noqa: E402
    as_jsonable,
    build_frames_manifest,
    colmap_camera_center,
    compare_colmap_to_slam,
    draw_hand_mask,
    export_slam,
    frame_quality_report,
    hand_chain_candidates,
    parse_colmap_model,
    read_slam_records,
    trajectory_metrics,
    umeyama_alignment,
    write_colmap_summary,
)


def make_annotation(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        slam = h5.create_group("slam")
        slam.create_dataset("frame_names", data=np.asarray([b"100.jpg", b"200.jpg", b"300.jpg"]))
        slam.create_dataset("trans_xyz", data=np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32))
        slam.create_dataset("quat_wxyz", data=np.asarray([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32))
        slam.create_dataset("point_cloud", data=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.float32))


def test_as_jsonable_decodes_bytes() -> None:
    assert as_jsonable(np.asarray(b"pinhole")) == "pinhole"
    assert as_jsonable(np.asarray([1, 2])).copy() == [1, 2]


def test_read_slam_records(tmp_path: Path) -> None:
    annotation = tmp_path / "annotation.hdf5"
    make_annotation(annotation)
    records = read_slam_records(annotation)
    assert records[1]["timestamp"] == "200"
    assert records[1]["position_xyz"] == [1.0, 0.0, 0.0]


def test_export_slam_writes_summary(tmp_path: Path) -> None:
    annotation = tmp_path / "annotation.hdf5"
    make_annotation(annotation)
    summary = export_slam(annotation, tmp_path, max_rows=2)
    assert summary["num_exported_poses"] == 2
    assert (tmp_path / "slam_poses_tum.txt").exists()


def test_build_frames_manifest_matches_nearest_pose() -> None:
    frames = [{"image": "frames/frame_00000.jpg", "source_frame": 1}]
    calibration = {"cam0": {"K": [1, 2, 3, 4], "D": [0, 0, 0, 0]}}
    slam = [
        {"index": 0, "timestamp": "100", "position_xyz": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
        {"index": 2, "timestamp": "300", "position_xyz": [2, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
    ]
    manifest = build_frames_manifest(frames, calibration, slam, "cam0")
    assert manifest[0]["camera_intrinsics"] == [1, 2, 3, 4]
    assert manifest[0]["nearest_slam_pose"]["timestamp"] == "100"


def test_parse_colmap_model_text_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "sparse" / "0"
    model_dir.mkdir(parents=True)
    (model_dir / "cameras.txt").write_text("1 PINHOLE 640 480 500 500 320 240\n", encoding="utf-8")
    (model_dir / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 frame_00000.jpg\n"
        "10 20 1 30 40 2\n",
        encoding="utf-8",
    )
    (model_dir / "points3D.txt").write_text(
        "1 0.0 1.0 2.0 255 255 255 0.5 1 10\n"
        "2 1.0 2.0 3.0 255 255 255 0.7 1 20\n",
        encoding="utf-8",
    )
    summary = parse_colmap_model(model_dir)
    assert summary["num_cameras"] == 1
    assert summary["num_registered_images"] == 1
    assert summary["num_sparse_points"] == 2
    assert summary["mean_reprojection_error"] == 0.6
    assert summary["image_poses"][0]["name"] == "frame_00000.jpg"
    write_colmap_summary(tmp_path, summary)
    assert (tmp_path / "colmap_summary.json").exists()
    assert (tmp_path / "colmap_summary.svg").exists()


def test_compare_colmap_to_slam_uses_camera_centers(tmp_path: Path) -> None:
    manifest = [{
        "source_frame": 42,
        "nearest_slam_pose": {"timestamp": "100", "position_xyz": [1.0, 2.0, 3.0]},
    }]
    manifest_path = tmp_path / "frames_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    comparison = compare_colmap_to_slam(
        {"image_poses": [{"name": "frame_00000_src_000042.jpg", "qvec_wxyz": [1.0, 0.0, 0.0, 0.0], "tvec_xyz": [-1.0, -2.0, -3.0]}]},
        manifest_path,
    )
    assert comparison["num_matched_frames"] == 1
    assert comparison["mean_translation_delta"] == 0.0


def test_colmap_camera_center_identity_rotation() -> None:
    center = colmap_camera_center([1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    assert center == [-1.0, -2.0, -3.0]


def test_umeyama_recovers_similarity_transform() -> None:
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    angle = 0.7
    R_true = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    dst = (2.0 * (R_true @ src.T)).T + np.array([1.0, -2.0, 0.5])
    scale, R, t = umeyama_alignment(src, dst)
    assert abs(scale - 2.0) < 1e-9
    assert np.allclose(R, R_true)
    metrics = trajectory_metrics(src, dst)
    assert metrics["ate_rmse"] < 1e-9
    assert metrics["rpe_rmse"] < 1e-9
    assert abs(metrics["scale"] - 2.0) < 1e-9


def test_hand_chain_candidates_has_four_compositions() -> None:
    candidates = hand_chain_candidates(np.eye(4), np.eye(4))
    assert len(candidates) == 4
    for T in candidates.values():
        assert np.allclose(T, np.eye(4))


def test_draw_hand_mask_blocks_joint_region() -> None:
    mask = draw_hand_mask((200, 300), [(150.0, 100.0)], [0.5], focal_px=400.0, dilation_px=5)
    assert mask.shape == (200, 300)
    assert mask[100, 150] == 0
    assert mask[5, 5] == 255
    assert (mask == 0).mean() > 0.01


def test_frame_quality_report_flags_blur(tmp_path: Path) -> None:
    import cv2

    rng = np.random.default_rng(0)
    sharp = (rng.integers(0, 256, size=(200, 200, 3))).astype(np.uint8)
    blurry = cv2.GaussianBlur(sharp, (31, 31), 12)
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    cv2.imwrite(str(frame_dir / "a.jpg"), sharp)
    cv2.imwrite(str(frame_dir / "b.jpg"), blurry)
    frames = [
        {"image": "frames/a.jpg", "source_frame": 0},
        {"image": "frames/b.jpg", "source_frame": 180},
    ]
    summary = frame_quality_report(tmp_path, frames)
    assert summary["num_frames"] == 2
    assert summary["frames"][0]["blurry"] is False
    assert summary["frames"][1]["blurry"] is True
    assert (tmp_path / "frame_quality.json").exists()


def test_xperience_reconstruction_adapter_paths(tmp_path: Path) -> None:
    adapter = XperienceReconstructionAdapter(tmp_path)
    assert adapter.annotation_path == tmp_path / "annotation.hdf5"
    assert adapter.video_path == tmp_path / "fisheye_cam0.mp4"
    assert "slam_poses" in adapter.describe()["signals"]
