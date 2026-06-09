from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from reconstruction_demo import as_jsonable, build_frames_manifest, export_slam, read_slam_records  # noqa: E402


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
