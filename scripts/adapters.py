from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class VideoFrameSource(Protocol):
    """Provides the egocentric video that will be sampled into frames."""

    video_path: Path


class CalibrationSource(Protocol):
    """Provides camera intrinsics, distortion, and rig metadata."""

    annotation_path: Path


class PoseSource(Protocol):
    """Provides timestamped camera poses or pose-like SLAM records."""

    annotation_path: Path


@dataclass(frozen=True)
class XperienceReconstructionAdapter:
    """Boundary object for the Xperience-10M sample reconstruction layout."""

    data_root: Path
    video_name: str = "fisheye_cam0.mp4"

    @property
    def video_path(self) -> Path:
        return self.data_root / self.video_name

    @property
    def annotation_path(self) -> Path:
        return self.data_root / "annotation.hdf5"

    def describe(self) -> dict:
        return {
            "adapter": "XperienceReconstructionAdapter",
            "video_path": str(self.video_path),
            "annotation_path": str(self.annotation_path),
            "signals": ["video_frames", "camera_calibration", "slam_poses", "slam_point_cloud"],
        }
