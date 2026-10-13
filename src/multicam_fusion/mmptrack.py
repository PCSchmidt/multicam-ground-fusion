"""MMPTrack loader: calibration -> ground-plane homographies, per-frame labels and top-down ground truth.

Conventions were inferred by fitting the supplied calibration to the ground-truth boxes (docs/PROTOCOL_SECOND_SCENE.md):
the calibration `Rotation` is camera-to-world (row-major 3x3) and `Translation` is the camera position in the world,
both in millimetres, so a world point X maps to the camera as R^T (X - T); the ground plane is world z = 0; the
top-down CSV columns are (person id, V index, U index, height), with world U or V = Min + index * VoxelSizeInMM.
Everything returned here is in metres.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import Array


@dataclass(frozen=True)
class MmpCamera:
    camera_id: int
    k: Array
    rot: Array  # camera-to-world rotation
    pos_mm: Array  # camera position, world frame

    def world_to_pixel_h(self) -> Array:
        """Homography from ground-plane (U, V, 1) in metres to pixels."""
        rt = self.rot.T
        cols = np.column_stack([1000.0 * rt[:, 0], 1000.0 * rt[:, 1], -rt @ self.pos_mm])
        return np.asarray(self.k @ cols, dtype=np.float64)

    def pixel_to_world_h(self) -> Array:
        return np.asarray(np.linalg.inv(self.world_to_pixel_h()), dtype=np.float64)


@dataclass(frozen=True)
class MmpSpace:
    min_u_m: float
    max_u_m: float
    min_v_m: float
    max_v_m: float
    voxel_m: float

    def contains(self, xy: Array) -> bool:
        return bool(self.min_u_m <= xy[0] <= self.max_u_m and self.min_v_m <= xy[1] <= self.max_v_m)


def load_calibration(path: Path) -> tuple[dict[int, MmpCamera], MmpSpace]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cams: dict[int, MmpCamera] = {}
    for c in raw["Cameras"]:
        e, i = c["ExtrinsicParameters"], c["IntrinsicParameters"]
        k = np.array([[i["Fx"], 0.0, i["Cx"]], [0.0, i["Fy"], i["Cy"]], [0.0, 0.0, 1.0]])
        cams[int(c["CameraId"])] = MmpCamera(
            int(c["CameraId"]),
            k,
            np.array(e["Rotation"], dtype=float).reshape(3, 3),
            np.array(e["Translation"], dtype=float),
        )
    s = raw["Space"]
    space = MmpSpace(
        s["MinU"] / 1000.0,
        s["MaxU"] / 1000.0,
        s["MinV"] / 1000.0,
        s["MaxV"] / 1000.0,
        s["VoxelSizeInMM"] / 1000.0,
    )
    return cams, space


class MmpSequence:
    """One labelled sequence: label and top-down zips plus the scene calibration."""

    def __init__(self, root: Path, scene: str, name: str) -> None:
        self.name = name
        self.cams, self.space = load_calibration(root / scene / "calibrations.json")
        self._labels = zipfile.ZipFile(root / scene / "labels.zip")
        self._top = zipfile.ZipFile(root / scene / "topdown.zip")
        self._label_names = set(self._labels.namelist())
        self._prefix = name
        self.frames = sorted(
            {
                int(m.group(1))
                for n in self._label_names
                if (m := re.search(r"rgb_(\d+)_\d+\.json$", n))
            }
        )

    def ground_truth(self, frame: int) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        text = self._top.read(f"{self._prefix}/topdown_{frame:05d}.csv").decode("utf-8")
        for line in text.strip().splitlines():
            a = [x.strip() for x in line.split(",")]
            out[int(a[0])] = (
                self.space.min_u_m
                + float(a[2]) * self.space.voxel_m,  # the second index column runs along U
                self.space.min_v_m + float(a[1]) * self.space.voxel_m,
            )
        return out

    def boxes(self, frame: int, camera: int) -> dict[int, tuple[float, float, float, float]]:
        name = f"{self._prefix}/rgb_{frame:05d}_{camera}.json"
        if name not in self._label_names:
            return {}
        return {
            int(pid): (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            for pid, b in json.loads(self._labels.read(name)).items()
        }
