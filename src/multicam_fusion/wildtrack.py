"""Wildtrack loader: calibration -> ground-plane homographies, annotations -> ground truth (numpy/scipy only).

World coordinates follow the dataset README: a 480 x 1440 grid, X first, 2.5 cm spacing, origin (-3.0, -9.0) m;
calibration translations are in centimetres, so they are divided by 100 to work in metres.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import Array

CAMERAS = ("CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3")
GRID_W = 480
GRID_STEP_M = 0.025
GRID_ORIGIN_M = (-3.0, -9.0)


def position_to_world(position_id: int) -> tuple[float, float]:
    """Grid index -> (x, y) in metres (the README formula, with the intended integer grid arithmetic)."""
    return (
        GRID_ORIGIN_M[0] + GRID_STEP_M * (position_id % GRID_W),
        GRID_ORIGIN_M[1] + GRID_STEP_M * (position_id // GRID_W),
    )


def _floats(text: str | None) -> list[float]:
    return [float(x) for x in re.split(r"\s+", (text or "").strip()) if x]


@dataclass(frozen=True)
class CameraModel:
    name: str
    k: Array  # 3x3 intrinsics (undistorted images)
    rvec: Array  # world -> camera rotation vector
    tvec_m: Array  # world -> camera translation, metres

    def world_to_pixel_h(self) -> Array:
        """Homography mapping ground-plane (x, y, 1) in metres to pixels (plane z = 0)."""
        r = Rotation.from_rotvec(self.rvec).as_matrix()
        return np.asarray(
            self.k @ np.column_stack([r[:, 0], r[:, 1], self.tvec_m]), dtype=np.float64
        )

    def pixel_to_world_h(self) -> Array:
        return np.asarray(np.linalg.inv(self.world_to_pixel_h()), dtype=np.float64)


def load_camera(root: Path, name: str) -> CameraModel:
    intr = ET.parse(root / "calibrations" / "intrinsic_zero" / f"intr_{name}.xml").getroot()
    k = np.array(_floats(intr.findtext("camera_matrix/data"))).reshape(3, 3)
    extr = ET.parse(root / "calibrations" / "extrinsic" / f"extr_{name}.xml").getroot()
    rvec = np.array(_floats(extr.findtext("rvec")))
    tvec = np.array(_floats(extr.findtext("tvec"))) / 100.0
    return CameraModel(name, k, rvec, tvec)


def load_cameras(root: Path) -> list[CameraModel]:
    return [load_camera(root, n) for n in CAMERAS]


@dataclass(frozen=True)
class GtPerson:
    person_id: int
    xy_m: tuple[float, float]
    boxes: dict[
        int, tuple[float, float, float, float]
    ]  # camera index -> (xmin, ymin, xmax, ymax) where visible


def load_frame(root: Path, frame: int) -> list[GtPerson]:
    """Ground truth of one annotated frame; frame is the file number (0, 5, 10, ...)."""
    rows = json.loads(
        (root / "annotations_positions" / f"{frame:08d}.json").read_text(encoding="utf-8")
    )
    out = []
    for r in rows:
        boxes = {
            int(v["viewNum"]): (
                float(v["xmin"]),
                float(v["ymin"]),
                float(v["xmax"]),
                float(v["ymax"]),
            )
            for v in r["views"]
            if v["xmin"] != -1
        }
        out.append(GtPerson(int(r["personID"]), position_to_world(int(r["positionID"])), boxes))
    return out


def frame_numbers(root: Path) -> list[int]:
    return sorted(int(p.stem) for p in (root / "annotations_positions").glob("*.json"))
