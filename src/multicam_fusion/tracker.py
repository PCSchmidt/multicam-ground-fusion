"""Ground-plane tracker. Detections are untrusted evidence; association is deterministic."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from .geometry import Array, apply_homography, bottom_center

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    camera: str
    bbox: BBox


@dataclass(frozen=True)
class TrackerConfig:
    meas_sigma_m: float = 0.5  # std of a projected ground point, metres (tuned on dev seeds 0-4)
    accel_sigma: float = 1.0  # process noise: std of unmodelled acceleration, m/s^2
    gate_prob: float = 0.99  # chi-square gate on squared Mahalanobis distance (2 dof)
    max_misses: int = 5
    min_hits: int = 3  # confirmed after this many updates
    v_max_mps: float = 12.0  # physical plausibility ceiling; above it a track is flagged
    init_vel_sigma: float = 3.0


@dataclass
class Track:
    track_id: int
    x: Array  # [px, py, vx, vy]
    p: Array
    hits: int = 1
    misses: int = 0
    flags: set[str] = field(default_factory=set)

    @property
    def position(self) -> Array:
        return self.x[:2]

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))


class GroundPlaneTracker:
    def __init__(self, homographies: dict[str, Array], cfg: TrackerConfig | None = None) -> None:
        self.h = homographies
        self.cfg = cfg or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_id = 0
        self._gate = float(chi2.ppf(self.cfg.gate_prob, df=2))
        r = self.cfg.meas_sigma_m**2
        self._r = np.eye(2) * r
        self._hmat = np.hstack([np.eye(2), np.zeros((2, 2))])
        self.rejected = 0  # associations refused by the gate, for audit

    # --- Kalman primitives -------------------------------------------------------------
    def _predict(self, t: Track, dt: float) -> None:
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        q_a = self.cfg.accel_sigma**2
        g = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        t.x = f @ t.x
        t.p = f @ t.p @ f.T + g @ g.T * q_a

    def mahalanobis_sq(self, t: Track, z: Array) -> float:
        s = self._hmat @ t.p @ self._hmat.T + self._r
        y = z - self._hmat @ t.x
        return float(y @ np.linalg.solve(s, y))

    def _update(self, t: Track, z: Array) -> None:
        s = self._hmat @ t.p @ self._hmat.T + self._r
        k = t.p @ self._hmat.T @ np.linalg.inv(s)
        t.x = t.x + k @ (z - self._hmat @ t.x)
        t.p = (np.eye(4) - k @ self._hmat) @ t.p
        t.hits += 1
        t.misses = 0
        if t.speed > self.cfg.v_max_mps:
            t.flags.add("IMPLAUSIBLE_SPEED")  # flagged for review; never silently "fixed"

    def _spawn(self, z: Array) -> None:
        p = np.diag([self.cfg.meas_sigma_m**2] * 2 + [self.cfg.init_vel_sigma**2] * 2)
        self.tracks.append(Track(self._next_id, np.array([z[0], z[1], 0.0, 0.0]), p))
        self._next_id += 1

    # --- one time step -----------------------------------------------------------------
    def step(self, dt: float, detections: list[Detection]) -> list[Track]:
        """Advance by dt, fuse detections camera by camera, return confirmed tracks."""
        for t in self.tracks:
            self._predict(t, dt)
        touched: set[int] = set()
        for cam in sorted({d.camera for d in detections}):
            dets = [d for d in detections if d.camera == cam]
            pts = apply_homography(self.h[cam], np.array([bottom_center(d.bbox) for d in dets]))
            matched_dets: set[int] = set()
            if self.tracks:
                cost = np.array([[self.mahalanobis_sq(t, z) for z in pts] for t in self.tracks])
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols, strict=True):
                    if cost[r, c] <= self._gate:
                        self._update(self.tracks[r], pts[c])
                        touched.add(self.tracks[r].track_id)
                        matched_dets.add(int(c))
                    else:
                        self.rejected += 1
            for c in range(len(dets)):
                if c not in matched_dets:
                    self._spawn(pts[c])
                    touched.add(self._next_id - 1)
        for t in self.tracks:
            if t.track_id not in touched:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.cfg.max_misses]
        return self.confirmed()

    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.hits >= self.cfg.min_hits]
