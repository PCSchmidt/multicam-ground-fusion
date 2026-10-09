"""Seeded synthetic multi-camera scenario and CLEAR-MOT style metrics (no external dataset)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import Array, apply_homography
from .tracker import Detection, GroundPlaneTracker, TrackerConfig

# World (m) -> pixel homographies, ~10 px/m with mild perspective.
H_W2P: dict[str, Array] = {
    "A": np.array([[10.0, 1.0, 50.0], [0.5, 9.0, 40.0], [0.0005, 0.0003, 1.0]]),
    "B": np.array([[-9.0, 2.0, 500.0], [1.0, 11.0, 20.0], [-0.0004, 0.0006, 1.0]]),
}
# Each camera sees only part of the 40x40 m yard; the overlap is where fusion happens.
FOV_X: dict[str, tuple[float, float]] = {"A": (0.0, 26.0), "B": (14.0, 40.0)}


def h_p2w() -> dict[str, Array]:
    return {k: np.linalg.inv(v) for k, v in H_W2P.items()}


@dataclass(frozen=True)
class Metrics:
    mota: float
    motp_m: float
    id_switches: int
    false_negatives: int
    false_positives: int
    n_gt: int


def simulate(
    seed: int,
    cameras: tuple[str, ...] = ("A", "B"),
    n_targets: int = 4,
    frames: int = 200,
    dt: float = 0.1,
    dropout: float = 0.2,
    pixel_sigma: float = 3.0,
    cfg: TrackerConfig | None = None,
) -> Metrics:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(5, 35, (n_targets, 2))
    ang = rng.uniform(0, 2 * np.pi, n_targets)
    vel = np.stack([np.cos(ang), np.sin(ang)], axis=1) * rng.uniform(0.8, 2.0, (n_targets, 1))
    tracker = GroundPlaneTracker({c: h_p2w()[c] for c in cameras}, cfg)
    last_match: dict[int, int] = {}
    fn = fp = idsw = n_gt = 0
    dist_sum, n_match = 0.0, 0
    for _ in range(frames):
        pos += vel * dt
        for ax in (0, 1):  # reflect at yard walls
            low, high = pos[:, ax] < 0, pos[:, ax] > 40
            vel[low | high, ax] *= -1
            pos[:, ax] = np.clip(pos[:, ax], 0, 40)
        dets: list[Detection] = []
        for cam in cameras:
            lo, hi = FOV_X[cam]
            for p in pos:
                if not (lo <= p[0] <= hi) or rng.random() < dropout:
                    continue
                px = apply_homography(H_W2P[cam], p[None, :])[0] + rng.normal(0, pixel_sigma, 2)
                dets.append(Detection(cam, (px[0] - 10, px[1] - 60, px[0] + 10, px[1])))
        tracks = tracker.step(dt, dets)
        n_gt += n_targets
        if not tracks:
            fn += n_targets
            continue
        cost = np.linalg.norm(pos[:, None, :] - np.array([t.position for t in tracks])[None], axis=2)
        rows, cols = linear_sum_assignment(cost)
        matched_gt = set()
        for r, c in zip(rows, cols, strict=True):
            if cost[r, c] > 1.5:
                continue
            matched_gt.add(int(r))
            tid = tracks[c].track_id
            if r in last_match and last_match[int(r)] != tid:
                idsw += 1
            last_match[int(r)] = tid
            dist_sum += float(cost[r, c])
            n_match += 1
        fn += n_targets - len(matched_gt)
        fp += len(tracks) - len(matched_gt)
    return Metrics(
        mota=1.0 - (fn + fp + idsw) / n_gt,
        motp_m=dist_sum / n_match if n_match else float("nan"),
        id_switches=idsw,
        false_negatives=fn,
        false_positives=fp,
        n_gt=n_gt,
    )
