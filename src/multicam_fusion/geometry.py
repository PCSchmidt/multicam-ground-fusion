from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def apply_homography(h: Array, uv: Array) -> Array:
    """Map pixel coordinates (N,2) to ground-plane coordinates (N,2)."""
    pts = np.atleast_2d(uv).astype(np.float64)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    w = homog[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("point maps to the horizon (w ~ 0)")
    return np.asarray(homog[:, :2] / w, dtype=np.float64)


def estimate_homography(src: Array, dst: Array) -> Array:
    """Direct linear transform from >= 4 point pairs, normalised so H[2,2] = 1."""
    if len(src) < 4 or src.shape != dst.shape:
        raise ValueError("need >= 4 matching point pairs")
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    return np.asarray(h / h[2, 2], dtype=np.float64)


def bottom_center(bbox: tuple[float, float, float, float]) -> Array:
    x0, _, x1, y1 = bbox
    return np.array([(x0 + x1) / 2.0, y1], dtype=np.float64)


def covariance_ellipse(cov2: Array, confidence: float = 0.95) -> tuple[float, float, float]:
    """(semi_major, semi_minor, angle_rad) of the confidence ellipse of a 2x2 covariance."""
    from scipy.stats import chi2

    vals, vecs = np.linalg.eigh(cov2)
    vals = np.clip(vals, 0.0, None)
    scale = float(chi2.ppf(confidence, df=2))
    major = float(np.sqrt(scale * vals[1]))
    minor = float(np.sqrt(scale * vals[0]))
    angle = float(np.arctan2(vecs[1, 1], vecs[0, 1]))
    return major, minor, angle
