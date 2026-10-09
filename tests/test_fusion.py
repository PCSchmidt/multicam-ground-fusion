from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from multicam_fusion import (
    Detection,
    GroundPlaneTracker,
    TrackerConfig,
    apply_homography,
    covariance_ellipse,
    estimate_homography,
)
from multicam_fusion.sim import H_W2P, h_p2w, simulate


def test_homography_roundtrip() -> None:
    world = np.array([[3.0, 4.0], [20.0, 30.0], [39.0, 1.0]])
    pix = apply_homography(H_W2P["A"], world)
    assert np.allclose(apply_homography(h_p2w()["A"], pix), world, atol=1e-9)


def test_dlt_recovers_homography() -> None:
    src = np.array([[0, 0], [40, 0], [40, 40], [0, 40], [20, 20.0]])
    dst = apply_homography(H_W2P["B"], src)
    est = estimate_homography(src, dst)
    assert np.allclose(apply_homography(est, src), dst, atol=1e-6)
    with pytest.raises(ValueError):
        estimate_homography(src[:3], dst[:3])


def test_horizon_point_rejected() -> None:
    h = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 0.0]])
    with pytest.raises(ValueError):
        apply_homography(h, np.array([[1.0, 1.0]]))


def test_ellipse_axes_match_eigenvalues() -> None:
    major, minor, _ = covariance_ellipse(np.diag([4.0, 1.0]), 0.95)
    assert major == pytest.approx(2 * minor)
    assert major == pytest.approx(np.sqrt(5.991464547107979 * 4.0))


def det(cam: str, world_xy: tuple[float, float]) -> Detection:
    u, v = apply_homography(H_W2P[cam], np.array([world_xy]))[0]
    return Detection(cam, (u - 10, v - 60, u + 10, v))


def test_two_cameras_one_object_make_one_track() -> None:
    tr = GroundPlaneTracker({"A": h_p2w()["A"], "B": h_p2w()["B"]})
    for _ in range(6):
        out = tr.step(0.1, [det("A", (20.0, 20.0)), det("B", (20.0, 20.0))])
    assert len(out) == 1 and len(tr.tracks) == 1


def test_far_detection_is_gated_not_merged() -> None:
    tr = GroundPlaneTracker({"A": h_p2w()["A"]})
    for _ in range(4):
        tr.step(0.1, [det("A", (5.0, 5.0))])
    tr.step(0.1, [det("A", (5.0, 5.0)), det("A", (24.0, 24.0))])
    assert len(tr.tracks) == 2


def test_track_deleted_after_misses() -> None:
    tr = GroundPlaneTracker({"A": h_p2w()["A"]}, TrackerConfig(max_misses=2))
    for _ in range(4):
        tr.step(0.1, [det("A", (5.0, 5.0))])
    for _ in range(4):
        tr.step(0.1, [])
    assert tr.tracks == []


def test_implausible_speed_is_flagged_not_hidden() -> None:
    tr = GroundPlaneTracker(
        {"A": h_p2w()["A"]}, TrackerConfig(v_max_mps=1.0, gate_prob=0.999999, accel_sigma=50.0)
    )
    x = 5.0
    for _ in range(12):
        x += 0.8  # 8 m/s at dt = 0.1
        tr.step(0.1, [det("A", (x, 10.0))])
    assert any("IMPLAUSIBLE_SPEED" in t.flags for t in tr.tracks)


def test_fusion_beats_single_camera_in_overlap_heavy_scene() -> None:
    both = [simulate(s, ("A", "B"), dropout=0.4).mota for s in range(5)]
    single = [simulate(s, ("A",), dropout=0.4).mota for s in range(5)]
    assert np.mean(both) > np.mean(single)


@settings(max_examples=15, deadline=None)
@given(st.integers(0, 5000))
def test_covariance_stays_symmetric_psd(seed: int) -> None:
    rng = np.random.default_rng(seed)
    tr = GroundPlaneTracker({"A": h_p2w()["A"]})
    for _ in range(15):
        pos = (10 + rng.normal(0, 0.2), 10 + rng.normal(0, 0.2))
        tr.step(0.1, [det("A", pos)] if rng.random() < 0.8 else [])
    for t in tr.tracks:
        assert np.allclose(t.p, t.p.T, atol=1e-8)
        assert float(np.min(np.linalg.eigvalsh((t.p + t.p.T) / 2))) > -1e-8
