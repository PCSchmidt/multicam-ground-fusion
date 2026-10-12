from __future__ import annotations

import numpy as np

from multicam_fusion.geometry import apply_homography
from multicam_fusion.wildtrack import CameraModel, position_to_world


def test_grid_index_to_world_follows_the_readme() -> None:
    assert position_to_world(0) == (-3.0, -9.0)
    assert position_to_world(1) == (-3.0 + 0.025, -9.0)
    assert position_to_world(480) == (-3.0, -9.0 + 0.025)
    x, y = position_to_world(480 * 1440 - 1)
    assert abs(x - (-3.0 + 0.025 * 479)) < 1e-12 and abs(y - (-9.0 + 0.025 * 1439)) < 1e-12


def test_ground_homography_inverts_and_matches_projection() -> None:
    cam = CameraModel(
        "t",
        np.array([[1700.0, 0.0, 960.0], [0.0, 1700.0, 540.0], [0.0, 0.0, 1.0]]),
        np.array([1.7, 0.4, -0.3]),
        np.array([-5.0, 0.4, 9.0]),
    )
    pts = np.array([[0.0, 0.0], [3.0, -2.0], [-1.5, 6.0]])
    pix = apply_homography(cam.world_to_pixel_h(), pts)
    back = apply_homography(cam.pixel_to_world_h(), pix)
    assert np.allclose(back, pts, atol=1e-9)
    # the same point through the full 3-D pinhole model (plane z = 0)
    from scipy.spatial.transform import Rotation

    r = Rotation.from_rotvec(cam.rvec).as_matrix()
    for p, uv in zip(pts, pix, strict=True):
        c = cam.k @ (r @ np.array([p[0], p[1], 0.0]) + cam.tvec_m)
        assert np.allclose(c[:2] / c[2], uv, atol=1e-9)
