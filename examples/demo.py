"""Two calibrated cameras watch two people cross an overlap zone; one then sprints impossibly fast."""

from __future__ import annotations

import numpy as np

from multicam_fusion import (
    Detection,
    GroundPlaneTracker,
    TrackerConfig,
    apply_homography,
    covariance_ellipse,
)
from multicam_fusion.sim import H_W2P, h_p2w


def det(cam: str, xy: tuple[float, float]) -> Detection:
    u, v = apply_homography(H_W2P[cam], np.array([xy]))[0]
    return Detection(cam, (u - 10, v - 60, u + 10, v))  # box whose bottom-centre is the foot


def main() -> None:
    tracker = GroundPlaneTracker(
        {"A": h_p2w()["A"], "B": h_p2w()["B"]}, TrackerConfig(v_max_mps=8.0, accel_sigma=20.0)
    )
    rng = np.random.default_rng(0)
    walker = np.array([16.0, 10.0])
    runner = np.array([20.0, 4.0])
    for frame in range(30):
        walker += np.array([0.12, 0.0])  # 1.2 m/s
        runner += np.array([0.0, 0.1]) if frame < 15 else np.array([0.0, 0.9])  # 1 m/s, then 9 m/s
        dets = []
        for cam in ("A", "B"):  # both cameras see both people (overlap zone 14-26 m)
            for p in (walker, runner):
                if 14 <= p[0] <= 26:
                    dets.append(det(cam, tuple(p + rng.normal(0, 0.15, 2))))
        confirmed = tracker.step(0.1, dets)
    print(f"{len(dets)} detections this frame (2 cameras x 2 people), {len(tracker.tracks)} tracks:")
    for t in sorted(confirmed, key=lambda t: t.track_id):
        major, minor, _ = covariance_ellipse(t.p[:2, :2], 0.95)
        flags = ", ".join(sorted(t.flags)) or "none"
        print(
            f"  track {t.track_id}: x={t.position[0]:5.1f} y={t.position[1]:5.1f} m  "
            f"speed {t.speed:4.1f} m/s  95% ellipse {major:.2f} x {minor:.2f} m  flags: {flags}"
        )


if __name__ == "__main__":
    main()
