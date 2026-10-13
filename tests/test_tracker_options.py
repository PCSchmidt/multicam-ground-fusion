from __future__ import annotations

import numpy as np

from multicam_fusion.tracker import Detection, GroundPlaneTracker, TrackerConfig

H = {"A": np.eye(3)}


def det(x: float, y: float) -> Detection:
    return Detection("A", (x - 1.0, y - 2.0, x + 1.0, y))  # bottom-centre is (x, y)


def run(cfg: TrackerConfig, frames: list[list[Detection]]) -> list[int]:
    tr = GroundPlaneTracker(H, cfg)
    return [len(tr.step(0.5, d)) for d in frames]


def test_defaults_reproduce_the_original_confirmation() -> None:
    cfg = TrackerConfig(min_hits=2)
    assert run(cfg, [[det(5, 5)], [det(5, 5)], [det(5, 5)]]) == [0, 1, 1]


def test_confirm_frames_rejects_a_single_frame_false_detection() -> None:
    # two detections of the same point in one frame come from two cameras in real use; here from one camera two boxes
    one_frame = [[det(5, 5), det(5.1, 5.1)], [], []]
    persistent = [[det(5, 5)], [det(5, 5)], [det(5, 5)], [det(5, 5)]]
    loose = TrackerConfig(min_hits=1, confirm_frames=1)
    strict = TrackerConfig(min_hits=1, confirm_frames=2)
    assert run(loose, one_frame)[0] >= 1  # appears at once
    assert run(strict, one_frame)[0] == 0  # never confirmed by one frame
    assert run(strict, persistent) == [0, 1, 1, 1]


def test_report_coasting_false_hides_tracks_without_a_current_update() -> None:
    frames = [[det(5, 5)], [det(5, 5)], [], []]
    assert run(TrackerConfig(min_hits=2, report_coasting=True), frames) == [0, 1, 1, 1]
    assert run(TrackerConfig(min_hits=2, report_coasting=False), frames) == [0, 1, 0, 0]
