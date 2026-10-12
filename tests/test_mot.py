from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from multicam_fusion.mot import Frame, evaluate


def line(n: int, ids: tuple[int, ...] = (0, 1)) -> list[Frame]:
    return [{i: (float(t) * 0.1, float(i) * 5.0) for i in ids} for t in range(n)]


def test_perfect_tracking_scores_one() -> None:
    g = line(20)
    r = evaluate(g, [dict(f) for f in g])
    assert (r.mota, r.idf1, r.id_switches, r.fp, r.fn) == (1.0, 1.0, 0, 0, 0)
    assert r.motp_m == 0.0


def test_relabelling_is_an_id_switch_but_not_a_position_error() -> None:
    g = line(20, (0,))
    p: list[Frame] = [{(7 if t < 10 else 8): f[0]} for t, f in enumerate(g)]
    r = evaluate(g, p)
    assert r.id_switches == 1 and r.fp == 0 and r.fn == 0
    assert r.switch_events == ((10, 0, 7, 8),)
    assert r.mota == 1.0 - 1 / 20
    assert (
        abs(r.idf1 - 2 * 10 / (20 + 20)) < 1e-12
    )  # only the better half of the trajectory is identity-correct


def test_sticky_match_survives_a_closer_competitor() -> None:
    g: list[Frame] = [{0: (0.0, 0.0)}, {0: (0.0, 0.0)}]
    p: list[Frame] = [{1: (0.1, 0.0)}, {1: (0.4, 0.0), 2: (0.05, 0.0)}]
    r = evaluate(g, p)
    assert (
        r.id_switches == 0 and r.fp == 1 and r.fn == 0
    )  # track 1 is kept; track 2 is a false positive


def test_missing_and_extra_detections_are_counted() -> None:
    g: list[Frame] = [{0: (0.0, 0.0)}, {0: (0.0, 0.0)}]
    p: list[Frame] = [{}, {3: (5.0, 5.0)}]
    r = evaluate(g, p)
    assert (r.tp, r.fp, r.fn) == (0, 1, 2)
    assert r.mota == 1.0 - 3 / 2


@given(st.integers(min_value=1, max_value=30), st.integers(min_value=0, max_value=5))
def test_mota_identity(n_frames: int, n_people: int) -> None:
    g = line(n_frames, tuple(range(n_people)))
    r = evaluate(g, [{} for _ in g])
    assert r.tp == 0 and r.fn == r.n_gt and r.fp == 0
    assert r.mota == (1.0 - 1.0 if r.n_gt else 1.0)


def test_per_frame_stats_sum_to_the_totals() -> None:
    g = line(20, (0, 1))
    p: list[Frame] = [{(7 if t < 10 else 8): f[0]} for t, f in enumerate(g)]  # person 1 never tracked
    r = evaluate(g, p)
    assert sum(x[0] for x in r.frame_stats) == r.n_gt
    assert sum(x[1] for x in r.frame_stats) == r.fn
    assert sum(x[2] for x in r.frame_stats) == r.fp
    assert sum(x[3] for x in r.frame_stats) == r.id_switches == 1
