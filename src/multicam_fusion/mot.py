"""CLEAR-MOT and IDF1 on the ground plane, with sticky matching (Bernardin & Stiefelhagen 2008).

Ground truth and predictions are lists over frames of {id: (x, y)} in metres. A prediction matches a ground-truth
person if it is within `threshold_m`. A match made in the previous frame is kept while it stays within the threshold,
so an identity switch is a change of matched track id for the same person, not a re-assignment artefact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

Frame = dict[int, tuple[float, float]]


@dataclass(frozen=True)
class MotResult:
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    id_switches: int
    mota: float
    motp_m: float
    idf1: float
    idtp: int
    switch_events: tuple[tuple[int, int, int, int], ...] = field(
        default=()
    )  # (frame, gt, old_track, new_track)
    fp_tracks: tuple[tuple[int, int], ...] = field(
        default=()
    )  # (frame, track) unmatched predictions
    frame_stats: tuple[tuple[int, int, int, int], ...] = field(
        default=()
    )  # per frame: (n_gt, fn, fp, id_switches), for block bootstrap


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def evaluate(gt: list[Frame], pred: list[Frame], threshold_m: float = 0.5) -> MotResult:
    if len(gt) != len(pred):
        raise ValueError("gt and pred must cover the same frames")
    last: dict[int, int] = {}  # gt id -> track id at its previous match (for switches)
    prev: dict[int, int] = {}  # gt id -> track id matched in the previous frame (sticky)
    tp = fp = fn = idsw = 0
    dsum = 0.0
    n_gt = n_pred = 0
    events: list[tuple[int, int, int, int]] = []
    fp_tracks: list[tuple[int, int]] = []
    pair_count: dict[tuple[int, int], int] = {}
    stats: list[tuple[int, int, int, int]] = []
    for t, (g, p) in enumerate(zip(gt, pred, strict=True)):
        n_gt += len(g)
        n_pred += len(p)
        matches: dict[int, int] = {}
        for gid, tid in prev.items():
            if gid in g and tid in p and _dist(g[gid], p[tid]) <= threshold_m:
                matches[gid] = tid
        taken = set(matches.values())
        gl = [i for i in g if i not in matches]
        pl = [i for i in p if i not in taken]
        if gl and pl:
            cost = np.array([[_dist(g[i], p[j]) for j in pl] for i in gl])
            for r, c in zip(*linear_sum_assignment(cost), strict=True):
                if cost[r, c] <= threshold_m:
                    matches[gl[int(r)]] = pl[int(c)]
        sw_before = idsw
        for gid, tid in matches.items():
            if gid in last and last[gid] != tid:
                idsw += 1
                events.append((t, gid, last[gid], tid))
            last[gid] = tid
            dsum += _dist(g[gid], p[tid])
            pair_count[(gid, tid)] = pair_count.get((gid, tid), 0) + 1
        tp += len(matches)
        fn += len(g) - len(matches)
        fp += len(p) - len(matches)
        fp_tracks.extend((t, tid) for tid in p if tid not in set(matches.values()))
        stats.append((len(g), len(g) - len(matches), len(p) - len(matches), idsw - sw_before))
        prev = matches
    # IDF1: best one-to-one assignment of whole ground-truth trajectories to whole track trajectories
    gids = sorted({a for a, _ in pair_count})
    tids = sorted({b for _, b in pair_count})
    idtp = 0
    if gids and tids:
        m = np.zeros((len(gids), len(tids)))
        for (a, b), n in pair_count.items():
            m[gids.index(a), tids.index(b)] = n
        r, c = linear_sum_assignment(-m)
        idtp = int(m[r, c].sum())
    idf1 = 2 * idtp / (n_gt + n_pred) if (n_gt + n_pred) else 1.0
    return MotResult(
        n_gt=n_gt,
        n_pred=n_pred,
        tp=tp,
        fp=fp,
        fn=fn,
        id_switches=idsw,
        mota=1.0 - (fn + fp + idsw) / n_gt if n_gt else 1.0,
        motp_m=dsum / tp if tp else float("nan"),
        idf1=idf1,
        idtp=idtp,
        switch_events=tuple(events),
        fp_tracks=tuple(fp_tracks),
        frame_stats=tuple(stats),
    )
