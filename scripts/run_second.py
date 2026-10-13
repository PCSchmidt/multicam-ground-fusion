"""Design of the stricter track-creation rule on Wildtrack development frames (docs/PROTOCOL_SECOND_SCENE.md).

    python scripts/run_second.py tune     # writes results/second_chosen.json (config A = real_chosen D, config B = tuned)
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace

import run_real as R

GRID_FRAMES = (1, 2, 3)
GRID_COASTING = (True, False)


def tune() -> None:
    chosen = json.loads((R.RESULTS / "real_chosen.json").read_text(encoding="utf-8"))["D"]
    base = replace(
        R.DEFAULT,
        meas_sigma_m=chosen["meas_sigma_m"],
        accel_sigma=chosen["accel_sigma"],
        min_hits=chosen["min_hits"],
    )
    rows = []
    best = None
    for score, cf, coast in itertools.product(R.SCORE_GRID, GRID_FRAMES, GRID_COASTING):
        cfg = replace(base, confirm_frames=cf, report_coasting=coast)
        run = R.track(R.DEV, "D", score, tuple(range(7)), cfg)
        r = R.evaluate(R.gt_frames(R.DEV), run.pred, 0.5)
        row = {"score": score, "confirm_frames": cf, "report_coasting": coast, **R.summary(r)}
        rows.append(row)
        print(row, flush=True)
        # ties go to the simpler rule: fewer confirm frames, then coasting reported (the original behaviour)
        key = (round(r.mota, 6), -cf, coast)
        if best is None or key > best[0]:
            best = (key, row)
    assert best is not None
    out = {
        "A": {**chosen, "confirm_frames": 1, "report_coasting": True},
        "B": {
            "score": best[1]["score"],
            "meas_sigma_m": chosen["meas_sigma_m"],
            "accel_sigma": chosen["accel_sigma"],
            "min_hits": chosen["min_hits"],
            "confirm_frames": best[1]["confirm_frames"],
            "report_coasting": best[1]["report_coasting"],
        },
        "grid": rows,
    }
    (R.RESULTS / "second_chosen.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("chosen B:", out["B"])


def tune_consensus() -> None:
    """Exploratory config C (Deviation 1): camera consensus, designed on the same development frames."""
    path = R.RESULTS / "second_chosen.json"
    cur = json.loads(path.read_text(encoding="utf-8"))
    b = cur["B"]
    base = replace(
        R.DEFAULT, meas_sigma_m=b["meas_sigma_m"], accel_sigma=b["accel_sigma"], min_hits=b["min_hits"],
        report_coasting=False, confirm_frames=1,
    )  # fmt: skip
    rows = []
    best = None
    for score, cc in itertools.product(R.SCORE_GRID, (1, 2, 3, 4)):
        run = R.track(R.DEV, "D", score, tuple(range(7)), replace(base, confirm_cameras=cc))
        r = R.evaluate(R.gt_frames(R.DEV), run.pred, 0.5)
        row = {"score": score, "confirm_cameras": cc, **R.summary(r)}
        rows.append(row)
        print(row, flush=True)
        key = (round(r.mota, 6), -cc)
        if best is None or key > best[0]:
            best = (key, row)
    assert best is not None
    cur["C"] = {
        "score": best[1]["score"], "meas_sigma_m": b["meas_sigma_m"], "accel_sigma": b["accel_sigma"],
        "min_hits": b["min_hits"], "confirm_frames": 1, "report_coasting": False, "confirm_cameras": best[1]["confirm_cameras"],
    }  # fmt: skip
    cur["grid_C"] = rows
    path.write_text(json.dumps(cur, indent=1), encoding="utf-8")
    print("chosen C:", cur["C"])


if __name__ == "__main__":
    {"tune": tune, "tune_c": tune_consensus}[sys.argv[1]]()
