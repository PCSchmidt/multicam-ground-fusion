"""Real-data study on Wildtrack (docs/PROTOCOL_REAL.md).

    python scripts/run_real.py tune        # development frames only: grid search, writes results/real_chosen.json
    python scripts/run_real.py detector    # detector precision/recall per camera on development frames
    python scripts/run_real.py test        # ONE-SHOT: evaluates the frozen configuration on the test frames

The test run refuses to overwrite results/real_test.json.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from multicam_fusion.geometry import Array, apply_homography
from multicam_fusion.mot import Frame, MotResult, evaluate
from multicam_fusion.tracker import Detection, GroundPlaneTracker, TrackerConfig
from multicam_fusion.wildtrack import (
    CAMERAS,
    GtPerson,
    frame_numbers,
    load_cameras,
    load_frame,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Wildtrack_dataset"
RESULTS = ROOT / "results"
DT = 0.5
AREA = (-3.0, 9.0, -9.0, 27.0)  # x0, x1, y0, y1 (m): the annotated ground area
FRAMES = frame_numbers(DATA)
DEV, TEST = FRAMES[:200], FRAMES[200:]
CAMS = load_cameras(DATA)
H_P2W = {CAMERAS[i]: CAMS[i].pixel_to_world_h() for i in range(7)}
GT = {f: load_frame(DATA, f) for f in FRAMES}
DET = np.load(ROOT / "data" / "detections.npz")

TRACK_GRID = {
    "meas_sigma_m": [0.2, 0.3, 0.5, 0.75, 1.0, 1.5],
    "accel_sigma": [0.5, 1.0, 2.0, 3.0],
    "min_hits": [2, 3],
}
SCORE_GRID = [0.5, 0.7, 0.9]
DEFAULT = TrackerConfig()  # the synthetic-study defaults (meas 0.5, accel 1.0, min_hits 3)
NO_GATE = 1.0 - 1e-12


def in_area(xy: Array) -> bool:
    return bool(AREA[0] <= xy[0] < AREA[1] and AREA[2] <= xy[1] < AREA[3])


def detections(frame: int, mode: str, score: float, cams: tuple[int, ...]) -> list[Detection]:
    out: list[Detection] = []
    for ci in cams:
        if mode == "O":
            boxes = [p.boxes[ci] for p in GT[frame] if ci in p.boxes]
        else:
            arr = DET[f"c{ci}_f{frame}"]
            boxes = [tuple(map(float, r[:4])) for r in arr if r[4] >= score]
        for b in boxes:
            xy = apply_homography(H_P2W[CAMERAS[ci]], np.array([[(b[0] + b[2]) / 2, b[3]]]))[0]
            if in_area(xy):
                out.append(Detection(CAMERAS[ci], (b[0], b[1], b[2], b[3])))
    return out


@dataclass
class Run:
    pred: list[Frame]
    speeds: list[dict[int, float]]  # confirmed track id -> speed, per frame
    rejected: int


def track(
    frames: list[int],
    mode: str,
    score: float,
    cams: tuple[int, ...],
    cfg: TrackerConfig,
    live_only: bool = False,
) -> Run:
    tr = GroundPlaneTracker({CAMERAS[i]: H_P2W[CAMERAS[i]] for i in cams}, cfg)
    pred: list[Frame] = []
    speeds: list[dict[int, float]] = []
    for f in frames:
        ts = tr.step(DT, detections(f, mode, score, cams))
        if live_only:  # report only tracks updated in this frame, not coasting ones (docs/PROTOCOL_REAL.md, Deviation 1)
            ts = [t for t in ts if t.misses == 0]
        pred.append({t.track_id: (float(t.position[0]), float(t.position[1])) for t in ts})
        speeds.append({t.track_id: t.speed for t in ts})
    return Run(pred, speeds, tr.rejected)


def gt_frames(frames: list[int]) -> list[Frame]:
    return [{p.person_id: p.xy_m for p in GT[f]} for f in frames]


def summary(r: MotResult) -> dict[str, float]:
    return {
        "mota": r.mota,
        "motp_m": r.motp_m,
        "idf1": r.idf1,
        "id_switches": r.id_switches,
        "fp": r.fp,
        "fn": r.fn,
        "n_gt": r.n_gt,
        "n_pred": r.n_pred,
    }


def mota_boot(
    results: dict[str, MotResult], base: str, n_blocks: int = 10, draws: int = 2000
) -> dict[str, Any]:
    """Block bootstrap of MOTA per variant, and paired MOTA differences against `base`."""
    rng = np.random.default_rng(0)
    stats = {k: np.array(v.frame_stats, dtype=float) for k, v in results.items()}
    n = len(next(iter(stats.values())))
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = {
        k: np.array([a[edges[i] : edges[i + 1]].sum(axis=0) for i in range(n_blocks)])
        for k, a in stats.items()
    }

    def mota(b: Any) -> Any:
        return 1.0 - (b[..., 1] + b[..., 2] + b[..., 3]).sum(-1) / b[..., 0].sum(-1)

    idx = rng.integers(0, n_blocks, (draws, n_blocks))
    out: dict[str, Any] = {}
    draws_by = {k: mota(b[idx]) for k, b in blocks.items()}
    for k, d in draws_by.items():
        entry: dict[str, Any] = {"ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        if k != base:
            diff = d - draws_by[base]
            entry["diff_vs_base_ci95"] = [
                float(np.percentile(diff, 2.5)),
                float(np.percentile(diff, 97.5)),
            ]
        out[k] = entry
    return out


# ------------------------------------------------------------------------------------------ tuning


def tune_mode(mode: str) -> dict[str, Any]:
    all7 = tuple(range(7))
    best: tuple[float, int, dict[str, Any]] | None = None
    scores = [0.0] if mode == "O" else SCORE_GRID
    rows = []
    for score, ms, acc, mh in itertools.product(scores, *TRACK_GRID.values()):
        cfg = replace(DEFAULT, meas_sigma_m=ms, accel_sigma=acc, min_hits=mh)
        r = evaluate(gt_frames(DEV), track(DEV, mode, score, all7, cfg).pred, 0.5)
        rows.append({"score": score, "meas_sigma_m": ms, "accel_sigma": acc, "min_hits": mh, **summary(r)})
        key = (r.mota, -r.id_switches)
        if best is None or key > (best[0], -best[1]):
            best = (r.mota, r.id_switches, rows[-1])
    assert best is not None
    return {"best": best[2], "grid": rows}


def cmd_tune() -> None:
    RESULTS.mkdir(exist_ok=True)
    chosen = {}
    for mode in ("O", "D"):
        t = tune_mode(mode)
        chosen[mode] = {k: t["best"][k] for k in ("score", "meas_sigma_m", "accel_sigma", "min_hits")}
        print(mode, "chosen on development frames:", t["best"])
        (RESULTS / f"real_dev_grid_{mode}.json").write_text(json.dumps(t, indent=1), encoding="utf-8")
    (RESULTS / "real_chosen.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")


# ------------------------------------------------------------------------------------------ detector


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def detector_quality(frames: list[int], score: float) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for ci in range(7):
        tp = fp = fn = 0
        for f in frames:
            gts = [p.boxes[ci] for p in GT[f] if ci in p.boxes]
            dets = [(d.bbox) for d in detections(f, "D", score, (ci,))]
            used: set[int] = set()
            for g in gts:
                best, bi = 0.0, -1
                for j, d in enumerate(dets):
                    if j not in used and (v := iou(g, d)) > best:
                        best, bi = v, j
                if best >= 0.5:
                    used.add(bi)
                    tp += 1
                else:
                    fn += 1
            fp += len(dets) - len(used)
        out[CAMERAS[ci]] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def cmd_detector() -> None:
    chosen = json.loads((RESULTS / "real_chosen.json").read_text(encoding="utf-8"))
    q = detector_quality(DEV, chosen["D"]["score"])
    (RESULTS / "real_detector_dev.json").write_text(json.dumps(q, indent=1), encoding="utf-8")
    for k, v in q.items():
        print(f"{k}: precision {v['precision']:.3f} recall {v['recall']:.3f}")


# ------------------------------------------------------------------------------------------ test


def flag_analysis(run: Run, res: MotResult, ceiling: float) -> dict[str, float]:
    """Rising-edge speed flags vs identity switches / false-positive tracks within +-2 frames."""
    events: set[tuple[int, int]] = set()
    for t, sp in enumerate(run.speeds):
        for tid, v in sp.items():
            was = t > 0 and run.speeds[t - 1].get(tid, 0.0) > ceiling
            if v > ceiling and not was:
                events.add((t, tid))
    flagged_frames = sum(1 for sp in run.speeds for v in sp.values() if v > ceiling)
    track_frames = sum(len(sp) for sp in run.speeds)
    err: set[tuple[int, int]] = {(t, tid) for t, tid in res.fp_tracks}
    for t, _gid, old, new in res.switch_events:
        err.add((t, old))
        err.add((t, new))

    def near(ev: tuple[int, int], pool: set[tuple[int, int]]) -> bool:
        return any((ev[0] + d, ev[1]) in pool for d in range(-2, 3))

    explained = sum(1 for e in events if near(e, err))
    sw_hit = sum(
        1
        for t, _g, old, new in res.switch_events
        if near((t, old), events) or near((t, new), events)
    )
    all_tf = {(t, tid) for t, sp in enumerate(run.speeds) for tid in sp}
    base = sum(1 for e in all_tf if near(e, err)) / len(all_tf) if all_tf else float("nan")
    return {
        "ceiling_mps": ceiling,
        "flag_events": len(events),
        "flagged_track_frames": flagged_frames,
        "track_frames": track_frames,
        "flagged_share": flagged_frames / track_frames if track_frames else 0.0,
        "flag_events_near_error": explained,
        "precision_vs_error": explained / len(events) if events else float("nan"),
        "base_rate_error_near_track_frame": base,
        "id_switches": len(res.switch_events),
        "id_switches_recovered": sw_hit,
    }


def evaluate_variants(frames: list[int], mode: str, chosen: dict[str, Any]) -> dict[str, Any]:
    all7 = tuple(range(7))
    c = chosen[mode]
    tuned = replace(DEFAULT, meas_sigma_m=c["meas_sigma_m"], accel_sigma=c["accel_sigma"], min_hits=c["min_hits"])
    nogate = replace(tuned, gate_prob=NO_GATE)
    score = c["score"]
    variants: dict[str, tuple[tuple[int, ...], TrackerConfig]] = {
        "fuse-7": (all7, tuned),
        "fuse-7-live": (all7, tuned),
        "fuse-7-default": (all7, DEFAULT),
        "fuse-7-nogate": (all7, nogate),
        **{f"single-{CAMERAS[i]}": ((i,), tuned) for i in range(7)},
    }
    gts = gt_frames(frames)
    out: dict[str, Any] = {"config": c, "variants": {}}
    runs: dict[str, Run] = {}
    res05: dict[str, MotResult] = {}
    for name, (cams, cfg) in variants.items():
        run = track(frames, mode, score, cams, cfg, live_only=name == "fuse-7-live")
        runs[name] = run
        res05[name] = evaluate(gts, run.pred, 0.5)
        r10 = evaluate(gts, run.pred, 1.0)
        out["variants"][name] = {"at_0.5m": summary(res05[name]), "at_1.0m": summary(r10), "rejected": run.rejected}
    out["bootstrap_mota_at_0.5m"] = mota_boot(res05, "fuse-7")
    out["flags"] = {
        str(v): flag_analysis(runs["fuse-7"], res05["fuse-7"], v) for v in (12.0, 4.0)
    }
    singles = [out["variants"][f"single-{n}"]["at_0.5m"]["mota"] for n in CAMERAS]
    out["single_camera_median_mota"] = float(np.median(singles))
    return out


def cmd_test() -> None:
    dest = RESULTS / "real_test.json"
    if dest.exists():
        sys.exit("results/real_test.json exists: the test frames are one-shot (docs/PROTOCOL_REAL.md)")
    chosen = json.loads((RESULTS / "real_chosen.json").read_text(encoding="utf-8"))
    out = {m: evaluate_variants(TEST, m, chosen) for m in ("O", "D")}
    out["detector_quality_test"] = detector_quality(TEST, chosen["D"]["score"])
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for m in ("O", "D"):
        for name, v in out[m]["variants"].items():
            s = v["at_0.5m"]
            print(f"{m} {name:16s} MOTA {s['mota']:.3f} IDF1 {s['idf1']:.3f} IDSW {int(s['id_switches'])} FP {int(s['fp'])} FN {int(s['fn'])}")


def cmd_dev_report() -> None:
    """Development-frame numbers for the same variants (not one-shot)."""
    chosen = json.loads((RESULTS / "real_chosen.json").read_text(encoding="utf-8"))
    out = {m: evaluate_variants(DEV, m, chosen) for m in ("O", "D")}
    (RESULTS / "real_dev.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for m in ("O", "D"):
        for name, v in out[m]["variants"].items():
            s = v["at_0.5m"]
            print(f"DEV {m} {name:16s} MOTA {s['mota']:.3f} IDF1 {s['idf1']:.3f} IDSW {int(s['id_switches'])} FP {int(s['fp'])} FN {int(s['fn'])}")


if __name__ == "__main__":
    {"tune": cmd_tune, "detector": cmd_detector, "test": cmd_test, "dev": cmd_dev_report}[sys.argv[1]]()
