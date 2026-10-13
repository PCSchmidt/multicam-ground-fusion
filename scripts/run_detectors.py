"""Detector study (docs/PROTOCOL_DETECTORS.md).

    python scripts/run_detectors.py dev     # choose each detector's score threshold on the development sequences
    python scripts/run_detectors.py test    # ONE-SHOT on the three test sequences; refuses to overwrite
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from mmp2_common import DETECTORS, DEV_STEPS, MMP2, RESULTS, SEQUENCES, STEP
from run_mmp import BLOCKS_PER_SEQUENCE, blocks, cfg_of, detections, iou, pooled

from multicam_fusion.mmptrack import MmpSequence
from multicam_fusion.mot import Frame, MotResult, evaluate
from multicam_fusion.tracker import GroundPlaneTracker

DT = STEP / 15.0
THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
Job = tuple[str, str, str, str, dict[str, Any], tuple[int, ...], int | None]


def run_job(job: Job) -> tuple[str, str, str, str, MotResult, MotResult]:
    tag, mode, det, cname, c, cams, limit = job
    _, name = SEQUENCES[tag]
    seq = MmpSequence(MMP2, tag, name)
    arr = np.load(MMP2 / f"{tag}_{det}.npz") if mode == "D" else None
    frames = seq.frames[::STEP]
    if limit is not None:
        frames = frames[:limit]
    tr = GroundPlaneTracker({f"c{k}": seq.cams[k].pixel_to_world_h() for k in cams}, cfg_of(c, len(seq.cams)))
    pred: list[Frame] = []
    for f in frames:
        ts = tr.step(DT, detections(seq, arr, f, mode, c["score"], cams))
        pred.append({t.track_id: (float(t.position[0]), float(t.position[1])) for t in ts})
    gt: list[Frame] = [dict(seq.ground_truth(f)) for f in frames]
    return tag, mode, det, cname, evaluate(gt, pred, 0.5), evaluate(gt, pred, 1.0)


def cams_of(tag: str) -> tuple[int, ...]:
    _, name = SEQUENCES[tag]
    return tuple(sorted(MmpSequence(MMP2, tag, name).cams))


def run_dev() -> None:
    cfgs = json.loads((RESULTS / "second_chosen.json").read_text(encoding="utf-8"))
    tags = [t for t in SEQUENCES if t.startswith("dev")]
    jobs: list[Job] = [
        (t, "D", d, f"B@{thr}", {**cfgs["B"], "score": thr}, cams_of(t), DEV_STEPS)
        for d in DETECTORS
        for thr in THRESHOLDS
        for t in tags
    ]
    scores: dict[str, dict[str, float]] = {d: {} for d in DETECTORS}
    store: dict[tuple[str, str], list[MotResult]] = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for tag, _mode, det, cname, r05, _r10 in pool.map(run_job, jobs):
            store.setdefault((det, cname), []).append(r05)
    chosen: dict[str, float] = {}
    for d in DETECTORS:
        for thr in THRESHOLDS:
            scores[d][str(thr)] = pooled(store[(d, f"B@{thr}")])["mota"]
        chosen[d] = max(THRESHOLDS, key=lambda t: (round(scores[d][str(t)], 6), t))
        print(d, {k: round(v, 3) for k, v in scores[d].items()}, "chosen", chosen[d], flush=True)
    (RESULTS / "detectors_dev.json").write_text(json.dumps({"chosen": chosen, "dev_mota": scores}, indent=1), encoding="utf-8")


def quality(tag: str, det: str, score: float) -> dict[str, dict[str, float]]:
    _, name = SEQUENCES[tag]
    seq = MmpSequence(MMP2, tag, name)
    arr = np.load(MMP2 / f"{tag}_{det}.npz")
    out: dict[str, dict[str, float]] = {}
    for cam in sorted(seq.cams):
        tp = fp = fn = 0
        for f in seq.frames[::STEP]:
            gts = list(seq.boxes(f, cam).values())
            dets = [d.bbox for d in detections(seq, arr, f, "D", score, (cam,))]
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
        out[f"c{cam}"] = {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}
    return out


def run_test() -> None:
    dest = RESULTS / "detectors_test.json"
    if dest.exists():
        sys.exit("results/detectors_test.json exists: the test sequences are one-shot (docs/PROTOCOL_DETECTORS.md)")
    cfgs = json.loads((RESULTS / "second_chosen.json").read_text(encoding="utf-8"))
    chosen = json.loads((RESULTS / "detectors_dev.json").read_text(encoding="utf-8"))["chosen"]
    tags = [t for t in SEQUENCES if t.startswith("test")]
    jobs: list[Job] = []
    for t in tags:
        cams = cams_of(t)
        for d in DETECTORS:
            for cn in ("A", "B", "C"):
                jobs.append((t, "D", d, f"{cn}-fuse", {**cfgs[cn], "score": chosen[d]}, cams, None))
            for k in cams:
                jobs.append((t, "D", d, f"B-single-c{k}", {**cfgs["B"], "score": chosen[d]}, (k,), None))
        for cn in ("A", "B"):
            jobs.append((t, "O", "o", f"{cn}-fuse", {**cfgs[cn], "score": 0.0}, cams, None))
    store: dict[tuple[str, str, str, str], tuple[MotResult, MotResult]] = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for tag, mode, det, cname, r05, r10 in pool.map(run_job, jobs):
            store[(tag, mode, det, cname)] = (r05, r10)
            print(tag, mode, det, f"{cname:14s} MOTA {r05.mota:.3f} IDF1 {r05.idf1:.3f} FP {r05.fp} FN {r05.fn}", flush=True)
    out: dict[str, Any] = {"chosen_thresholds": chosen, "per_sequence": {}, "pooled": {}, "bootstrap": {}, "quality": {}}
    for (tag, mode, det, cname), (r05, r10) in store.items():
        out["per_sequence"][f"{tag}|{mode}|{det}|{cname}"] = {"at_0.5m": pooled([r05]), "at_1.0m": pooled([r10])}
    groups = sorted({(m, d, c) for (_, m, d, c) in store if "single" not in c})
    for m, d, c in groups:
        out["pooled"][f"{m}|{d}|{c}"] = {
            "at_0.5m": pooled([store[(t, m, d, c)][0] for t in tags]),
            "at_1.0m": pooled([store[(t, m, d, c)][1] for t in tags]),
        }
    rng = np.random.default_rng(0)
    nb = BLOCKS_PER_SEQUENCE * len(tags)
    idx = rng.integers(0, nb, (2000, nb))
    draws = {}
    for m, d, c in groups:
        b = np.concatenate([blocks(store[(t, m, d, c)][0]) for t in tags])
        draws[f"{m}|{d}|{c}"] = 1.0 - (b[idx][..., 1] + b[idx][..., 2] + b[idx][..., 3]).sum(-1) / b[idx][..., 0].sum(-1)
        out["bootstrap"][f"{m}|{d}|{c}"] = {"ci95": [float(np.percentile(draws[f'{m}|{d}|{c}'], q)) for q in (2.5, 97.5)]}
    pairs = [
        ("D|d2|B-fuse", "D|d0|B-fuse"), ("D|d1|B-fuse", "D|d0|B-fuse"), ("D|d0|B-fuse", "D|d0|A-fuse"),
        ("D|d2|C-fuse", "D|d2|B-fuse"), ("O|o|B-fuse", "D|d2|B-fuse"), ("D|d2|B-fuse", "D|d1|B-fuse"),
    ]  # fmt: skip
    for a, b in pairs:
        diff = draws[a] - draws[b]
        out["bootstrap"][f"{a} minus {b}"] = {
            "point": out["pooled"][a]["at_0.5m"]["mota"] - out["pooled"][b]["at_0.5m"]["mota"],
            "ci95": [float(np.percentile(diff, q)) for q in (2.5, 97.5)],
        }
    for t in tags:
        for d in DETECTORS:
            out["quality"][f"{t}|{d}"] = quality(t, d, chosen[d])
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("pooled MOTA at 0.5 m [95%] / at 1.0 m / IDF1")
    for k, v in out["pooled"].items():
        print(f"  {k:16s} {v['at_0.5m']['mota']:.3f} {[round(x, 3) for x in out['bootstrap'][k]['ci95']]} / {v['at_1.0m']['mota']:.3f} / {v['at_0.5m']['idf1']:.3f}  FP {int(v['at_0.5m']['fp'])} FN {int(v['at_0.5m']['fn'])}")
    for k, v in out["bootstrap"].items():
        if "minus" in k:
            print("  ", k, round(v["point"], 3), [round(x, 3) for x in v["ci95"]])


if __name__ == "__main__":
    {"dev": run_dev, "test": run_test}[sys.argv[1]]()
