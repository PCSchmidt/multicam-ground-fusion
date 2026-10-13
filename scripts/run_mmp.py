"""One-shot evaluation on three MMPTrack sequences (docs/PROTOCOL_SECOND_SCENE.md).

    python scripts/run_mmp.py test     # needs results/second_chosen.json and data/mmp/<scene>_det.npz; refuses to overwrite

Configs A, B and C are frozen in results/second_chosen.json (designed on Wildtrack development frames only).
"""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from multicam_fusion.geometry import apply_homography, bottom_center
from multicam_fusion.mmptrack import MmpSequence
from multicam_fusion.mot import Frame, MotResult, evaluate
from multicam_fusion.tracker import Detection, GroundPlaneTracker, TrackerConfig

ROOT = Path(__file__).resolve().parent.parent
MMP = ROOT / "data" / "mmp"
RESULTS = ROOT / "results"
SCENES = ("lobby", "office", "retail")
STEP = 3
DT = STEP / 15.0
BLOCKS_PER_SEQUENCE = 20


def cfg_of(c: dict[str, Any], n_cameras: int) -> TrackerConfig:
    cc = int(c.get("confirm_cameras", 1))
    if "confirm_cameras" in c and c["confirm_cameras"] > 1:  # config C scales with the number of cameras
        cc = math.ceil(c["confirm_cameras"] / 7.0 * n_cameras)
    return TrackerConfig(
        meas_sigma_m=c["meas_sigma_m"],
        accel_sigma=c["accel_sigma"],
        min_hits=c["min_hits"],
        confirm_frames=c.get("confirm_frames", 1),
        report_coasting=c.get("report_coasting", True),
        confirm_cameras=cc,
    )


def detections(seq: MmpSequence, det: Any, frame: int, mode: str, score: float, cams: tuple[int, ...]) -> list[Detection]:
    out: list[Detection] = []
    for cam in cams:
        if mode == "O":
            boxes = list(seq.boxes(frame, cam).values())
        else:
            boxes = [tuple(map(float, r[:4])) for r in det[f"c{cam}_f{frame}"] if r[4] >= score]
        h = seq.cams[cam].pixel_to_world_h()
        for b in boxes:
            xy = apply_homography(h, np.asarray(bottom_center((b[0], b[1], b[2], b[3])))[None])[0]
            if seq.space.contains(xy):
                out.append(Detection(f"c{cam}", (b[0], b[1], b[2], b[3])))
    return out


def run_one(job: tuple[str, str, str, dict[str, Any], tuple[int, ...]]) -> tuple[str, str, str, MotResult, MotResult]:
    scene, mode, name, c, cams = job
    seq = MmpSequence(MMP, scene, f"{scene}_0")
    det = np.load(MMP / f"{scene}_det.npz") if mode == "D" else None
    frames = seq.frames[::STEP]
    tr = GroundPlaneTracker({f"c{k}": seq.cams[k].pixel_to_world_h() for k in cams}, cfg_of(c, len(seq.cams)))
    pred: list[Frame] = []
    for f in frames:
        ts = tr.step(DT, detections(seq, det, f, mode, c["score"], cams))
        pred.append({t.track_id: (float(t.position[0]), float(t.position[1])) for t in ts})
    gt: list[Frame] = [{k: v for k, v in seq.ground_truth(f).items()} for f in frames]
    return scene, mode, name, evaluate(gt, pred, 0.5), evaluate(gt, pred, 1.0)


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def detector_quality(scene: str, score: float) -> dict[str, dict[str, float]]:
    seq = MmpSequence(MMP, scene, f"{scene}_0")
    det = np.load(MMP / f"{scene}_det.npz")
    out: dict[str, dict[str, float]] = {}
    for cam in sorted(seq.cams):
        tp = fp = fn = 0
        for f in seq.frames[::STEP]:
            gts = list(seq.boxes(f, cam).values())
            dets = [d.bbox for d in detections(seq, det, f, "D", score, (cam,))]
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
        out[f"c{cam}"] = {"precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0, "tp": tp, "fp": fp, "fn": fn}
    return out


def blocks(res: MotResult) -> np.ndarray:
    a = np.array(res.frame_stats, dtype=float)
    edges = np.linspace(0, len(a), BLOCKS_PER_SEQUENCE + 1).astype(int)
    return np.array([a[edges[i] : edges[i + 1]].sum(axis=0) for i in range(BLOCKS_PER_SEQUENCE)])


def pooled(results: list[MotResult]) -> dict[str, float]:
    n_gt = sum(r.n_gt for r in results)
    n_pred = sum(r.n_pred for r in results)
    fn, fp, sw = (sum(getattr(r, k) for r in results) for k in ("fn", "fp", "id_switches"))
    tp = sum(r.tp for r in results)
    dsum = sum(r.motp_m * r.tp for r in results if r.tp)
    idtp = sum(r.idtp for r in results)
    return {
        "mota": 1.0 - (fn + fp + sw) / n_gt, "motp_m": dsum / tp if tp else float("nan"),
        "idf1": 2 * idtp / (n_gt + n_pred), "id_switches": sw, "fp": fp, "fn": fn, "n_gt": n_gt, "n_pred": n_pred,
    }  # fmt: skip


def cmd_test() -> None:
    dest = RESULTS / "second_test.json"
    if dest.exists():
        sys.exit("results/second_test.json exists: the new scenes are one-shot (docs/PROTOCOL_SECOND_SCENE.md)")
    chosen = json.loads((RESULTS / "second_chosen.json").read_text(encoding="utf-8"))
    A, B, C = chosen["A"], chosen["B"], chosen["C"]
    jobs: list[tuple[str, str, str, dict[str, Any], tuple[int, ...]]] = []
    for sc in SCENES:
        cams = tuple(sorted(MmpSequence(MMP, sc, f"{sc}_0").cams))
        for mode in ("D", "O"):
            jobs += [(sc, mode, "A-fuse", A, cams), (sc, mode, "B-fuse", B, cams), (sc, mode, "C-fuse", C, cams)]
        for k in cams:
            jobs += [(sc, "D", f"A-single-c{k}", A, (k,)), (sc, "D", f"B-single-c{k}", B, (k,))]
    store: dict[tuple[str, str, str], tuple[MotResult, MotResult]] = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for scene, mode, name, r05, r10 in pool.map(run_one, jobs):
            store[(scene, mode, name)] = (r05, r10)
            print(scene, mode, f"{name:18s} MOTA {r05.mota:.3f} IDF1 {r05.idf1:.3f} IDSW {r05.id_switches} FP {r05.fp} FN {r05.fn}", flush=True)
    out: dict[str, Any] = {"configs": chosen, "per_sequence": {}, "pooled": {}, "bootstrap": {}, "detector_quality": {}}
    for (scene, mode, name), (r05, r10) in store.items():
        out["per_sequence"][f"{scene}|{mode}|{name}"] = {"at_0.5m": {k: v for k, v in pooled([r05]).items()}, "at_1.0m": pooled([r10])}
    groups = sorted({(mode, name) for _, mode, name in store})
    for mode, name in groups:
        if "single" in name:
            continue
        out["pooled"][f"{mode}|{name}"] = {
            "at_0.5m": pooled([store[(s, mode, name)][0] for s in SCENES]),
            "at_1.0m": pooled([store[(s, mode, name)][1] for s in SCENES]),
        }
    rng = np.random.default_rng(0)
    nb = BLOCKS_PER_SEQUENCE * len(SCENES)
    idx = rng.integers(0, nb, (2000, nb))
    bl = {f"{m}|{n}": np.concatenate([blocks(store[(s, m, n)][0]) for s in SCENES]) for m, n in groups if "single" not in n}
    draws = {k: 1.0 - (b[idx][..., 1] + b[idx][..., 2] + b[idx][..., 3]).sum(-1) / b[idx][..., 0].sum(-1) for k, b in bl.items()}
    for k, d in draws.items():
        out["bootstrap"][k] = {"ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    for a, b in (("D|B-fuse", "D|A-fuse"), ("D|C-fuse", "D|B-fuse"), ("D|C-fuse", "D|A-fuse"), ("O|B-fuse", "O|A-fuse"), ("O|C-fuse", "O|B-fuse")):
        diff = draws[a] - draws[b]
        out["bootstrap"][f"{a} minus {b}"] = {"point": out["pooled"][a]["at_0.5m"]["mota"] - out["pooled"][b]["at_0.5m"]["mota"], "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
    for sc in SCENES:
        out["detector_quality"][sc] = {"A": detector_quality(sc, A["score"]), "B": detector_quality(sc, B["score"])}
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("pooled:")
    for k, v in out["pooled"].items():
        print(f"  {k:16s} MOTA {v['at_0.5m']['mota']:.3f} {[round(x, 3) for x in out['bootstrap'][k]['ci95']]} IDF1 {v['at_0.5m']['idf1']:.3f} IDSW {int(v['at_0.5m']['id_switches'])} FP {int(v['at_0.5m']['fp'])} FN {int(v['at_0.5m']['fn'])}")
    for k, v in out["bootstrap"].items():
        if "minus" in k:
            print("  ", k, round(v["point"], 3), [round(x, 3) for x in v["ci95"]])


if __name__ == "__main__":
    {"test": cmd_test}[sys.argv[1]]()
