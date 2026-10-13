# Results: a stricter track-creation rule, tested on three MMPTrack sequences

Plan: [PROTOCOL_SECOND_SCENE.md](PROTOCOL_SECOND_SCENE.md), pushed before the rule was tuned and before any run on the new scenes, with one deviation logged before those runs. The configs were designed on Wildtrack development frames only and frozen; each of the three sequences (`lobby_0` with 4 cameras, `office_0` with 5, `retail_0` with 6; 41,632 person-frames at about 5 frames per second) was run **once** (`results/second_test.json`). Match threshold 0.5 m unless stated; intervals are a block bootstrap (60 blocks), paired for differences.

> **Read this first.** Three sequences of one dataset, from an unofficial mirror, with calibration conventions I inferred (not documented), 640 x 360 images with unmodelled lens distortion, and a tracker I wrote. The settings are the ones designed on a different scene (Wildtrack), nothing was tuned here. The headline is that **the Wildtrack results were much harsher than this scene, and the tracker's problems are mostly about frame rate, crowding and detection, not geometry.**

## Headline (pooled over the three sequences)

| Input | Config | MOTA [95%] | IDF1 | ID switches | False pos. | False neg. | MOTA at 1.0 m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Dataset's own boxes** | A (original tracker) | 0.997 [0.993, 1.000] | 0.966 | 7 | 103 | 12 | 0.997 |
| | B (confirm 3 frames, live-only) | 0.996 [0.992, 0.999] | 0.966 | 7 | 84 | 60 | 0.997 |
| | C (B + camera consensus, exploratory) | 0.998 [0.993, 1.000] | 0.966 | 7 | 84 | 12 | 0.998 |
| **COCO detector** | A | 0.568 [0.515, 0.624] | 0.394 | 190 | 10,536 | 7,274 | 0.835 |
| | B | 0.589 [0.538, 0.642] | 0.399 | 209 | 9,597 | 7,309 | 0.853 |
| | C (exploratory) | 0.603 [0.555, 0.654] | 0.403 | 199 | 8,901 | 7,445 | 0.865 |

Paired differences with the detector: **B minus A +0.021 [+0.015, +0.028]**; C minus B +0.014 [+0.008, +0.019]; C minus A +0.035 [+0.025, +0.045]. With the dataset's boxes: B minus A -0.001 [-0.002, 0.000]; C minus B +0.001 [0.000, +0.002].

## Per sequence, detector input (MOTA)

| Sequence (cameras) | A fuse | B fuse | C fuse | Median single camera (A / B) | Best single camera (A / B) |
| --- | --- | --- | --- | --- | --- |
| lobby (4) | 0.448 | 0.482 | 0.512 | 0.317 / 0.359 | 0.535 / **0.546** |
| office (5) | 0.806 | 0.814 | 0.813 | 0.327 / 0.331 | 0.536 / 0.513 |
| retail (6) | 0.428 | 0.449 | 0.460 | 0.187 / 0.203 | 0.370 / 0.381 |

Detector precision and recall at IoU 0.5 ranged per camera from 0.58 to 0.93 and from 0.26 to 0.76 (lowest recall in the retail scene).

## What the study found

1. **The tracker is nearly perfect with clean boxes on this data.** MOTA 0.996 to 0.998 with 7 identity switches across 41,632 person-frames. Wildtrack's oracle MOTA of 0.79 (0.92 live-only) was not a limit of the geometry or the association; it came from a crowded scene (about 24 people per frame) at 2 frames per second, where a person moves 0.7 m between frames and the 0.5 m threshold is tight. At about 5 frames per second with about 7 people per frame, the same code makes almost no mistakes.
2. **The stricter rule barely helps, as the development stage already hinted.** B beats A by 0.021 with the detector and not at all with clean boxes, far short of the +0.3 I predicted. On Wildtrack the confirmation rule was a negligible part of the gain, and most of it came from live-only output; here live-only output gains little because coasting tracks are rare at 5 frames per second. Camera consensus (C) adds another 0.014: small but its interval is above zero in all pooled comparisons.
3. **With the detector the pipeline is far better than on Wildtrack but still false-positive dominated.** MOTA 0.57 to 0.60 against -0.17 on Wildtrack. 8,901 to 10,536 false positives and about 7,300 misses over 41,632 person-frames. Much of that is localisation, not hallucination: at a 1.0 m match threshold MOTA rises to 0.84 to 0.87, because a detection placed 0.5 to 1 m from the person's feet counts as a false positive and a miss at 0.5 m. Bottom-centre projection of a box that cuts off the feet is the weak point, and the camera geometry does not fix it.
4. **Fusion helps clearly against the median single camera and mostly against the best one.** Fused MOTA beats the median single camera in every sequence by 0.12 to 0.48. It beats the best single camera in `office_0` (0.81 against 0.51 to 0.54) and `retail_0` (0.45 against 0.37 to 0.38), but **not in `lobby_0`** (B fuse 0.482, C fuse 0.512, best single camera 0.546): in that 4-camera scene the best camera is almost as good as the fusion. This differs from Wildtrack (fusion lost to the best camera) and from my expectation (the best single camera beaten in at most one of three).
5. **IDF1 stays low with the detector (0.39 to 0.40)** and is high with clean boxes (0.97): identity is a detection problem, not a tracking one, in these sequences.

## Against the expectations written in advance

| Expectation | Outcome |
| --- | --- |
| A with the detector: MOTA below 0.2, down to -1.0 | **Wrong:** 0.568 |
| B at least 0.3 above A, interval above zero | **Wrong:** +0.021 (interval above zero, but ten times smaller) |
| B fuse-all beats the median single camera by at least 0.1 in every sequence | **Yes** (+0.12 to +0.48) |
| B fuse-all beats the best single camera in at most one of three | **Wrong:** in two of three |
| Oracle B fuse-all at least 0.75 | Yes (0.996) |
| Oracle A lower than B by at least 0.05 | **Wrong:** A is 0.001 higher |
| IDF1 with the detector below 0.6 | Yes (0.39 to 0.40) |
| The Wildtrack-chosen score threshold is not optimal here | Not assessed (no tuning on these scenes) |

## What this does and does not show

- **Shown:** the tracker's geometry and association are sound on a second, easier scene; the Wildtrack failure was driven by crowding, a low frame rate and detector localisation error; confirmation rules and camera consensus are small effects on top of live-only output; fusion beats a typical single camera by a wide margin and the best single camera in two of three sequences.
- **Not shown:** that the pattern holds in dense scenes at a high frame rate, or in Wildtrack at 5 frames per second (not available); that the detector is the right one (a COCO model at 640 x 360 is weak and no other detector was tried); calibration correctness beyond the inferred conventions (median ground error of the dataset's own boxes was 0.16 to 0.39 m).
- **Not claimed:** that confirmation rules help; the evidence says they barely do.

## Process notes and limits

1. Deviation 1: the confirmation rule's effect on Wildtrack was negligible, and exploratory config C was added before any new-scene run, with its camera count scaled to each scene (`ceil(4/7 x cameras)`, i.e. 3, 3 and 4).
2. Every sequence was subsampled by taking every third frame (my choice).
3. Dataset provenance, conventions and limits: see the protocol. The data stay local; the repository holds code and derived numbers only.

## Reproduce

```bash
# data/mmp/<scene>/{calibrations.json,labels.zip,topdown.zip,images.zip} from the mirror (validation split, lobby_0, office_0, retail_0)
python scripts/run_second.py tune && python scripts/run_second.py tune_c     # Wildtrack development frames only
python scripts/detect_mmp.py lobby office retail                              # one detector pass, cached
python scripts/run_mmp.py test                                                # one-shot; refuses to overwrite
```
