# Results: is the detector the bottleneck? Three detectors on fresh MMPTrack sequences

Plan: [PROTOCOL_DETECTORS.md](PROTOCOL_DETECTORS.md), pushed before any new detector was run or trained. Thresholds were frozen on development sequences (`results/detectors_dev.json`: D0 0.9, D1 0.7, D2 0.9). The three test sequences (`lobby_1`, `office_1`, `retail_1`, validation split, session 64pm; 4, 5 and 6 cameras) were run **once** (`results/detectors_test.json`). Match threshold 0.5 m unless stated; intervals are a block bootstrap (60 blocks, 2,000 resamples), paired for differences.

> **Read this first.** Three scenes of one dataset from an unofficial mirror, with calibration conventions I inferred. D2 was fine-tuned on an earlier session of the *same rooms and cameras*, so its result measures adapting a detector to a site, not generalising to a new one. Two of the three thresholds sit at the edge of the grid (0.9), so they may be too low. One author, one fine-tuning recipe, untuned.

## Headline (pooled over the three test sequences, MOTA at 0.5 m [95%])

| Input | A (original) | B (confirm 3 frames, live-only) | C (B + consensus, exploratory) | B at 1.0 m | B IDF1 | B false pos. / false neg. |
| --- | --- | --- | --- | --- | --- | --- |
| **D0** COCO Faster R-CNN | 0.485 [0.419, 0.550] | 0.524 [0.465, 0.582] | 0.568 [0.515, 0.619] | 0.801 | 0.403 | 11,606 / 7,777 |
| **D1** COCO RetinaNet | 0.575 [0.534, 0.614] | 0.594 [0.557, 0.631] | 0.593 [0.557, 0.630] | 0.819 | 0.401 | 8,071 / 8,309 |
| **D2** Faster R-CNN fine-tuned on the site | 0.734 [0.702, 0.761] | 0.759 [0.730, 0.784] | 0.780 [0.751, 0.806] | 0.786 | 0.452 | 8,861 / 786 |
| Dataset's own boxes (O) | 0.987 [0.979, 0.993] | 0.986 [0.979, 0.992] | not run | 0.991 | 0.858 | 371 / 177 |

Paired differences (config B unless stated): D2 minus D0 **+0.235 [+0.164, +0.304]**; D1 minus D0 +0.070 [+0.027, +0.114]; D2 minus D1 +0.165 [+0.115, +0.212]; D0 B minus D0 A +0.039 [+0.029, +0.049]; D2 C minus D2 B +0.021 [+0.013, +0.030]; O minus D2 +0.227 [+0.204, +0.253].

## Per sequence (config B, MOTA at 0.5 m)

| Sequence (cameras) | D0 fused / best single | D1 fused / best single | D2 fused / best single | Own boxes |
| --- | --- | --- | --- | --- |
| lobby (4) | 0.384 / 0.439 | 0.539 / 0.433 | 0.824 / **0.971** | 0.999 |
| office (5) | 0.764 / 0.506 | 0.720 / 0.409 | 0.725 / **0.755** | 0.970 |
| retail (6) | 0.469 / 0.382 | 0.531 / 0.345 | 0.676 / 0.573 | 0.983 |

Per-camera detector precision and recall at IoU 0.5 (D2): lobby 0.91 to 0.99 and 0.91 to 0.99; office 0.69 to 0.98 and 0.57 to 0.97; retail 0.68 to 0.86 and 0.36 to 0.69. For D0 recall was 0.15 to 0.67.

## Against the hypotheses written in advance

| | Prediction | Outcome |
| --- | --- | --- |
| E1 | D2 beats D0 by at least 0.20, interval above zero | **Met**: +0.235 [+0.164, +0.304] |
| E2 | D1 within +-0.10 of D0 | **Met**: +0.070 (but its interval, +0.027 to +0.114, is above zero and reaches the edge of the band) |
| E3 | With D2, MOTA at 1.0 m minus at 0.5 m below 0.05 | **Met**: 0.786 - 0.759 = 0.027 (D0: 0.277) |
| E4 | D0 config A within +-0.15 of 0.568, and B minus A below 0.05 | **Met**: A = 0.485 (-0.083), B minus A = +0.039 |
| E5 | With D2, fused B beats the best single camera in all three sequences | **Not met**: beats it in retail only; loses in lobby (0.824 against 0.971) and office (0.725 against 0.755) |
| E6 | With D2, C minus B below 0.01 | **Not met**: +0.021 [+0.013, +0.030]; consensus still removes false positives |
| E7 | D2 stays at least 0.05 below the dataset's boxes | **Met**: 0.227 below |

Five of seven met, two not.

## What the study found

1. **The detector is most of the gap, and site adaptation closes about half of the distance to the dataset's own boxes (0.235 of 0.46).** Fine-tuning on one earlier session of the same rooms lifts pooled MOTA from 0.52 to 0.76 and cuts false negatives from 7,777 to 786. A second COCO detector from another architecture family gains +0.07 over the first, so a detector change helps, but adaptation helps more than twice as much.
2. **Localisation was a large part of the old gap and the fine-tuned detector mostly removes it.** With D0 the 1.0 m figure was 0.28 above the 0.5 m figure; with D2 it is 0.03 (E3). At 1.0 m D2 is slightly *worse* than D0 (0.786 against 0.801), because D2 has more false positives left once boxes are placed well.
3. **What remains is false positives and identity, not misses.** D2 B has 8,861 false positives and 786 misses, and its IDF1 is 0.45 against 0.86 with the dataset's boxes. Per-camera precision is mostly 0.8 to 0.99, so I expect most of these are duplicate or ghost ground points from cross-camera disagreement, not hallucinated people; I did not diagnose this. Consensus (C) removes about 900 of them (+0.021).
4. **Fusion does not always beat the best camera once the detector is good.** In lobby the best single D2 camera scores 0.971 against 0.824 fused: that camera sees almost everyone with 0.99 precision and recall, and merging in the other three adds errors. With a weak detector (D0, D1) fusion beat the best single camera in five of six cases, the exception being D0 in lobby. So the value of fusion shrinks as the per-camera detector improves, a pattern not predicted in advance.
5. **The tracker rules matter little again.** B beats A by 0.039 with D0, 0.019 with D1 and 0.025 with D2.

## Limits and caveats

- **Site adaptation, not generalisation** (stated in the protocol). D2 would be expected to do worse on a room it has not seen; that is untested.
- **Thresholds at the grid edge.** D0 and D2 both chose 0.9, the top of {0.3, 0.5, 0.7, 0.9}; higher thresholds were not tried. D2's large false-positive count may partly reflect that.
- **Replication is approximate.** Earlier D0 config A scored 0.568 on other sequences; here 0.485 (E4 held, with a margin of 0.07).
- **Development and fine-tuning used the same sequence names** (`office_0` appears in both rows of the protocol) but in different sessions (63am against 64am) and splits, as the protocol specifies. I did not check frame-level overlap beyond that.
- **Camera and scene effects are large.** Per-sequence results differ by up to 0.4 MOTA, and three sequences is a small sample; the pooled intervals understate that because blocks within a sequence are correlated.
- Config C is exploratory (as in the second-scene study); its numbers are not part of the pre-registered claims.
- The dataset copy is from an unofficial mirror; nothing from it is committed. Fine-tuned weights are not committed either.
