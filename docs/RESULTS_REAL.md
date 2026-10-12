# Results on real footage (Wildtrack)

Plan: [PROTOCOL_REAL.md](PROTOCOL_REAL.md), pushed before any detector or tracker ran on the data, with two deviations logged before the test frames were run. Tuning used the first 200 annotated frames; the last 200 (4,733 person-frames, 100 s at 2 fps) were run **once** (`results/real_test.json`). Match threshold 0.5 m unless stated. MOTA intervals are a block bootstrap over ten 20-frame blocks.

> **Read this first.** This is one scene, 200 s of footage at 2 frames per second, a generic COCO detector that was never trained on it, and a dataset copy from an unofficial mirror (provenance and checks in the protocol). The tracker is mine, the protocol is mine, and I wrote both before seeing numbers that mostly contradict what I expected. The synthetic results in [RESULTS.md](RESULTS.md) do not describe real behaviour.

## Headline

| Input | Variant | MOTA | IDF1 | ID switches | False pos. | False neg. |
| --- | --- | --- | --- | --- | --- | --- |
| **Oracle boxes** | fuse-7 (primary) | **0.793** [0.740, 0.844] | 0.630 | 145 | 740 | 95 |
| | fuse-7-live (exploratory) | **0.917** [0.888, 0.944] | 0.669 | 158 | 131 | 102 |
| | median of the 7 single cameras | 0.582 (best 0.761, IDIAP2) | | | | |
| **COCO detector** | fuse-7 (primary) | **-0.169** [-0.252, -0.068] | 0.375 | 257 | 4,555 | 723 |
| | fuse-7-live (exploratory) | **0.102** [0.033, 0.168] | 0.414 | 253 | 3,233 | 765 |
| | median of the 7 single cameras | 0.006 (best 0.241, IDIAP1) | | | | |

"fuse-7" is the project's tracker as it was: all seven cameras fused, reporting every confirmed track, including tracks that are coasting after a missed update. "fuse-7-live" is the same tracker reporting only tracks updated in the current frame (Deviation 1).

## What the study found

1. **The synthetic numbers did not transfer.** The simulator gave MOTA 0.93 to 0.96 for two cameras. On real footage with perfect boxes, the project's tracker scores 0.79, and with a real detector it scores below zero.
2. **With perfect boxes, most of the error was an output-policy artefact, not tracking.** The primary variant reports coasting tracks for up to five frames after losing a person, which created 740 false positives against 145 identity switches. Reporting only live tracks cut false positives to 131 and raised MOTA from 0.79 to 0.92 (paired difference +0.09 to +0.16). That fix was found on development frames and logged before the test run, but it is a post-hoc choice, so I label it exploratory.
3. **With a real detector the tracker is dominated by false positives.** 4,555 false positives and 723 misses on 4,733 person-frames. The detector's per-camera precision on test ranged from 0.34 (CVLab4) to 0.92 (IDIAP1) at IoU 0.5 and recall from 0.52 to 0.84. Fusing seven cameras multiplies the number of detections that can start a track, and the tracker starts a track immediately from any unmatched detection, so detector errors add up across cameras rather than cancel. This is my reading of the numbers; I did not test it with a modified spawning rule.
4. **Fusion beat one camera with perfect boxes and lost to the best single camera with the detector.** Oracle: 0.79 against a median of 0.58 (a gain of 0.21, as expected), but the best single camera (IDIAP2) reached 0.76, only 0.03 below. Detector: the median single camera scored 0.006 and the best 0.241, both above the primary fusion's -0.169; even the live variant (0.102) is below the best single camera. The project's central claim, that more cameras help, **holds for clean detections and fails for these real ones.**
5. **Gating helped, clearly for clean detections.** Oracle: removing the gate cost 0.11 MOTA (paired difference -0.16 to -0.06) and raised identity switches from 145 to 393 (IDF1 0.63 to 0.37). Detector: the MOTA difference is not established (-0.05, interval -0.23 to +0.12), because dropping the gate trades false positives for misses, but identity switches rise (257 to 343) and IDF1 falls from 0.375 to 0.233. My expectation that the MOTA change would be under 0.03 was wrong for the oracle input.
6. **The synthetic parameters transferred better than expected.** With oracle boxes the default (`meas_sigma_m` 0.5, `accel_sigma` 1) scored 0.773, within 0.02 of the tuned configuration (0.793, interval of the difference -0.035 to -0.003), with a higher IDF1 (0.750 against 0.630). With the detector the default is worse (-0.302 against -0.169). Tuning on development frames chose `meas_sigma_m` = 0.75 for both inputs, not the 0.3 I expected for the oracle.
7. **The 12 m/s plausibility flag never fired.** Zero events on either input across 5,378 and 8,565 track-frames. At 2 frames per second a pedestrian moves under a metre per frame, and identity errors rarely produce a jump of six metres. An exploratory 4 m/s ceiling flagged 1.3% (oracle) and 3.4% (detector) of track-frames; every flag event was within two frames of an identity switch or a false-positive track, but a large share of all track-frames is near an error anyway (29% oracle, 68% detector, the base rates), and the 4 m/s flags recovered only 50 of 145 identity switches (34%) with the oracle and 58 of 257 (23%) with the detector. They are informative but incomplete, and the 12 m/s default is a ceiling too high to be useful at this frame rate.

## Against the expectations written in advance

| Expectation | Outcome |
| --- | --- |
| O, fuse-7: MOTA 0.80 to 0.95, IDF1 above 0.8 | Slightly below (0.793); **IDF1 0.63, below** |
| D, fuse-7: MOTA 0.45 to 0.75, IDF1 0.5 to 0.8 | **Far off: -0.169, IDF1 0.375** |
| Median single camera at least 0.2 MOTA below fuse-7 (D) | **Wrong:** the median single camera is 0.175 *above* the primary fusion |
| Tuned `meas_sigma_m` at most 0.3 (O), at least 0.75 (D) | **Wrong for O** (0.75); D at the grid's 0.75 |
| Gating changes MOTA by under 0.03 but raises switches | **Wrong on MOTA for O** (0.11); switches rise as predicted |
| Flags: under 1% of track-frames at 12 m/s, recovering under 20% of switches | Under 1% (none); recovery cannot be assessed (0 events). The 4 m/s ceiling flagged *fewer* well-behaved tracks than I expected (all near errors) |
| Camera-count ranking holds but absolute numbers do not | Absolute numbers did not hold; the ranking held only for the oracle |

## What this does and does not show

- **Shown:** the tracker's geometry is sound (calibration check: median 0.07 to 0.13 m, tracker with clean boxes gets 0.92 with live output). The weaknesses are in what the tracker does with imperfect detections: immediate track creation from any unmatched box, coasting-track output, and no identity features. Gating clearly helps.
- **Not shown:** that a better spawning rule (for example requiring support from two cameras or two frames) would fix it. I have not tried it, and the test frames are now used. A follow-up needs a new pre-registered split or a second scene.
- **Not claimed:** that multi-camera fusion is not worth it. This tests this tracker, with this detector, on this scene, at 2 frames per second.

## Process notes and limits

1. Deviation 1: `fuse-7-live` added after seeing development results and before the test run. Deviation 2: development results contradicted the expectations (logged before the test).
2. The tuning grid used the primary output policy; `fuse-7-live` was not tuned separately.
3. MOTP is 0.09 m with oracle boxes and 0.19 m with the detector (fuse-7).
4. The detector was chosen on development frames by tracker MOTA (score threshold 0.9), not by detection quality, and was not fine-tuned. A detector trained on Wildtrack would probably do better; that was not the question.
5. One scene, 2 fps, no appearance features. One author. The dataset copy is an unofficial mirror, unverified against the official host.

## Reproduce

```bash
python scripts/extract_wildtrack.py data/raw/Wildtrack_dataset_full.zip data   # zip with wrong zip64 offsets
python scripts/detect.py                  # one detector pass, cached in data/ (git-ignored)
python scripts/run_real.py tune           # development frames only (about 30 minutes)
python scripts/run_real.py detector && python scripts/run_real.py dev
python scripts/run_real.py test           # one-shot; refuses to overwrite results/real_test.json
```
