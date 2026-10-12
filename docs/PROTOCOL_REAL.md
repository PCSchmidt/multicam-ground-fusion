# Protocol: does ground-plane fusion work on real multi-camera footage? (written before any detector or tracker was run on Wildtrack)

## Why

Everything in [RESULTS.md](RESULTS.md) is synthetic: a simulator I wrote, with noise I chose, scored by a metric I wrote. Its own "Development finding" says the main risk for real data is the noise model. This study runs the same tracker on a public real dataset, with a real detector, and reports whatever happens, including whether the project's own claims (fusion beats one camera; gating helps; plausibility flags are useful) survive.

## Data (provenance stated, because it is not the official host)

- **Wildtrack** (Chavdarova et al., CVPR 2018): seven static, synchronised, calibrated HD cameras over one public square, 400 annotated frames at 2 fps (file numbers 0 to 1995 in steps of 5), ground-truth person positions on a 480 x 1440 ground grid (2.5 cm cells) with person identities and per-camera boxes.
- **Where I got it.** The official EPFL host could not be reached from this machine (DNS failure). I downloaded `Wildtrack_dataset_full.zip` (6,807,496,358 bytes, SHA-256 `7f630f3adf305d05362379f2603c67b0c6a0b173dcec508020e98a7bde4e4492`) from a Hugging Face user's mirror (`sein123/wildtrack`). **I cannot verify it against the official copy.** The archive's central directory has wrong zip64 offsets, so standard unzip tools fail; `scripts/extract_wildtrack.py` recovers every entry and checks size and CRC-32 against the directory (3,231 of 3,231 files verified).
- **Terms.** The dataset README asks users to cite the paper; I found no licence text, so I treat it as research use only. **Nothing from the dataset is committed**: images, annotations and detections stay in `data/` (git-ignored). The repository holds code, the SHA-256 above, and derived numbers.
- **Calibration check done before this protocol (development frames only, ground-truth boxes, no tracker or detector involved).** Projecting each ground-truth box's bottom-centre to the ground plane with the supplied calibration gives a median error of 0.07 to 0.13 m against the annotated position, per camera (90th percentile 0.10 to 0.28 m); world units are metres once the calibration translations (centimetres) are divided by 100.

## Split

Annotated frames in time order, indices 0 to 199 are **development** and 200 to 399 are **test**. Tuning uses only development frames. Each final configuration is run on the test frames **once**; the runner writes a results file and refuses to overwrite it.

## Inputs to the tracker

1. **O, oracle boxes.** The dataset's own per-camera boxes as detections (no identities, no noise added). Isolates calibration, association and filtering from detection.
2. **D, a real detector.** torchvision `fasterrcnn_resnet50_fpn_v2` with COCO weights (`FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`, box mAP 46.7 on COCO val2017), class `person`, full-resolution 1920 x 1080 images, run once over all 2,800 images and cached. It is not trained or fine-tuned on Wildtrack. Detections are kept if their score is at least a threshold chosen on development frames from {0.5, 0.7, 0.9}, and only if the bottom-centre projects inside the annotated area (x in [-3, 9) m, y in [-9, 27) m). This region filter uses geometry only, never ground truth, and is applied to every variant, because people outside the annotated area are not labelled.

## Tracker settings and tuning (development frames only)

The tracker in `src/multicam_fusion/tracker.py` is used unchanged. Grid, with all seven cameras fused and ground-truth matching threshold 0.5 m, objective MOTA on development frames:

- `meas_sigma_m` in {0.2, 0.3, 0.5, 0.75, 1.0, 1.5}
- `accel_sigma` in {0.5, 1, 2, 3}
- `min_hits` in {2, 3}
- detector score threshold in {0.5, 0.7, 0.9} (D only)

The synthetic default (`meas_sigma_m = 0.5`, `accel_sigma = 1`, `min_hits = 3`) is also evaluated on test so the cost of not re-tuning is visible. Frame spacing is `dt = 0.5` s.

## Variants evaluated once on the test frames (for O and for D)

| Variant | What it tests |
| --- | --- |
| **fuse-7**: all cameras, gated (tuned) | the main claim |
| **fuse-7-default**: all cameras, gated, synthetic default parameters | transfer of the synthetic tuning |
| **fuse-7-nogate**: all cameras, same parameters, gate effectively off (`gate_prob = 1 - 1e-12`) | does gating help |
| **single-camera**: each of the seven cameras alone, gated (tuned) | does fusion beat one camera |

## Metrics

- **Ground-plane CLEAR-MOT and IDF1** (`src/multicam_fusion/mot.py`, sticky matching) at a match threshold of **0.5 m** (the Wildtrack convention) and, as a robustness row, 1.0 m: MOTA, MOTP (m), identity switches, false positives, false negatives, IDF1.
- **Intervals:** a block bootstrap over the test timeline (10 blocks of 20 frames, resampled with replacement, 2,000 draws) on the per-block error counts for MOTA; paired differences between variants use the same resampled blocks. IDF1 is reported without an interval.
- **Detector quality:** per-camera precision and recall of person detections against the dataset's boxes at IoU >= 0.5 (inside the annotated area), development and test.
- **Plausibility flags** (`IMPLAUSIBLE_SPEED`, ceiling 12 m/s by default): share of track-frames flagged; and, for flagged tracks, whether an identity switch or a false-positive track starts within +-2 frames; recall of identity switches by flags. Also an exploratory ceiling of 4 m/s, labelled as such.
- **Refused associations** (`tracker.rejected`) per run.

## Pre-stated expectations (written so they can be wrong)

- **O, fuse-7, tuned:** MOTA 0.80 to 0.95 on test, IDF1 above 0.8. Most remaining error is identity switches and duplicate tracks, not misses.
- **D, fuse-7, tuned:** MOTA 0.45 to 0.75 and IDF1 0.5 to 0.8, far below the synthetic 0.93 to 0.96, from missed occluded people, boxes cut by occlusion (bottom-centre bias), and unlabelled people inside the area.
- **Fusion versus one camera:** the median single camera is at least 0.2 MOTA below fuse-7 for D.
- **Tuned `meas_sigma_m`:** at most 0.3 for O, at least 0.75 for D. The default 0.5 will be too small for D.
- **Gating:** removing the gate changes MOTA by less than 0.03 but raises identity switches and false-positive tracks; i.e. gating matters for track quality more than for MOTA. If MOTA falls by more than 0.03 without the gate I will say the claim held; if identity switches do not rise I will say it did not.
- **Flags:** fewer than 1% of track-frames flagged at 12 m/s; they recover under 20% of identity switches (most switches are not teleports). A 4 m/s ceiling flags much more and mostly well-behaved pedestrians.
- **Synthetic results do not transfer:** the ranking of cameras counts (more cameras better) will, the absolute numbers will not.

## Limits, stated in advance

One scene, 200 s, 2 fps (large motion between frames), seven cameras, pedestrians only. Ground truth is a grid cell per person, so MOTP has a floor of about 0.05 m. The detector is a generic COCO model, not tuned to this scene. No appearance features, so identity relies on position alone. The tracker's constant-velocity model and Hungarian association are the project's own, so this tests that implementation, not "multi-camera tracking" in general. One author, who wrote the tracker and the protocol. The dataset copy is an unofficial mirror. The oracle input is optimistic by construction.

## Deviations

(none yet)
