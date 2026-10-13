# Protocol: a stricter track-creation rule, designed on Wildtrack and tested on three scenes it has never seen (written before the rule was tuned and before any detector or tracker ran on the new scenes)

Follows [PROTOCOL_REAL.md](PROTOCOL_REAL.md) and [RESULTS_REAL.md](RESULTS_REAL.md). That study found that the tracker, fed by an off-the-shelf detector, scores MOTA -0.17 on Wildtrack, with 4,555 false positives on 4,733 person-frames, and that fusion lost to the best single camera. My reading was that two design choices cause it: a track is created from any single unmatched box, and tracks that lost their detections are still reported. That reading was never tested. The Wildtrack test frames are now used up, so this study designs the fix on Wildtrack **development** frames only and measures it, once, on different scenes.

## The change (two options on `TrackerConfig`, defaults reproduce the original tracker)

- `confirm_frames`: a track is reported only after it has been updated in at least this many distinct time steps (several cameras seeing a person in one step count once).
- `report_coasting`: if false, only tracks updated in the current step are reported.

Both are tested in `tests/test_tracker_options.py`.

## Design on Wildtrack development frames (no other data)

Detector input, all seven cameras fused, the tracker parameters already tuned on those frames (`meas_sigma_m` 0.75, `accel_sigma` 2, `min_hits` 3). Grid: `confirm_frames` in {1, 2, 3} x `report_coasting` in {true, false} x detector score threshold in {0.5, 0.7, 0.9}. Objective: MOTA at 0.5 m on the development frames; ties to the simpler rule. The winner is **config B**. **Config A** is the original tracker with the Wildtrack-tuned values already frozen (score 0.9, `confirm_frames` 1, coasting reported). Nothing else changes between A and B, and nothing is tuned on the new scenes.

## New scenes: MMPTrack

- **Data.** MMPTrack (Han et al., multi-camera multi-person tracking, indoor scenes with calibrated cameras, per-camera boxes with identities, and top-down ground-truth positions). Provenance, as before: I could not verify the official host from here, and used a third party's Hugging Face mirror (`hungnguyen190204/MMPTracking`), which I cannot check against the original. I found no licence text, so the data stay local and **nothing is committed** except code and derived numbers.
- **Sequences.** Three validation sequences, one per scene type, fixed before looking at any result: `lobby_0` (4 cameras), `office_0` (5 cameras), `retail_0` (6 cameras). Images are 640 x 360. Every third labelled frame is used (about 5 frames per second, `dt` = 0.2 s, assuming the 15 frames per second of the dataset).
- **Conventions inferred before this protocol (ground-truth boxes and positions only, no detector, no tracker).** The calibration rotation is camera-to-world and the translation is the camera position, both in millimetres (a world point maps to the camera as R^T (X - T)); the ground plane is z = 0; the top-down CSV columns are (person id, V index, U index, height) with world U or V = Min + index x 20 mm. I found these by fitting the supplied calibration to the labelled boxes on `lobby_0` frames 0 to 580 (best of 48 variants, median reprojection error 15.8 px). With the final conventions, the ground-plane error of the dataset's own box bottom-centres is a median of 0.16 to 0.39 m and a 90th percentile of 0.23 to 0.63 m per camera across the three scenes, larger than Wildtrack's 0.07 to 0.13 m (image distortion is not modelled).
- **Region.** A detection counts only if its bottom-centre projects inside the calibration's stated space (geometry only, as for Wildtrack).
- **Inputs.** O, the dataset's own boxes; D, the same frozen COCO Faster R-CNN detector, run once per image, with the score threshold that config A or B carries.

## What is run, once, on each of the three sequences

| Run | What it tests |
| --- | --- |
| A, fuse-all, D | the original tracker with Wildtrack-tuned values on new scenes |
| B, fuse-all, D | the stricter rule, frozen from Wildtrack development |
| B, each camera alone, D | does fusion beat one camera now |
| A, each camera alone, D | the same for the original |
| A and B, fuse-all, O | the tracker with clean boxes |

## Metrics

Ground-plane CLEAR-MOT and IDF1 at a 0.5 m match threshold (and 1.0 m as a robustness row): MOTA, MOTP, identity switches, false positives, false negatives, IDF1. Pooled over the three sequences (counts summed) and per sequence. Intervals: block bootstrap, 20 blocks per sequence (60 in all), resampled with replacement, 2,000 draws, paired for differences between runs. Detector precision and recall per camera at IoU 0.5 against the dataset's boxes.

## Pre-stated expectations (written so they can be wrong)

- **A, D, pooled:** MOTA below 0.2, anywhere down to -1.0: the original rule does not become good on a new scene with a worse (low-resolution) detector.
- **B, D, pooled:** MOTA at least 0.3 above A, with a paired interval entirely above zero. B stays below the oracle runs.
- **Fusion:** under B, fuse-all beats the median single camera by at least 0.1 MOTA in every sequence, and beats the **best** single camera in at most one of the three.
- **Oracle:** B fuse-all (O) at least 0.75 MOTA pooled; A fuse-all (O) lower than B by at least 0.05.
- **IDF1 (D):** below 0.6 for both rules.
- **Transfer:** the detector score threshold chosen on Wildtrack is not optimal on the new scenes (a post hoc look will not be used to change anything).

## Limits, stated in advance

Three sequences of one dataset; 640 x 360 images from wide-angle cameras whose distortion I do not model; conventions inferred by me, not taken from documentation; an unofficial mirror; the same author wrote the tracker, the rule and this protocol. The rule is designed on a single other scene (Wildtrack, with seven cameras at 2 frames per second) and tested at 5 frames per second, which changes what a "frame" means, so a failure may be about frame rate rather than the rule. Subsampling every third frame is my choice. The two options are generic and not claimed to be the best way to confirm tracks.

## Deviations

(none yet)
