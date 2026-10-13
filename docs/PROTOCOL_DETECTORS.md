# Protocol: is the detector the bottleneck? Three detectors on fresh MMPTrack sequences (written before any new detector was run or trained)

Follows [PROTOCOL_SECOND_SCENE.md](PROTOCOL_SECOND_SCENE.md) and [RESULTS_SECOND_SCENE.md](RESULTS_SECOND_SCENE.md). On MMPTrack the tracker is almost perfect with the dataset's own boxes (MOTA 0.996) and reaches 0.57 to 0.60 with an off-the-shelf COCO detector; at a 1 m match threshold the detector runs reach 0.84 to 0.87, so much of the loss looks like detection and localisation, not tracking. I have not tested that. This study asks how much of the gap each of three detectors closes, on sequences that were not used before.

## Detectors

| Name | What it is |
| --- | --- |
| **D0** | torchvision Faster R-CNN ResNet50-FPN v2, COCO weights, frozen (the detector used so far) |
| **D1** | torchvision RetinaNet ResNet50-FPN v2, COCO weights, frozen (a different architecture family; box mAP 41.5 on COCO against 46.7 for D0) |
| **D2** | D0 **fine-tuned** on person boxes from MMPTrack training sequences of the same three scenes, from a different recording session than any test sequence |

Fine-tuning recipe, fixed now and not tuned: start from the COCO weights, replace the box head with a two-class head (background, person), SGD with learning rate 0.01, momentum 0.9, weight decay 1e-4, batch size 4, 2 epochs, cosine decay, random horizontal flip, images at the torchvision default resize. Training images are every 15th frame (about 1 frame per second) of each camera, with the dataset's per-camera person boxes as targets, from the three fine-tuning sequences below. The only quantity chosen on development data is the score threshold.

## Sequences (all different from the sequences used before; all downloaded from the same unofficial mirror)

| Role | lobby | office | retail |
| --- | --- | --- | --- |
| **Fine-tuning** (training split, session 63am) | `lobby_1` | `office_0` | `retail_0` |
| **Development** (training split, session 64am; used for the score threshold only) | `lobby_0` | `office_0` | `retail_1` |
| **Test** (validation split, session 64pm; run once) | `lobby_1` | `office_1` | `retail_1` |

The three roles use different recording sessions and, within a scene, different sequences, so a detector fine-tuned on a morning session is tested on an afternoon session of the same room and cameras (a deployment-like setting: a detector adapted to a site, tested later at that site). Development uses the first 600 sampled frames of each development sequence (every third labelled frame, about 5 frames per second). Test uses all sampled frames.

## Tracker

The three frozen configurations A, B and C from [PROTOCOL_SECOND_SCENE.md](PROTOCOL_SECOND_SCENE.md) (`results/second_chosen.json`) with `dt` = 0.2 s, camera-consensus count scaled to each scene's camera count, and **nothing else tuned**. The detector score threshold is chosen per detector on the development sequences from {0.3, 0.5, 0.7, 0.9} by pooled MOTA of config B fused over all cameras at a 0.5 m match threshold, then used for A, B and C.

## What is run once on each test sequence

For each of D0, D1, D2: configs A, B and C fused over all cameras; config B with each camera alone. For the dataset's own boxes (O): A and B fused. Ground-plane CLEAR-MOT and IDF1 at 0.5 m and 1.0 m, pooled over the three sequences with a block bootstrap (20 blocks per sequence, 2,000 resamples, paired); per-sequence values; detector precision and recall per camera at IoU 0.5.

## Hypotheses, with criteria stated now

- **E1 (detection is the bottleneck).** D2 with config B reaches pooled MOTA at least 0.20 above D0 with config B, with a paired interval entirely above zero.
- **E2 (architecture matters less than domain).** D1 with config B is within +-0.10 MOTA of D0 with config B.
- **E3 (localisation).** With D2, pooled MOTA at 1.0 m minus pooled MOTA at 0.5 m is below 0.05 (with D0 it was about 0.26).
- **E4 (replication).** D0 with config A on the new sequences scores within +-0.15 of the earlier 0.568, and config B minus config A is below 0.05.
- **E5 (fusion).** With D2, fused config B beats the best single camera in all three test sequences.
- **E6 (consensus).** With D2, config C minus config B is below 0.01 pooled MOTA (few false positives remain for consensus to remove).
- **E7 (ceiling).** D2 with config B stays below the dataset's own boxes (O) by at least 0.05 pooled MOTA.

## Limits, stated in advance

Three scenes, one dataset, one mirror. Fine-tuning and test share scenes and camera positions, so D2 tests adaptation to a site, not generalisation to new sites. The 640 x 360 images are low resolution and the dataset's labels include small or partly visible people that a detector may reasonably skip. D1 and D0 are two COCO detectors, not a survey. The threshold is tuned on 600 sampled frames per scene. A single fine-tuning recipe, untuned. One author.

## Deviations

(none yet)
