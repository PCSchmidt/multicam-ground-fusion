# Results

Synthetic only: 4 targets moving in a 40 x 40 m yard for 200 frames at 10 Hz, wall reflections, pixel noise 3 px, per-detection dropout, two cameras with perspective homographies (camera A sees x in 0-26 m, camera B 14-40 m, so the overlap is 14-26 m). Mean over 10 held-out seeds (100-109) per row. Reproduce: `python scripts/benchmark.py`.

`meas_sigma_m` was tuned on dev seeds 0-4 only.

| Cameras | Dropout | MOTA | MOTP (m) | ID switches / run | FN / run | FP / run |
|---|---|---|---|---|---|---|
| A | 0.1 | 0.665 | 0.198 | 1.2 | 262 | 5 |
| A | 0.3 | 0.650 | 0.222 | 1.8 | 270 | 8 |
| A | 0.5 | 0.613 | 0.258 | 5.7 | 298 | 6 |
| A+B | 0.1 | 0.952 | 0.177 | 3.1 | 8 | 28 |
| A+B | 0.3 | 0.960 | 0.190 | 3.6 | 14 | 15 |
| A+B | 0.5 | 0.926 | 0.225 | 5.3 | 41 | 13 |

(800 ground-truth object-frames per run.)

## Reading it

- Single-camera MOTA is bounded by field of view: camera A cannot see 35% of the yard, so about 260 of 800 object-frames are structural misses. The two-camera gain is mostly coverage, not smarter filtering; the table does not separate the two.
- Fusion in the overlap raises MOTP slightly (0.18 vs 0.20 m at dropout 0.1).
- ID switches rise with dropout in both settings; two-camera runs have slightly more switches than one-camera runs at low dropout (3.1 vs 1.2); likely camera hand-over, not tested.
- N=10 seeds: treat differences of a few ID switches as noise.

## Development finding

With the first default (`meas_sigma_m=0.3`) the two-camera FP count was about 194 per run and MOTA 0.72. The actual per-axis world error from 3 px of pixel noise is about 0.33 m, so the gate rejected valid detections and spawned duplicate tracks. Raising sigma to 0.5 fixed it. Lesson: the gate is only as good as the noise model, and this is the main risk for real data.

## Not measured

Real detections (YOLO + ByteTrack), Wildtrack / PETS 2009, IDF1, crossing-target identity swaps, drift detection recall, BEV rendering.
