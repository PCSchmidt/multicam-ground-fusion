# multicam-ground-fusion

Multi-camera, multi-target tracking on a shared ground plane. Detector boxes from calibrated cameras are projected through planar homographies to metric ground coordinates, associated to global tracks by the Hungarian algorithm behind a chi-square **Mahalanobis gate**, and fused with a constant-velocity Kalman filter. Tracks that move faster than a physical ceiling are *flagged*, never silently corrected.

**Status:** v0.1 scaffold, validated on a seeded synthetic scene only. No YOLO/ByteTrack front end, no Wildtrack/PETS data, no BEV renderer yet. See [docs/RESULTS.md](docs/RESULTS.md).

## Design rule

The detector is probabilistic and untrusted. Everything downstream is deterministic: homography projection, gate threshold (`chi2.ppf(0.99, 2)`), assignment, track birth/death, and the `IMPLAUSIBLE_SPEED` flag.

## Use

```python
from multicam_fusion import GroundPlaneTracker, Detection
tracker = GroundPlaneTracker({"A": H_A_pix2world, "B": H_B_pix2world})
confirmed = tracker.step(dt=0.1, detections=[Detection("A", (x0, y0, x1, y1)), ...])
for t in confirmed:
    t.position, t.speed, t.flags
```

`estimate_homography(src_pts, dst_pts)` (DLT) builds a pixel-to-world matrix from surveyed points; `covariance_ellipse(t.p[:2, :2])` gives the drawing parameters for a bird's-eye view.

## Checks

```bash
pytest                         # incl. hypothesis: covariance stays symmetric PSD
mypy src tests scripts         # strict
python scripts/benchmark.py    # synthetic MOTA / ID-switch table
```

## Limits

- The measurement is linear after projection, so this is a plain Kalman filter. An EKF becomes necessary only if you filter in pixel space; not implemented.
- Measurement noise is a single fixed sigma. Real homography error grows with distance from the camera; a per-detection covariance via the homography Jacobian is the obvious next step.
- No appearance re-identification: cross-camera association is geometry-only, so crossing targets can swap IDs.
- Detections are bottom-centre points; bbox bottoms are biased when feet are occluded.
