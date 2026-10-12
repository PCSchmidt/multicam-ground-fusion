"""Run the person detector once over every Wildtrack image and cache the boxes (docs/PROTOCOL_REAL.md).

    python scripts/detect.py            # writes data/detections.npz (git-ignored)

torchvision Faster R-CNN ResNet50-FPN v2, COCO weights, class `person`. Boxes with score >= 0.3 are cached; the
score threshold used by the study is chosen later on development frames only.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torchvision.io import read_image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

from multicam_fusion.wildtrack import frame_numbers

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Wildtrack_dataset"
PERSON = 1  # COCO category id in torchvision's label map
MIN_SCORE = 0.3


def main() -> None:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=MIN_SCORE).cuda().eval()
    frames = frame_numbers(DATA)
    out: dict[str, np.ndarray] = {}
    t0 = time.time()
    with torch.no_grad():
        for cam in range(7):
            for i, f in enumerate(frames):
                img = read_image(str(DATA / "Image_subsets" / f"C{cam + 1}" / f"{f:08d}.png"))
                img = img[:3].cuda().float() / 255.0
                res = model([img])[0]
                keep = res["labels"] == PERSON
                b = res["boxes"][keep].cpu().numpy()
                s = res["scores"][keep].cpu().numpy()
                out[f"c{cam}_f{f}"] = np.column_stack([b, s]).astype(np.float32)
            print(f"camera {cam + 1}/7 done, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(ROOT / "data" / "detections.npz", **out)
    print("saved", len(out), "arrays")


if __name__ == "__main__":
    main()
