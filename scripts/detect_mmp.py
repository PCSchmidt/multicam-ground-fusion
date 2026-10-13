"""Run the frozen person detector once over every third frame of the chosen MMPTrack sequences (docs/PROTOCOL_SECOND_SCENE.md).

    python scripts/detect_mmp.py lobby office retail      # writes data/mmp/<scene>_det.npz (git-ignored)
"""

from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torchvision.io import ImageReadMode, decode_jpeg
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

from multicam_fusion.mmptrack import MmpSequence

ROOT = Path(__file__).resolve().parent.parent
MMP = ROOT / "data" / "mmp"
STEP = 3  # every third labelled frame (about 5 frames per second)
MIN_SCORE = 0.3


def main(scenes: list[str]) -> None:
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, box_score_thresh=MIN_SCORE
    ).cuda().eval()  # fmt: skip
    for sc in scenes:
        name = f"{sc}_0"
        seq = MmpSequence(MMP, sc, name)
        imgs = zipfile.ZipFile(MMP / sc / "images.zip")
        out: dict[str, np.ndarray] = {}
        t0 = time.time()
        with torch.no_grad():
            for i, f in enumerate(seq.frames[::STEP]):
                for cam in sorted(seq.cams):
                    data = imgs.read(f"{name}/rgb_{f:05d}_{cam}.jpg")
                    img = decode_jpeg(torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=ImageReadMode.RGB)
                    res = model([img.cuda().float() / 255.0])[0]
                    keep = res["labels"] == 1
                    out[f"c{cam}_f{f}"] = np.column_stack(
                        [res["boxes"][keep].cpu().numpy(), res["scores"][keep].cpu().numpy()]
                    ).astype(np.float32)
                if i % 200 == 0:
                    print(sc, i, "of", len(seq.frames[::STEP]), f"{time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(MMP / f"{sc}_det.npz", **out)
        print(sc, "saved", len(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
