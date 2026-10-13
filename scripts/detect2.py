"""Run one detector over the sampled frames of the chosen sequences and cache the person boxes (docs/PROTOCOL_DETECTORS.md).

    python scripts/detect2.py d0 dev_lobby dev_office dev_retail     # writes data/mmp2/<tag>_<det>.npz

d0 = Faster R-CNN v2 (COCO), d1 = RetinaNet v2 (COCO), d2 = d0 fine-tuned by scripts/finetune_detector.py.
Development sequences use only the first DEV_STEPS sampled frames.
"""

from __future__ import annotations

import sys
import time
import zipfile

import numpy as np
import torch
from mmp2_common import DEV_STEPS, MMP2, SEQUENCES, STEP
from torchvision.io import ImageReadMode, decode_jpeg
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from multicam_fusion.mmptrack import MmpSequence

MIN_SCORE = 0.3


def build(det: str) -> torch.nn.Module:
    if det == "d0":
        return fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, box_score_thresh=MIN_SCORE)
    if det == "d1":
        return retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1, score_thresh=MIN_SCORE)
    if det == "d2":
        m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, box_score_thresh=MIN_SCORE)
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
        m.load_state_dict(torch.load(MMP2 / "d2_frcnn.pt", map_location="cpu"))
        return m
    raise SystemExit(f"unknown detector {det}")


def main(det: str, tags: list[str]) -> None:
    model = build(det).cuda().eval()
    for tag in tags:
        _, name = SEQUENCES[tag]
        seq = MmpSequence(MMP2, tag, name)
        imgs = zipfile.ZipFile(MMP2 / tag / "images.zip")
        frames = seq.frames[::STEP]
        if tag.startswith("dev"):
            frames = frames[:DEV_STEPS]
        out: dict[str, np.ndarray] = {}
        t0 = time.time()
        with torch.no_grad():
            for i, f in enumerate(frames):
                for cam in sorted(seq.cams):
                    data = imgs.read(f"{name}/rgb_{f:05d}_{cam}.jpg")
                    img = decode_jpeg(torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=ImageReadMode.RGB)
                    res = model([img.cuda().float() / 255.0])[0]
                    keep = res["labels"] == 1
                    out[f"c{cam}_f{f}"] = np.column_stack(
                        [res["boxes"][keep].cpu().numpy(), res["scores"][keep].cpu().numpy()]
                    ).astype(np.float32)
                if i % 300 == 0:
                    print(det, tag, i, "of", len(frames), f"{time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(MMP2 / f"{tag}_{det}.npz", **out)
        print(det, tag, "saved", len(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
