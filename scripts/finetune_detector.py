"""Fine-tune the COCO Faster R-CNN on MMPTrack person boxes (D2 of docs/PROTOCOL_DETECTORS.md).

    python scripts/finetune_detector.py       # writes data/mmp2/d2_frcnn.pt

Recipe fixed in the protocol: two-class head, SGD lr 0.01 momentum 0.9 weight decay 1e-4, batch 4, 2 epochs, cosine
decay, horizontal flip; every 15th labelled frame of every camera of the three fine-tuning sequences.
"""

from __future__ import annotations

import random
import time
import zipfile
from pathlib import Path

import torch
from mmp2_common import MMP2, SEQUENCES
from torchvision.io import ImageReadMode, decode_jpeg
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from multicam_fusion.mmptrack import MmpSequence

EPOCHS = 2
BATCH = 4
LR = 0.01
FRAME_STEP = 15


class Items:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str, int, int]] = []  # tag, name, frame, camera
        self.seqs: dict[str, MmpSequence] = {}
        self.zips: dict[str, zipfile.ZipFile] = {}
        for tag in ("ft_lobby", "ft_office", "ft_retail"):
            _, name = SEQUENCES[tag]
            seq = MmpSequence(MMP2, tag, name)
            self.seqs[tag] = seq
            self.zips[tag] = zipfile.ZipFile(MMP2 / tag / "images.zip")
            for f in seq.frames[::FRAME_STEP]:
                for cam in sorted(seq.cams):
                    self.entries.append((tag, name, f, cam))

    def get(self, i: int, flip: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tag, name, f, cam = self.entries[i]
        data = self.zips[tag].read(f"{name}/rgb_{f:05d}_{cam}.jpg")
        img = decode_jpeg(torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=ImageReadMode.RGB).float() / 255.0
        h, w = img.shape[1:]
        boxes = []
        for b in self.seqs[tag].boxes(f, cam).values():
            x0, y0, x1, y1 = max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                boxes.append([x0, y0, x1, y1])
        t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if flip:
            img = img.flip(-1)
            t = torch.stack([w - t[:, 2], t[:, 1], w - t[:, 0], t[:, 3]], dim=1) if len(t) else t
        return img, {"boxes": t, "labels": torch.ones(len(t), dtype=torch.int64)}


def main() -> None:
    random.seed(0)
    torch.manual_seed(0)
    items = Items()
    n = len(items.entries)
    print("training images:", n, flush=True)
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.cuda().train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)
    total = EPOCHS * (n // BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)
    t0 = time.time()
    step = 0
    for epoch in range(EPOCHS):
        order = list(range(n))
        random.shuffle(order)
        for i in range(0, n - BATCH + 1, BATCH):
            batch = [items.get(j, flip=random.random() < 0.5) for j in order[i : i + BATCH]]
            imgs = [b[0].cuda() for b in batch]
            targets = [{k: v.cuda() for k, v in b[1].items()} for b in batch]
            loss = sum(model(imgs, targets).values())
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{total} loss {float(loss):.3f} {time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), MMP2 / "d2_frcnn.pt")
    print("saved", Path(MMP2 / "d2_frcnn.pt"), flush=True)


if __name__ == "__main__":
    main()
