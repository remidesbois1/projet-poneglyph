"""Fine-tune Indra's line detector (det.onnx) on bubble crops with YOLO pseudo-labels.

There are no human line boxes: the targets are the production YOLO line boxes
(conf 0.10, line NMS 0.75).  Bubbles where YOLO found no line, or where a
reviewer saw that YOLO omitted visible text, are left out.  Targets use the
official encoder (indra_ocr.det_post.encode_targets: core band + top/bottom
distances at 1/2 resolution), and the input is the official page_to_input at
the bubble scale frozen on validation (0.75 x the crop size, jittered in training).
The published graph is trained as is and written back into a copy of det.onnx.

  python train_det.py --data /workspace/indra/data --out /workspace/indra/det_runs/d1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from indra_ocr.det_post import decode, encode_targets, page_to_input

from onnx_torch import OnnxModule


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-6)


def matches(ref, pred, thr=0.5) -> int:
    pairs = sorted(((iou(r, p), i, j) for i, r in enumerate(ref) for j, p in enumerate(pred)), reverse=True)
    ur, up, m = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in ur and j not in up:
            ur.add(i)
            up.add(j)
            m += 1
    return m


class Crops(torch.utils.data.Dataset):
    def __init__(self, data: Path, split: str, train: bool, scale: float):
        rows = [json.loads(l) for l in open(data / f"ppocr_{split}.jsonl", encoding="utf-8") if l.strip()]
        self.items = [{"crop": data / r["crop"], "boxes": np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in r["yolo_boxes"]], np.float32)}
                      for r in rows if r["yolo_boxes"] and not (train and r.get("train_excluded"))]
        self.dropped = len(rows) - len(self.items)
        self.train, self.scale = train, scale

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        gray = cv2.imread(str(it["crop"]), cv2.IMREAD_GRAYSCALE)
        boxes = it["boxes"].copy()
        scale = self.scale
        if self.train:
            g = gray.astype(np.float32)
            if random.random() < 0.3:
                g = (g - 128) * random.uniform(0.7, 1.2) + 128 + random.uniform(-25, 25)
            if random.random() < 0.2:
                g = cv2.GaussianBlur(g, (0, 0), random.uniform(0.3, 0.9))
            if random.random() < 0.2:
                g = g + np.random.normal(0, random.uniform(2, 10), g.shape)
            gray = g.clip(0, 255).astype(np.uint8)
            if random.random() < 0.3:  # white border, as when the bubble box is loose
                m = random.randint(2, 16)
                gray = cv2.copyMakeBorder(gray, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
                boxes = boxes + m
            scale *= math.exp(random.uniform(math.log(0.8), math.log(1.25)))
        long_side = max(32, round(max(gray.shape) * scale))
        x, s = page_to_input(gray, long_side)
        nh, nw = max(1, round(gray.shape[0] * s)), max(1, round(gray.shape[1] * s))
        core, dist, weight, _ = encode_targets(boxes * s, x.shape, (nh, nw))
        return x, core, dist, weight, i, s


def collate(batch):
    H = max(b[0].shape[0] for b in batch)
    W = max(b[0].shape[1] for b in batch)
    n = len(batch)
    x = np.zeros((n, 1, H, W), np.float32)
    core = np.zeros((n, H // 2, W // 2), np.float32)
    dist = np.zeros((n, 2, H // 2, W // 2), np.float32)
    weight = np.zeros((n, H // 2, W // 2), np.float32)
    for k, (xi, c, d, w, _, _) in enumerate(batch):
        h, wd = xi.shape
        x[k, 0, :h, :wd] = xi
        core[k, : c.shape[0], : c.shape[1]] = c
        dist[k, :, : c.shape[0], : c.shape[1]] = d
        weight[k, : c.shape[0], : c.shape[1]] = w
    return (torch.from_numpy(x), torch.from_numpy(core), torch.from_numpy(dist), torch.from_numpy(weight),
            [b[4] for b in batch], [b[5] for b in batch])


def size_batches(ds: Crops, batch: int, shuffle: bool):
    keys = []
    for it in ds.items:
        k = len(it["boxes"])
        keys.append(k * (random.uniform(0.8, 1.2) if shuffle else 1))
    order = sorted(range(len(ds)), key=lambda i: keys[i])
    out = [order[k:k + batch] for k in range(0, len(order), batch)]
    if shuffle:
        random.shuffle(out)
    return out


def loss_fn(out, core, dist, weight):
    logit, pred = out[:, 0], out[:, 1:3]
    l_core = (F.binary_cross_entropy_with_logits(logit, core, reduction="none") * weight).sum() / weight.sum().clamp(min=1)
    mask = core > 0.5
    if mask.any():
        tgt = dist.permute(0, 2, 3, 1)[mask]
        p = pred.permute(0, 2, 3, 1)[mask]
        l_dist = ((p - tgt).abs().sum(-1) / tgt.sum(-1).clamp(min=1)).mean()
    else:
        l_dist = out.sum() * 0
    return l_core + l_dist, l_core.item(), l_dist.item()


@torch.no_grad()
def evaluate(model, ds, device) -> dict:
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_sampler=size_batches(ds, 32, False), collate_fn=collate, num_workers=4)
    m = n_ref = n_pred = zero = 0
    for x, _, _, _, idx, scales in loader:
        out = model(x=x.to(device))["out"].float().cpu().numpy()
        for k, i in enumerate(idx):
            boxes = decode(out[k, :3])[:, :4] / scales[k]
            ref = ds.items[i]["boxes"]
            m += matches(ref, boxes)
            n_ref += len(ref)
            n_pred += len(boxes)
            zero += len(boxes) == 0
    model.train()
    p, r = m / max(n_pred, 1), m / max(n_ref, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "zero_line_bubbles": zero}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--models", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    models = args.models or args.data / "models"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    train, val = Crops(args.data, "train", True, args.scale), Crops(args.data, "val", False, args.scale)
    if args.limit:
        train.items, val.items = train.items[: args.limit], val.items[: args.limit]
    print(json.dumps({"train": len(train), "train_dropped": train.dropped, "val": len(val), "device": device}), flush=True)
    model = OnnxModule(models / "det.onnx", outputs=["out"]).to(device).train()
    ema = OnnxModule(models / "det.onnx", outputs=["out"]).to(device).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    decay = [p for p in model.parameters() if p.dim() > 1]
    no_decay = [p for p in model.parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    total = math.ceil(len(train) / args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / args.warmup if s < args.warmup
                                              else 0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, total - args.warmup))))
    history = [{"epoch": 0, **evaluate(model, val, device)}]
    print(json.dumps(history[-1]), flush=True)
    best, step, t0 = -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        loader = torch.utils.data.DataLoader(train, batch_sampler=size_batches(train, args.batch, True), collate_fn=collate,
                                             num_workers=args.workers, prefetch_factor=4 if args.workers else None)
        parts = []
        for x, core, dist, weight, _, _ in loader:
            out = model(x=x.to(device, non_blocking=True))["out"]
            loss, lc, ld = loss_fn(out, core.to(device), dist.to(device), weight.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            step += 1
            with torch.no_grad():
                d = min(args.ema, (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            parts.append((lc, ld))
        res = evaluate(ema, val, device)
        rec = {"epoch": epoch, "step": step, "loss_core": float(np.mean([p[0] for p in parts])),
               "loss_dist": float(np.mean([p[1] for p in parts])), **res, "minutes": (time.time() - t0) / 60}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if res["f1"] > best:
            best = res["f1"]
            ema.export(models / "det.onnx", args.out / "det_best.onnx")
        (args.out / "history.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "history": history}, indent=1))
    ema.export(models / "det.onnx", args.out / "det_last.onnx")
    print(json.dumps({"best_val_f1_vs_yolo": best}), flush=True)


if __name__ == "__main__":
    main()
