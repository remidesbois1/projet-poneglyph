"""Fine-tune Indra's Latin recognizer (rec.onnx) on Poneglyph bubbles.

Labels exist per bubble, not per line, so each training input is the bubble's
line windows side by side (IndraRec.stitch: official crop_line, tall window),
and the CTC target is the whole bubble transcription.  The published graph is
trained as is (onnx_torch.OnnxModule) and the weights are written back into a
copy of rec.onnx, so the result keeps rec.onnx's exact inputs and outputs.

Line boxes come from the production YOLO detector and, when given, from
Indra's det.onnx, so one recognizer serves both pipelines.

  python train_rec.py --data /workspace/data --out /workspace/runs/r1 --epochs 40 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import torch
from indra_ocr.rec_preproc import crop_line, is_dark_line, pad_batch
from indra_ocr.vocab import Vocab, canonical

from onnx_torch import OnnxModule

LOGPROBS = "/LogSoftmax_output_0"


def norm_metric(t: str) -> str:
    """runtime.cjs normalize(): NFKC, straight apostrophes, ellipsis, whitespace."""
    t = unicodedata.normalize("NFKC", t or "")
    for c in "\u2018\u2019\u00b4`":
        t = t.replace(c, "'")
    return " ".join(t.replace("\u2026", "...").split())


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def target_text(text: str, case: str) -> str:
    t = canonical(" ".join((text or "").split()))
    return t.upper() if case == "upper" else t


class Bubbles(torch.utils.data.Dataset):
    def __init__(self, data: Path, split: str, vocab: Vocab, height: int, window: int, train: bool, args, box_source: str = "mix"):
        rows = [json.loads(l) for l in open(data / f"ppocr_{split}.jsonl", encoding="utf-8") if l.strip()]
        indra = {}
        p = data / f"indra_boxes_{split}.jsonl"
        if p.exists():
            indra = {r["bubble_id"]: r["boxes"] for r in map(json.loads, open(p, encoding="utf-8"))}
        self.items, self.dropped = [], {"excluded": 0, "empty": 0, "oov": 0}
        for r in rows:
            if train and r.get("train_excluded"):
                self.dropped["excluded"] += 1
                continue
            text = target_text(r["reference"], args.case)
            if not text:
                self.dropped["empty"] += 1
                continue
            ids = vocab.encode(text)
            if ids is None:
                self.dropped["oov"] += 1
                continue
            yolo = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in r["yolo_boxes"]]
            self.items.append({"id": r["bubble_id"], "crop": data / r["crop"], "ids": ids, "text": r["reference"],
                               "yolo": yolo, "indra": indra.get(r["bubble_id"])})
        self.vocab, self.height, self.window, self.train, self.args = vocab, height, window, train, args
        self.box_source = box_source

    def __len__(self):
        return len(self.items)

    def boxes(self, it, gray) -> np.ndarray:
        src = self.box_source
        if src == "mix":
            src = "indra" if it["indra"] is not None and random.random() < self.args.p_indra else "yolo"
        b = it[src] if it[src] is not None else it["yolo"]
        b = np.array(b, np.float32).reshape(-1, 4)
        if not len(b):
            return np.array([[0, 0, gray.shape[1], gray.shape[0]]], np.float32)
        if self.train:
            h = (b[:, 3] - b[:, 1])[:, None]
            jit = np.random.uniform(-1, 1, b.shape).astype(np.float32) * h * np.array([0.15, 0.08, 0.15, 0.08], np.float32)
            grow = np.random.uniform(-0.12, 0.06)
            b = b + jit + np.array([0, -grow, 0, grow], np.float32) * h
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, gray.shape[1])
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, gray.shape[0])
            b[:, 2] = np.maximum(b[:, 2], b[:, 0] + 2)
            b[:, 3] = np.maximum(b[:, 3], b[:, 1] + 2)
        return b

    def photometric(self, gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.float32)
        if random.random() < 0.3:
            g = (g - 128) * random.uniform(0.7, 1.2) + 128 + random.uniform(-25, 25)
        if random.random() < 0.2:
            g = cv2.GaussianBlur(g, (0, 0), random.uniform(0.3, 0.9))
        if random.random() < 0.2:
            g = g + np.random.normal(0, random.uniform(2, 10), g.shape)
        if random.random() < 0.15:  # resolution loss
            s = random.uniform(0.55, 0.9)
            small = cv2.resize(g, (max(1, round(g.shape[1] * s)), max(1, round(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
            g = cv2.resize(small, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
        return g.clip(0, 255).astype(np.uint8)

    def __getitem__(self, i):
        it = self.items[i]
        gray = cv2.imread(str(it["crop"]), cv2.IMREAD_GRAYSCALE)
        if self.train:
            gray = self.photometric(gray)
        boxes = self.boxes(it, gray)
        crops = [crop_line(gray, b, self.height, window=self.window, dark=is_dark_line(gray, b)) for b in boxes]
        gap = random.randint(0, 8) if self.train else self.args.gap
        if gap and len(crops) > 1:
            pad = np.zeros((crops[0].shape[0], gap), np.float32)
            crops = [c for crop in crops for c in (crop, pad)][:-1]
        return np.concatenate(crops, 1), it["ids"], i


def collate(batch):
    arr, widths = pad_batch([b[0] for b in batch])
    targets = torch.tensor([t for b in batch for t in b[1]], dtype=torch.long)
    tlen = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    return torch.from_numpy(arr), torch.from_numpy(widths), targets, tlen, [b[2] for b in batch]


def width_sorted_batches(ds: Bubbles, batch: int, shuffle: bool) -> list[list[int]]:
    """Batches of similar size (box count x crop width as proxy), shuffled."""
    keys = []
    for i, it in enumerate(ds.items):
        n = max(1, len(it["yolo"]))
        keys.append((n * 40 + len(it["ids"])) * (random.uniform(0.9, 1.1) if shuffle else 1))
    order = sorted(range(len(ds)), key=lambda i: keys[i])
    batches = [order[k:k + batch] for k in range(0, len(order), batch)]
    if shuffle:
        random.shuffle(batches)
    return batches


@torch.no_grad()
def evaluate(model, ds, vocab, device, batch=64, case="original") -> dict:
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_sampler=width_sorted_batches(ds, batch, False), collate_fn=collate,
                                         num_workers=4, persistent_workers=False)
    errors = chars = exact = 0
    preds = {}
    for x, w, _, _, idx in loader:
        out = model(x=x.to(device), widths=w.to(device))
        lp, lens = out[LOGPROBS], out["lengths"]
        best = lp.argmax(-1).cpu().numpy()
        for k, i in enumerate(idx):
            seq = best[k, : int(lens[k])]
            keep = (seq != 0) & np.concatenate([[True], seq[1:] != seq[:-1]])
            text = vocab.decode(seq[keep].tolist())
            ref = ds.items[i]["text"]
            if case == "upper":
                ref = ref.upper()
            a, b = norm_metric(ref), norm_metric(text)
            errors += levenshtein(a, b)
            chars += len(a)
            exact += a == b
            preds[ds.items[i]["id"]] = text
    model.train()
    return {"cer": errors / max(chars, 1), "exact": exact / max(len(ds), 1), "n": len(ds), "preds": preds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--models", type=Path, default=None, help="folder with rec.onnx and meta.json (default: data/models)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--p-indra", type=float, default=0.4, help="share of samples read with Indra det boxes")
    ap.add_argument("--case", choices=["original", "upper"], default="original")
    ap.add_argument("--gap", type=int, default=0, help="inference gap between line windows")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: limit samples per split")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    models = args.models or args.data / "models"
    meta = json.loads((models / "meta.json").read_text(encoding="utf-8"))
    vocab = Vocab.from_json(meta["vocab"])
    height, window = meta["rec_height"], meta.get("rec_window", 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    train = Bubbles(args.data, "train", vocab, height, window, True, args)
    val = {src: Bubbles(args.data, "val", vocab, height, window, False, args, box_source=src) for src in ("yolo", "indra")}
    if args.limit:
        train.items = train.items[: args.limit]
        for v in val.values():
            v.items = v.items[: args.limit]
    print(json.dumps({"train": len(train), "dropped": train.dropped, "val": len(val["yolo"]), "val_dropped": val["yolo"].dropped,
                      "device": device, "torch": torch.__version__}), flush=True)

    model = OnnxModule(models / "rec.onnx", outputs=[LOGPROBS, "lengths"]).to(device).train()
    ema = OnnxModule(models / "rec.onnx", outputs=[LOGPROBS, "lengths"]).to(device).eval()
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    decay, no_decay = [], []
    for name, p in model.params.items():
        (decay if p.dim() > 1 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = steps_per_epoch * args.epochs

    def lr_at(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, total - args.warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    base = evaluate(model, val["yolo"], vocab, device, case=args.case)
    history = [{"epoch": 0, "val_yolo_cer": base["cer"], "val_yolo_exact": base["exact"]}]
    print(json.dumps(history[-1]), flush=True)
    best, step, t0 = math.inf, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        loader = torch.utils.data.DataLoader(train, batch_sampler=width_sorted_batches(train, args.batch, True), collate_fn=collate,
                                             num_workers=args.workers, prefetch_factor=4 if args.workers else None)
        losses = []
        for x, w, tg, tl, _ in loader:
            out = model(x=x.to(device, non_blocking=True), widths=w.to(device))
            lp = out[LOGPROBS].permute(1, 0, 2)  # T, B, C
            loss = ctc(lp, tg.to(device), out["lengths"], tl.to(device))
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
            losses.append(loss.item())
        res = {src: evaluate(ema, v, vocab, device, case=args.case) for src, v in val.items()}
        rec = {"epoch": epoch, "step": step, "loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0],
               "val_yolo_cer": res["yolo"]["cer"], "val_yolo_exact": res["yolo"]["exact"],
               "val_indra_cer": res["indra"]["cer"], "val_indra_exact": res["indra"]["exact"], "minutes": (time.time() - t0) / 60}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if res["yolo"]["cer"] < best:
            best = res["yolo"]["cer"]
            ema.export(models / "rec.onnx", args.out / "rec_best.onnx")
            torch.save({"epoch": epoch, "state": ema.state_dict()}, args.out / "best.pt")
            (args.out / "val_preds_best.json").write_text(json.dumps(res["yolo"]["preds"], ensure_ascii=False), encoding="utf-8")
        (args.out / "history.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "history": history}, indent=1))
    ema.export(models / "rec.onnx", args.out / "rec_last.onnx")
    print(json.dumps({"best_val_yolo_cer": best, "minutes": (time.time() - t0) / 60}), flush=True)


if __name__ == "__main__":
    main()
