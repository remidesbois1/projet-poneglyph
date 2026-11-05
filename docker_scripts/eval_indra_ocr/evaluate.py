"""Run Indra configurations on annotated bubble crops and write raw predictions.

Scoring (production post-processing + CER) is done by score.cjs, identical for
every configuration including the production PP-OCRv6 baseline.

  python evaluate.py --split val --name B_lines --det yolo --mode lines
  python evaluate.py --split test --name C_ft --det indra --det-scale 1.0 --mode stitched --rec outputs/ft/rec_poneglyph.onnx
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from indra_lib import (
    OUT,
    IndraDet,
    IndraRec,
    INDRA_DIR,
    Timer,
    adjust_boxes,
    boxes_array,
    join_lines,
    load_rows,
    read_gray,
    sort_line_boxes,
    whole_crop_box,
)


@dataclass(frozen=True)
class Config:
    det: str = "yolo"  # yolo (production boxes) | indra (det.onnx)
    det_scale: float = 1.0
    det_margin: int = 0
    det_thresh: float = 0.5
    mode: str = "lines"  # lines | stitched
    gap: int = 0
    grow_y: float = 0.0
    dehyphenate: bool = True
    lm: bool = False
    min_conf: float = 0.0
    rec: str = ""  # rec.onnx path; empty = official Indra v1.1.0
    det_model: str = ""  # det.onnx path; empty = official Indra v1.1.0


def predict(rows: list[dict], cfg: Config, det: IndraDet | None, rec: IndraRec) -> list[dict]:
    preds = []
    for r in rows:
        gray = read_gray(r["crop"])
        t = Timer()
        with t("detect"):
            if cfg.det == "yolo":
                boxes = boxes_array(r["yolo_boxes"])  # already in production reading order
            else:
                boxes = sort_line_boxes(det.detect(gray, cfg.det_scale, cfg.det_margin, cfg.det_thresh))
        n_lines = len(boxes)
        boxes = adjust_boxes(boxes, cfg.grow_y, gray.shape) if n_lines else whole_crop_box(gray)
        with t("recognize"):
            if cfg.mode == "stitched":
                raw, conf = rec.read_stitched(gray, boxes, cfg.gap)
                confs = [conf]
            else:
                lines = rec.read_lines(gray, boxes)
                confs = [c for _, c in lines]
                raw = join_lines([s for s, c in lines if c >= cfg.min_conf], cfg.dehyphenate)
        preds.append({"bubble_id": r["bubble_id"], "reference": r["reference"], "raw": raw, "lines": n_lines,
                      "boxes": np.round(boxes, 1).tolist() if cfg.det == "indra" else None,
                      "conf": float(np.mean(confs)) if confs else 0.0, "ms": t.ms})
    return preds


def full_indra(rows: list[dict], threads: int = 4) -> list[dict]:
    """Out-of-the-box indra_ocr on the crop (orientation, deskew, layout, order...)."""
    import indra_ocr

    conv = indra_ocr.load(path=str(INDRA_DIR), threads=threads, mode="ocr")
    preds = []
    for r in rows:
        page = conv.ocr(read_gray(r["crop"]))
        preds.append({"bubble_id": r["bubble_id"], "reference": r["reference"],
                      "raw": join_lines([l.text for l in page.lines], True), "lines": len(page.lines),
                      "ms": {"total": page.timings.get("total_ms", 0.0)}})
    return preds


def write(name: str, split: str, cfg: dict, preds: list[dict]) -> Path:
    path = OUT / "predictions" / split / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"config": cfg, "split": split, "n": len(preds)}, ensure_ascii=False) + "\n")
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--full-indra", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    for field, default in asdict(Config()).items():
        flag = "--" + field.replace("_", "-")
        if isinstance(default, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=default)
        else:
            ap.add_argument(flag, type=type(default), default=default)
    a = ap.parse_args()
    rows = load_rows(a.split)
    if a.full_indra:
        print(write(a.name, a.split, {"full_indra": True}, full_indra(rows, a.threads)))
        return
    cfg = Config(**{k: getattr(a, k) for k in asdict(Config())})
    rec = IndraRec(a.threads, Path(cfg.rec) if cfg.rec else INDRA_DIR / "rec.onnx", lm=cfg.lm)
    det = IndraDet(a.threads, Path(cfg.det_model) if cfg.det_model else INDRA_DIR / "det.onnx") if cfg.det == "indra" else None
    print(write(a.name, a.split, asdict(cfg), predict(rows, cfg, det, rec)))


if __name__ == "__main__":
    main()
