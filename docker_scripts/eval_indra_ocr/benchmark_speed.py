"""CPU latency of the three bubble pipelines in one harness (Python, ONNX Runtime CPU,
same session options, warm sessions, one bubble at a time, sequential).

A: YOLO lines + PP-OCRv6 (Python port of the production worker; the port's
   outputs are compared with the production Node outputs as a sanity check)
B: YOLO lines + Indra rec
C: Indra det + Indra rec

  python benchmark_speed.py --split test --threads 4 --b-mode lines --c-scale 1.0 [--rec path]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from indra_lib import (
    INDRA_DIR,
    OUT,
    YOLO_ONNX,
    IndraDet,
    IndraRec,
    PpocrRec,
    Timer,
    YoloLines,
    boxes_array,
    join_lines,
    load_rows,
    match_rate,
    sort_line_boxes,
    whole_crop_box,
)


def stats(values: list[float]) -> dict:
    v = np.array(values)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--b-mode", default="lines", choices=["lines", "stitched"])
    ap.add_argument("--c-mode", default="lines", choices=["lines", "stitched"])
    ap.add_argument("--c-scale", type=float, default=1.0)
    ap.add_argument("--c-margin", type=int, default=0)
    ap.add_argument("--rec", default="")
    ap.add_argument("--det-model", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rows = load_rows(a.split)[: a.limit or None]
    yolo, ppocr = YoloLines(a.threads), PpocrRec(a.threads)
    det = IndraDet(a.threads, Path(a.det_model) if a.det_model else INDRA_DIR / "det.onnx")
    rec = IndraRec(a.threads, Path(a.rec) if a.rec else INDRA_DIR / "rec.onnx")

    def read(gray, boxes, mode):
        boxes = boxes if len(boxes) else whole_crop_box(gray)
        return rec.read_stitched(gray, boxes, 0)[0] if mode == "stitched" else join_lines([t for t, _ in rec.read_lines(gray, boxes)], True)

    # warm-up on a few bubbles (allocations, first-run graph work)
    for r in rows[:5]:
        bgr = cv2.imread(str(OUT / r["crop"]))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        b = yolo.detect(bgr)
        ppocr.read(bgr, b)
        read(gray, b, a.b_mode)
        read(gray, sort_line_boxes(det.detect(gray, a.c_scale, a.c_margin)), a.c_mode)

    per = {k: {"detect": [], "recognize": [], "total": []} for k in "ABC"}
    port = {"box_matched": 0, "node_boxes": 0, "py_boxes": 0, "text_equal": 0}
    for r in rows:
        t = Timer()
        with t("decode"):
            bgr = cv2.imread(str(OUT / r["crop"]))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        with t("A_detect"):
            yb = yolo.detect(bgr)
        with t("A_recognize"):
            a_raw = ppocr.read(bgr, yb)
        with t("B_recognize"):
            read(gray, yb, a.b_mode)
        with t("C_detect"):
            cb = sort_line_boxes(det.detect(gray, a.c_scale, a.c_margin))
        with t("C_recognize"):
            read(gray, cb, a.c_mode)
        ms = t.ms
        for k, d, rc in (("A", "A_detect", "A_recognize"), ("B", "A_detect", "B_recognize"), ("C", "C_detect", "C_recognize")):
            per[k]["detect"].append(ms[d])
            per[k]["recognize"].append(ms[rc])
            per[k]["total"].append(ms[d] + ms[rc])
        m, n_ref, n_pred = match_rate(boxes_array(r["yolo_boxes"]), yb)
        port["box_matched"] += m
        port["node_boxes"] += n_ref
        port["py_boxes"] += n_pred
        port["text_equal"] += a_raw == r["raw"]
    sizes = {"yolo_lines.onnx": YOLO_ONNX.stat().st_size, "ppocrv6_rec.onnx": (YOLO_ONNX.parent / "ppocrv6_bubble_line_rec_webgpu.onnx").stat().st_size,
             "indra_det.onnx": (INDRA_DIR / "det.onnx").stat().st_size, "indra_rec.onnx": (Path(a.rec) if a.rec else INDRA_DIR / "rec.onnx").stat().st_size}
    report = {"split": a.split, "n": len(rows), "threads": a.threads, "settings": vars(a), "model_bytes": sizes,
              "python_port_check": {**port, "n": len(rows)},
              "ms": {k: {s: stats(v) for s, v in d.items()} for k, d in per.items()}}
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    name = f"speed_{a.split}_t{a.threads}{'_' + a.tag if a.tag else ''}.json"
    (OUT / "scores" / name).write_text(json.dumps(report, indent=1))
    print(json.dumps({"port": report["python_port_check"], **{k: {s: round(v["median"], 2) for s, v in d.items()} for k, d in report["ms"].items()}}, indent=1))


if __name__ == "__main__":
    main()
