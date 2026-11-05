"""Zero-shot Indra settings, chosen on validation only (test stays untouched).

Writes outputs/predictions/val/<name>.jsonl; score with `node score.cjs --split val`.
Also reports how Indra's line boxes agree with the production YOLO boxes (IoU >= 0.5),
a diagnostic only: YOLO boxes are not ground truth.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from evaluate import Config, full_indra, predict, write
from indra_lib import OUT, IndraDet, IndraRec, boxes_array, load_rows, match_rate

GRID: dict[str, Config] = {
    "B_lines": Config(det="yolo", mode="lines"),
    "B_lines_nodehyph": Config(det="yolo", mode="lines", dehyphenate=False),
    "B_lines_shrink10": Config(det="yolo", mode="lines", grow_y=-0.10),
    "B_lines_shrink20": Config(det="yolo", mode="lines", grow_y=-0.20),
    "B_lines_grow10": Config(det="yolo", mode="lines", grow_y=0.10),
    "B_lines_lm": Config(det="yolo", mode="lines", lm=True),
    "B_stitched_gap0": Config(det="yolo", mode="stitched", gap=0),
    "B_stitched_gap16": Config(det="yolo", mode="stitched", gap=16),
    **{f"C_lines_s{s}_m{m}": Config(det="indra", det_scale=s, det_margin=m, mode="lines")
       for s in (0.5, 0.75, 1.0, 1.5, 2.0) for m in (0, 16)},
}


def main():
    only = set(sys.argv[1:])
    rows = load_rows("val")
    recs = {False: IndraRec(4), True: IndraRec(4, lm=True)}
    det = IndraDet(4)
    agreement = {}
    for name, cfg in GRID.items():
        if only and name not in only:
            continue
        preds = predict(rows, cfg, det, recs[cfg.lm])
        write(name, "val", cfg.__dict__, preds)
        if cfg.det == "indra":
            m = np.array([match_rate(boxes_array(r["yolo_boxes"]), np.array(p["boxes"]).reshape(-1, 4)[: p["lines"]])
                          for r, p in zip(rows, preds)])
            agreement[name] = {"matched": int(m[:, 0].sum()), "yolo_boxes": int(m[:, 1].sum()), "indra_boxes": int(m[:, 2].sum()),
                               "zero_line_bubbles": sum(p["lines"] == 0 for p in preds)}
            print(name, agreement[name], flush=True)
        else:
            print(name, "done", flush=True)
    if not only or "C0_full_indra" in only:
        write("C0_full_indra", "val", {"full_indra": True}, full_indra(rows))
    if agreement:
        (OUT / "scores").mkdir(parents=True, exist_ok=True)
        (OUT / "scores/val_box_agreement.json").write_text(json.dumps(agreement, indent=1))


if __name__ == "__main__":
    main()
