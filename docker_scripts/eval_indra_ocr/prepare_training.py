"""Package the recognizer fine-tuning inputs (train + val only; test never leaves this machine).

- Indra det.onnx line boxes for every train/val bubble, with the settings frozen on validation
- crops, production YOLO boxes and references (outputs/ppocr_<split>.jsonl)
- rec.onnx + meta.json (Indra v1.1.0), and the training code
- an OOV report: reference characters missing from Indra's Latin vocabulary

  python prepare_training.py --det-scale 1.0 --det-margin 0
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

from indra_lib import HERE, INDRA_DIR, OUT, IndraDet, IndraRec, load_rows, oov, read_gray, sort_line_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-scale", type=float, required=True)
    ap.add_argument("--det-margin", type=int, default=0)
    a = ap.parse_args()
    det, vocab = IndraDet(8), IndraRec(1).vocab
    missing = collections.Counter()
    files = []
    for split in ("train", "val"):
        rows = load_rows(split)
        path = OUT / f"indra_boxes_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                b = sort_line_boxes(det.detect(read_gray(r["crop"]), a.det_scale, a.det_margin))
                f.write(json.dumps({"bubble_id": r["bubble_id"], "boxes": np.round(b, 2).tolist()}) + "\n")
                missing.update(oov(vocab, r["reference"]))
                files.append(r["crop"])
        files += [f"ppocr_{split}.jsonl", path.name]
    report = {"det": vars(a), "oov_chars": dict(missing.most_common())}
    (OUT / "training_inputs.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    archive = OUT / "indra_rec_training.tar"
    with tarfile.open(archive, "w") as tar:
        for rel in files:
            tar.add(OUT / rel, arcname=f"data/{rel}")
        for name in ("rec.onnx", "det.onnx", "meta.json"):
            tar.add(INDRA_DIR / name, arcname=f"data/models/{name}")
        for name in ("train_rec.py", "train_det.py", "onnx_torch.py", "requirements-train.txt", "run_training.sh", "run_det_training.sh"):
            tar.add(HERE / name, arcname=f"code/{name}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (OUT / "indra_rec_training.manifest.json").write_text(json.dumps({"sha256": digest, "files": len(files) + 6}))
    print(archive, archive.stat().st_size, digest)


if __name__ == "__main__":
    main()
