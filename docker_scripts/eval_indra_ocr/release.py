"""Build (and optionally publish) the browser release of the Poneglyph Indra bubble OCR.

  python release.py build                  # outputs/release/ (model card, ONNX, manifest, metrics)
  python release.py publish                # upload outputs/release/ to Hugging Face, verify downloads

The release holds exactly the two networks selected on validation, the browser
manifest (vocabulary, preprocessing constants, train-label text rules), Indra's
license and the evaluation report.  Nothing from outputs/ other than the listed
files is uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from indra_lib import HERE, INDRA_DIR, OUT

REPO = "Remidesbois/indra-ocr-poneglyph-bubble"
REL = OUT / "release"
REC = OUT / "ft/runs/r2_lr3e-4/rec_best.onnx"
DET = OUT / "ft/det_runs/d1_lr1e-3/det_last.onnx"
PPOCR_MANIFEST = HERE.parent / "retrain_ppocr_pipeline/outputs/models-published/browser_manifest.json"
BASE_REVISION = "a1283c4b42a273c09480ad50e71461daff50fcc7"  # datalab-to/indra-ocr tag v1.1.0
DET_SCALE = 0.75


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict:
    return json.loads((OUT / "scores" / name).read_text(encoding="utf-8"))


def metrics() -> dict:
    test, val = load("test_final.json"), load("val_finetune_det.json")
    boot = load("test_bootstrap.json")
    speed = {t: load(f"speed_test_t{t}_final.json") for t in (4, 1)}
    pick = lambda run: {k: v for k, v in run.items() if k not in ("per_bubble",)}
    names = {"A_yolo_ppocrv6": "A: YOLO lines + PP-OCRv6 (production)",
             "FT_B_yolo_r2best": "B: YOLO lines + Indra rec (fine-tuned)",
             "FT_C_indra_s0.75_r2best": "C: Indra det (original) + Indra rec (fine-tuned)",
             "FTD_C_indra_d1last_r2best": "C: Indra det (fine-tuned) + Indra rec (fine-tuned) - this release",
             "B_zeroshot_yolo_stitched": "B zero-shot: YOLO lines + Indra rec v1.1.0",
             "C_zeroshot_indra_s0.75_stitched": "C zero-shot: Indra det + rec v1.1.0",
             "C0_zeroshot_full_indra": "indra_ocr 1.1.2 full pipeline on the crop"}
    return {
        "date": "2026-11-05",
        "dataset": "Poneglyph validated reference bubble crops (One Piece, French), page-level splits of dataset-v5",
        "splits": {"train": 9256, "val": 1193, "test": 1221},
        "selection": "Every setting (detector scale, recognizer variant and epoch, detector variant and epoch) chosen on val only; test scored once per frozen configuration.",
        "scoring": "Production postprocessPpocrText (train-label lexicon, spacing, sentence case) applied to every configuration; then NFKC, straight apostrophes, expanded ellipses, collapsed whitespace; case and punctuation kept (docker_scripts/audit_ppocr_pipeline/runtime.cjs score).",
        "test": {names[k]: pick(v) for k, v in test["runs"].items() if k in names},
        "val_selected": pick(val["runs"]["FTD_C_d1_lr1e-3_last"]),
        "test_bootstrap_vs_production": boot,
        "browser_implementation_test": {
            "implementation": "frontend/src/lib/indraOcr.js (Poneglyph web worker code) on onnxruntime-node 1.24.3 CPU",
            "result": pick(load("test_js_parity.json")["runs"]["JS_indra_release"]),
            "identical_raw_texts_vs_python": {"val": "1193/1193", "test": "1219/1221"},
        },
        "cpu_latency_ms": {f"{t}_threads": {"pipelines": s["ms"], "n": s["n"], "note": "median/mean/p90/p99 per bubble over the first 300 test bubbles; Python + onnxruntime 1.31.1 CPU (Windows, 20 logical cores, idle machine), warm sessions, one bubble at a time; A and B use a Python port of the production YOLO/PP-OCR worker"}
                           for t, s in speed.items()},
        "training": {
            "recognizer": {"source": "rec.onnx of datalab-to/indra-ocr v1.1.0", "data": "bubble crops; production YOLO line boxes (60%) or Indra det boxes (40%), box jitter, photometric augmentation; line windows side by side, CTC on the whole bubble transcription",
                           "optimizer": "AdamW lr 3e-4, wd 0.01, warmup 300, cosine, batch 48, 40 epochs, EMA 0.999", "selected": "epoch 27 (best val CER)",
                           "hardware": "1x NVIDIA RTX PRO 6000 Blackwell (RunPod), 5.3 min"},
            "detector": {"source": "det.onnx of datalab-to/indra-ocr v1.1.0", "labels": "pseudo-labels: production YOLO26n line boxes (conf 0.10, line NMS 0.75); bubbles without YOLO lines and 28 reviewed YOLO omissions excluded",
                         "targets": "indra_ocr.det_post.encode_targets (core band + top/bottom distances)", "optimizer": "AdamW lr 1e-3, wd 0.01, warmup 200, cosine, batch 32, 30 epochs, EMA 0.999, input scale 0.75 jittered x0.8-1.25",
                         "selected": "last epoch (best val end-to-end CER)", "hardware": "1x NVIDIA RTX PRO 6000 Blackwell (RunPod), 3.6 min"},
        },
        "limits": ["Historical val/test pages were inspected in earlier audits: not a blind benchmark.",
                   "Detector labels are YOLO pseudo-labels, not human line boxes.",
                   "Short texts (<= 12 characters: SFX, lone punctuation) remain weaker than PP-OCRv6."],
    }


def card(m: dict, files: dict) -> str:
    t = m["test"]
    row = lambda k: t[k]["final"]
    lines = [
        "---", "license: other", "license_name: indra-ocr-openrail-m", "license_link: LICENSE", "language:", "- fr",
        "base_model: datalab-to/indra-ocr", "pipeline_tag: image-to-text", "library_name: onnx", "tags:", "- ocr", "- onnx", "- manga", "- indra-ocr", "---", "",
        "# Indra OCR Poneglyph - bubble OCR",
        "",
        "Fine-tune of [Datalab's Indra OCR](https://huggingface.co/datalab-to/indra-ocr) (tag `v1.1.0`) for French One Piece speech bubbles, as used by "
        "[Poneglyph](https://poneglyph.fr) in the browser. Two networks, unchanged in structure from Indra:",
        "",
        f"- `onnx/indra_det_bubble.onnx` ({files['det']['bytes'] / 1e3:.0f} kB): line detector, input a bubble crop at {DET_SCALE} x its size.",
        f"- `onnx/indra_rec_bubble.onnx` ({files['rec']['bytes'] / 1e6:.1f} MB): CTC recognizer, input the bubble's line windows side by side (one pass per bubble).",
        "- `onnx/browser_manifest.json`: vocabulary, preprocessing constants and the text rules learned from training labels.",
        "",
        "## Test results (1,221 bubbles, page-level holdout)",
        "",
        "| Pipeline | CER | Exact match |", "|---|---:|---:|",
    ]
    for k in t:
        if "zero-shot" in k or "full pipeline" in k:
            continue
        lines.append(f"| {k} | {100 * row(k)['cer']:.3f} % | {100 * row(k)['exact']:.2f} % |")
    b = m["test_bootstrap_vs_production"]["runs"]["FTD_C_indra_d1last_r2best"]
    s4 = m["cpu_latency_ms"]["4_threads"]["pipelines"]
    lines += [
        "",
        f"This release vs production: {b['errors']} vs {b['errors_A']} character errors, CER difference {b['delta_cer_pts']:+.3f} points "
        f"(95% paired page bootstrap {b['ci95'][0]:+.3f} to {b['ci95'][1]:+.3f}).",
        f"Median CPU latency per bubble (4 threads): {s4['C']['total']['median']:.1f} ms vs {s4['A']['total']['median']:.1f} ms for YOLO + PP-OCRv6.",
        f"The browser implementation (Poneglyph `frontend/src/lib/indraOcr.js`, ONNX Runtime) scores {100 * m['browser_implementation_test']['result']['final']['cer']:.3f} % CER on the same test "
        f"({m['browser_implementation_test']['identical_raw_texts_vs_python']['test']} raw texts identical to the Python reference).",
        "",
        "Selection was done on validation only. The test pages were inspected in earlier audits, so this is not a blind benchmark; "
        "detector labels are YOLO pseudo-labels. Short texts (sound effects, lone punctuation) remain weaker than PP-OCRv6. "
        "Full protocol, training settings and latency: `metrics/evaluation_20261105.json`.",
        "",
        "## Pipeline",
        "",
        "1. Grayscale bubble crop; `det` input = ink map (1 - gray/255) resized to long side round(0.75 x crop long side), padded to multiples of 32.",
        "2. Line boxes from the core/distance maps (indra_ocr `det_post.decode`), sorted by half-line bands then x.",
        "3. Each line is cropped with Indra's `crop_line` (tall window x3, height 96) and the windows are concatenated; no line found: the whole crop is one line.",
        "4. `rec` greedy CTC, then the manifest's `postprocess` rules (lexicon, French punctuation spacing, sentence case).",
        "",
        "Reference implementation: Python `docker_scripts/eval_indra_ocr` and browser `frontend/src/lib/indraOcr.js` in the Poneglyph repository.",
        "",
        "## License",
        "",
        "These weights are derived from Indra OCR and keep its modified OpenRAIL-M license (see `LICENSE`), including its use restrictions. "
        "Indra OCR is by [Datalab](https://www.datalab.to).",
    ]
    return "\n".join(lines) + "\n"


def build():
    if REL.exists():
        shutil.rmtree(REL)
    (REL / "onnx").mkdir(parents=True)
    (REL / "metrics").mkdir()
    shutil.copy(DET, REL / "onnx/indra_det_bubble.onnx")
    shutil.copy(REC, REL / "onnx/indra_rec_bubble.onnx")
    shutil.copy(INDRA_DIR / "LICENSE", REL / "LICENSE")
    meta = json.loads((INDRA_DIR / "meta.json").read_text(encoding="utf-8"))
    pp = json.loads(PPOCR_MANIFEST.read_text(encoding="utf-8"))["postprocess"]
    files = {k: {"path": f"onnx/indra_{k}_bubble.onnx", "sha256": sha(REL / f"onnx/indra_{k}_bubble.onnx"),
                 "bytes": (REL / f"onnx/indra_{k}_bubble.onnx").stat().st_size} for k in ("det", "rec")}
    manifest = {
        "name": "indra-ocr-poneglyph-bubble", "base_model": "datalab-to/indra-ocr", "base_revision": BASE_REVISION,
        "files": files, "det_scale": DET_SCALE, "rec_height": meta["rec_height"], "rec_window": meta["rec_window"],
        "blank_token_id": 0, "vocab": json.loads(meta["vocab"])["chars"],
        "postprocess": pp, "postprocess_source": "Remidesbois/pp-ocrv6-one-piece-bubble-line-rec a6dddec (rules built from training labels only)",
    }
    (REL / "onnx/browser_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    m = metrics()
    m["files"] = files
    (REL / "metrics/evaluation_20261105.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    (REL / "README.md").write_text(card(m, files), encoding="utf-8")
    print(json.dumps({p.relative_to(REL).as_posix(): p.stat().st_size for p in sorted(REL.rglob("*")) if p.is_file()}, indent=1))


def publish():
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    files = sorted(p for p in REL.rglob("*") if p.is_file())
    from huggingface_hub import CommitOperationAdd

    commit = api.create_commit(REPO, operations=[CommitOperationAdd(p.relative_to(REL).as_posix(), str(p)) for p in files],
                               commit_message="Indra OCR Poneglyph bubble release 2026-11-05")
    revision = commit.oid
    checks = {}
    for p in files:
        rel = p.relative_to(REL).as_posix()
        got = Path(hf_hub_download(REPO, rel, revision=revision, local_dir=OUT / "release-download"))
        checks[rel] = sha(got) == sha(p)
    out = {"repository": REPO, "revision": revision, "url": f"https://huggingface.co/{REPO}", "downloads_match": checks}
    (OUT / "release-published.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    if not all(checks.values()):
        raise SystemExit("downloaded files differ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "publish"])
    {"build": build, "publish": publish}[ap.parse_args().action]()
