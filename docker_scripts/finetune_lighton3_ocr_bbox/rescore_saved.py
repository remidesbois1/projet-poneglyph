"""Add word/text-exact metrics and compare archived predictions without inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from grounding import aggregate, format_grounding, page_metrics, parse_grounding


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rescore(pages):
    return aggregate([page_metrics(page["reference"], page["prediction"],
                                  bool(page.get("metrics", {}).get("truncated", False))) for page in pages])


def canonical(items):
    return format_grounding([{"text": item["text"], "bbox": [round(v) for v in item["bbox"]]} for item in items])


def image_fingerprint(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return rgb.size, hashlib.sha256(rgb.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    test_path = args.run / "test_benchmark.json"
    original = read(test_path)
    test_pages = original["pages"]
    current = {str(page["page_id"]): page for page in test_pages}
    report = {"source": "Frozen saved predictions; no new inference or model selection.",
              "test_prediction_file_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
              "definitions": {
                  "cer_and_wer": "Corpus edit count / GT character or whitespace-token count. IoU>=0.1 one-to-one region assignment; missed regions are deletions, extra regions insertions. NFC/whitespace only; case/punctuation retained.",
                  "exact_bubble_text_rate": "Exact normalized text in a spatially matched region / all GT regions, corpus weighted.",
                  "matched_text_exact": "Legacy page-macro exact-region rate, denominator includes all GT regions.",
                  "page_text_exact": "All page text identical in predicted reading order; ignores coordinate numbers and whitespace layout.",
                  "page_text_and_boxes_exact_at_0_5": "All regions matched one-to-one at IoU>=0.5, all texts exact, no extra/invalid region; ignores reading order.",
                  "exact_match": "Legacy literal full serialized response equality, including bbox numbers and layout.",
                  "reading_order_wer": "Corpus word edit count on complete page texts in reading order / GT words."},
              "test": rescore(test_pages), "validation": {}, "historical": {}, "common_test_comparisons": {}}
    for path in sorted(args.run.glob("validation_step_*.json")):
        report["validation"][path.stem] = rescore(read(path)["pages"])
    nontruncated = [p for p in test_pages if not p["metrics"].get("truncated")]
    report["diagnostic_nontruncated_pages"] = {"not_the_official_test_score": True,
        "excluded_page_ids": [p["page_id"] for p in test_pages if p["metrics"].get("truncated")],
        "metrics": rescore(nontruncated)}
    page_scores = [page_metrics(p["reference"], p["prediction"], bool(p["metrics"].get("truncated"))) for p in test_pages]
    rng = np.random.default_rng(42)
    bootstrap = {key: [] for key in ("wer", "page_text_exact", "exact_bubble_text_rate")}
    for _ in range(1000):
        sample = [page_scores[i] for i in rng.integers(0, len(page_scores), len(page_scores))]
        scores = aggregate(sample)
        for key in bootstrap:
            bootstrap[key].append(scores[key])
    report["test_ci95_added_metrics"] = {key: np.quantile(values, [.025, .975]).tolist() for key, values in bootstrap.items()}
    old_paths = {
        "Surya_full_page": "docker_scripts/finetune_surya_ocr_bbox/outputs_surya_bbox/final_merged/benchmark_surya_bbox.json",
        "LightOnOCR2_full_page": "docker_scripts/finetune_lighton_ocr_bbox/outputs_lighton_bbox/final_merged/benchmark_lighton_bbox.json",
    }
    for name, relative in old_paths.items():
        old = read(args.repo / relative)
        pages = [{"page_id": row["page_id"], "reference": canonical(row["gt_items"]),
                  "prediction": canonical(row["pred_items"]) if row["pred_items"] else "",
                  "legacy_error": row.get("error"), "image_file": row.get("image_file")}
                 for row in old["results"] if row["gt_items"]]
        report["historical"][name] = {"source": relative, "source_sha256": hashlib.sha256((args.repo / relative).read_bytes()).hexdigest(),
            "published_metrics": old["metrics"], "rescored_same_evaluator": rescore(pages),
            "limitation": "Old serialized malformed blocks and truncation flags are unavailable in canonical item lists; their invalid-block counts cannot be reconstructed."}
        old_common, new_common, changed, different_images = [], [], [], []
        old_dataset = args.repo / ("docker_scripts/finetune_surya_ocr_bbox/surya_bbox_dataset" if name == "Surya_full_page"
                                   else "docker_scripts/finetune_lighton_ocr_bbox/lighton_bbox_dataset")
        for page in pages:
            fresh = current.get(str(page["page_id"]))
            if fresh is None:
                continue
            if page["reference"] != canonical(parse_grounding(fresh["reference"])[0]):
                changed.append(page["page_id"])
                continue
            old_image = old_dataset / "test" / page["image_file"]
            new_image = args.run.parent.parent / "dataset/test" / fresh["image_file"]
            if not old_image.is_file() or not new_image.is_file() or image_fingerprint(old_image) != image_fingerprint(new_image):
                different_images.append(page["page_id"])
                continue
            old_common.append(page)
            new_common.append(fresh)
        report["common_test_comparisons"][name] = {
            "condition": "Pages from both archived test benchmarks, with identical normalized GT text/boxes/order and identical RGB pixels in retained dataset files. Historical training membership is not re-certified by this rescore.",
            "pages": len(old_common), "page_ids": [p["page_id"] for p in old_common],
            "excluded_changed_annotations": changed,
            "excluded_changed_or_missing_images": different_images,
            "old": rescore(old_common) if old_common else None,
            "LightOnOCR3": rescore(new_common) if new_common else None}
    crop_relative = "docker_scripts/finetune_surya_bubble_ocr/outputs_surya_bubble_ocr/rtx5090-20261005/final_merged/benchmark_test.json"
    report["historical"]["Surya_cropped_bubbles"] = {"source": crop_relative, "published_metrics": read(args.repo / crop_relative)["metrics"],
        "limitation": "Known cropped region input: no page detection or reading-order task; not comparable to full-page OCR."}
    output = args.run / "ocr_metrics_extended.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"test": report["test"], "nontruncated_diagnostic": report["diagnostic_nontruncated_pages"],
                      "common_test_comparisons": report["common_test_comparisons"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
