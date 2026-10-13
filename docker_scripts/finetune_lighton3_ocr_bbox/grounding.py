"""LightOnOCR-3's native grounding contract and honest localization/OCR metrics."""

import re
import unicodedata

import numpy as np
from Levenshtein import distance
from scipy.optimize import linear_sum_assignment

PROMPT = "grounding"
# The native model emits either a newline or inline text after the marker.
HEADER = re.compile(r"^!\[([a-z_]+\+?)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", re.MULTILINE)
TEXT_LABELS = {"text", "title", "list", "header", "footer", "page_number", "footnote", "caption", "formula", "code", "table", "aside_text"}


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def format_grounding(items):
    blocks = []
    for item in items:
        box = list(item["bbox"])
        if len(box) != 4 or any(not isinstance(v, int) or not 0 <= v <= 1000 for v in box) or box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f"Invalid normalized bbox: {box}")
        text = normalize(item["text"])
        if not text or HEADER.search(text):
            raise ValueError("Empty text or grounding marker in annotation")
        blocks.append(f"![text]({','.join(map(str, box))})\n{text}")
    return "\n\n".join(blocks)


def parse_grounding(text):
    headers = list(HEADER.finditer(text))
    items, invalid = [], 0
    for i, match in enumerate(headers):
        label = match[1].removesuffix("+")
        box = list(map(int, match.groups()[1:]))
        content = text[match.end():headers[i + 1].start() if i + 1 < len(headers) else len(text)].strip()
        if any(v > 1000 for v in box) or box[0] >= box[2] or box[1] >= box[3] or not content:
            invalid += 1
            continue
        # Manga supervision covers annotated textual zones, not panel descriptions.
        if label in TEXT_LABELS:
            items.append({"text": normalize(content), "bbox": box, "label": label})
    marker_count = len(re.findall(r"!\[[^\]\n]+\]\(", text))
    invalid += max(0, marker_count - len(headers))
    if text.strip() and (not headers or text[:headers[0].start()].strip()):
        invalid += 1
    return items, invalid


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def assign(matrix, threshold):
    """Maximum-cardinality assignment, then maximum IoU, at each threshold."""
    if not matrix.size:
        return []
    eligible = matrix >= threshold
    score = eligible.astype(float) * (min(matrix.shape) + 1) + matrix * eligible
    rows, cols = linear_sum_assignment(-score)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if eligible[r, c]]


def page_metrics(reference, prediction, truncated=False):
    gt, gt_invalid = parse_grounding(reference)
    pred, invalid = parse_grounding(prediction)
    if gt_invalid or not gt:
        raise ValueError("Invalid grounding reference")
    matrix = np.array([[iou(g["bbox"], p["bbox"]) for p in pred] for g in gt], dtype=float).reshape(len(gt), len(pred))
    pairs = assign(matrix, 0.1)
    matched_gt, matched_pred = {g for g, _ in pairs}, {p for _, p in pairs}
    # Missing zones count as deletions; extra zones count as insertions.
    edits = sum(distance(gt[g]["text"], pred[p]["text"]) for g, p in pairs)
    edits += sum(len(g["text"]) for i, g in enumerate(gt) if i not in matched_gt)
    edits += sum(len(p["text"]) for i, p in enumerate(pred) if i not in matched_pred)
    chars = sum(len(g["text"]) for g in gt)
    words = sum(len(g["text"].split()) for g in gt)
    word_edits = sum(distance(gt[g]["text"].split(), pred[p]["text"].split()) for g, p in pairs)
    word_edits += sum(len(g["text"].split()) for i, g in enumerate(gt) if i not in matched_gt)
    word_edits += sum(len(p["text"].split()) for i, p in enumerate(pred) if i not in matched_pred)
    exact_bubbles = sum(gt[g]["text"] == pred[p]["text"] for g, p in pairs)
    detection_pairs = assign(matrix, 0.5)
    gt_plain = "\n".join(g["text"] for g in gt)
    pred_plain = "\n".join(p["text"] for p in pred)
    out = {"cer": edits / max(chars, 1), "character_edits": edits, "reference_chars": chars,
           "wer": word_edits / max(words, 1), "word_edits": word_edits, "reference_words": words,
           "reading_order_word_edits": distance(gt_plain.split(), pred_plain.split()),
           "page_text_exact": float(normalize(gt_plain) == normalize(pred_plain)),
           "exact_bubble_text_count": exact_bubbles,
           "page_text_and_boxes_exact_at_0_5": float(invalid == 0 and len(gt) == len(pred) == len(detection_pairs)
                    and all(gt[g]["text"] == pred[p]["text"] for g, p in detection_pairs)),
           "reading_order_cer": distance(gt_plain, pred_plain) / max(len(gt_plain), 1),
           "mean_iou": sum(matrix[g, p] for g, p in pairs) / len(gt),
           "num_gt": len(gt), "num_pred": len(pred), "invalid_blocks": invalid,
           "format_valid": float(invalid == 0 and bool(pred)), "truncated": int(truncated),
           "exact_match": float(reference.strip() == prediction.strip()),
           "matched_text_exact": exact_bubbles / len(gt)}
    for threshold in (0.3, 0.5, 0.75, 0.9):
        count = len(assign(matrix, threshold))
        suffix = str(threshold).replace(".", "_")
        # Invalid markers are also false positives, never free output.
        precision, recall = count / max(len(pred) + invalid, 1), count / len(gt)
        out.update({f"tp@{suffix}": count, f"precision@{suffix}": precision, f"recall@{suffix}": recall,
                    f"f1@{suffix}": 2 * precision * recall / max(precision + recall, 1e-12)})
    return out


def aggregate(rows):
    if not rows:
        raise ValueError("No pages evaluated")
    metrics = {"num_samples": len(rows)}
    metrics["cer"] = sum(r["character_edits"] for r in rows) / max(sum(r["reference_chars"] for r in rows), 1)
    if all("word_edits" in r for r in rows):
        metrics.update(wer=sum(r["word_edits"] for r in rows) / max(sum(r["reference_words"] for r in rows), 1),
                       reading_order_wer=sum(r["reading_order_word_edits"] for r in rows) / max(sum(r["reference_words"] for r in rows), 1),
                       page_text_exact=float(np.mean([r["page_text_exact"] for r in rows])),
                       page_text_and_boxes_exact_at_0_5=float(np.mean([r["page_text_and_boxes_exact_at_0_5"] for r in rows])),
                       exact_bubble_text_rate=sum(r["exact_bubble_text_count"] for r in rows) / sum(r["num_gt"] for r in rows))
    for key in ("mean_iou", "reading_order_cer", "format_valid", "exact_match", "matched_text_exact"):
        metrics[key] = float(np.mean([r[key] for r in rows]))
    metrics.update(total_gt_bubbles=sum(r["num_gt"] for r in rows), total_pred_bubbles=sum(r["num_pred"] for r in rows),
                   invalid_blocks=sum(r["invalid_blocks"] for r in rows), truncated_pages=sum(r["truncated"] for r in rows))
    for threshold in (0.3, 0.5, 0.75, 0.9):
        suffix = str(threshold).replace(".", "_")
        tp = sum(r[f"tp@{suffix}"] for r in rows)
        precision = tp / max(metrics["total_pred_bubbles"] + metrics["invalid_blocks"], 1)
        recall = tp / max(metrics["total_gt_bubbles"], 1)
        metrics.update({f"precision@{suffix}": precision, f"recall@{suffix}": recall,
                        f"f1@{suffix}": 2 * precision * recall / max(precision + recall, 1e-12)})
    metrics["combined_score"] = .4 * (1 - min(metrics["cer"], 1)) + .3 * metrics["f1@0_5"] + .2 * metrics["mean_iou"] + .1 * metrics["format_valid"]
    return metrics
