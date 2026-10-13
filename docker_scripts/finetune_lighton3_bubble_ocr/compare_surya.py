"""Paired, page-cluster comparison on exactly the same frozen test crops."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from Levenshtein import distance
import train


def verify_scores(payload):
    samples = payload["samples"]
    characters = sum(len(s["reference"]) for s in samples)
    edits = sum(distance(s["reference"], s["prediction"]) for s in samples)
    exact = sum(s["reference"] == s["prediction"] for s in samples)
    word_errors = sum(distance(s["reference"].split(), s["prediction"].split()) for s in samples)
    words = sum(len(s["reference"].split()) for s in samples)
    for key, value in (("cer", edits/characters), ("wer", word_errors/words), ("exact_match", exact/len(samples))):
        if not math.isclose(payload["metrics"][key], value, abs_tol=1e-12):
            raise ValueError(f"Stored {key} differs from independent edit calculation")


def compare(new_path, surya_path):
    new = json.loads(new_path.read_text(encoding="utf-8"))
    old = json.loads(surya_path.read_text(encoding="utf-8"))
    verify_scores(new)
    verify_scores(old)
    legacy_old = old["metrics"]
    # The historical Surya report stripped quotes from predictions only. Keep
    # that score, and compare both models' actual stored raw outputs fairly.
    old["samples"] = [{**s, "prediction": train.normalize_transcription(s["raw_prediction"])} for s in old["samples"]]
    for sample in old["samples"]:
        sample["levenshtein"] = distance(sample["reference"], sample["prediction"])
        sample["cer"] = train.core.sample_cer(sample["prediction"], sample["reference"])
    old["metrics"] = {**legacy_old, **train.core.compute_text_metrics(old["samples"]),
        "confidence_intervals": train.core.bootstrap_test_metrics(old["samples"], iterations=5000),
        "scoring_version": "text-v2-preserve-visible-quotes"}
    fresh = {str(s["id"]): s for s in new["samples"]}
    previous = {str(s["id"]): s for s in old["samples"]}
    if set(fresh) != set(previous) or len(fresh) != 1612 or len(new["samples"]) != len(fresh) or len(old["samples"]) != len(previous):
        raise ValueError("The comparison requires the identical 1612 annotation IDs")
    pages = {}
    wins, losses, ties = 0, 0, 0
    for identity, sample in fresh.items():
        before = previous[identity]
        if sample["reference"] != before["reference"] or str(sample["page_id"]) != str(before["page_id"]):
            raise ValueError(f"Reference/page differs for {identity}")
        reference = sample["reference"]
        a, b = distance(reference, before["prediction"]), distance(reference, sample["prediction"])
        wins += b < a
        losses += b > a
        ties += b == a
        values = pages.setdefault(str(sample["page_id"]), np.zeros(6))
        values += (a, b, len(reference), before["prediction"] == reference, sample["prediction"] == reference, 1)
    matrix = np.array([pages[key] for key in sorted(pages)])
    rng = np.random.default_rng(42)
    draws = matrix[rng.integers(0, len(matrix), size=(5000, len(matrix)))].sum(axis=1)
    difference = (draws[:, 1] - draws[:, 0]) / draws[:, 2]
    exact_difference = (draws[:, 4] - draws[:, 3]) / draws[:, 5]
    totals = matrix.sum(axis=0)
    for payload, edits, exact in ((old, totals[0], totals[3]), (new, totals[1], totals[4])):
        if not math.isclose(payload["metrics"]["cer"], edits / totals[2], abs_tol=1e-12):
            raise ValueError("Stored CER does not match independently recomputed edits")
        if not math.isclose(payload["metrics"]["exact_match"], exact / totals[5], abs_tol=1e-12):
            raise ValueError("Stored exact match does not match the predictions")
        word_errors = sum(distance(s["reference"].split(), s["prediction"].split()) for s in payload["samples"])
        reference_words = sum(len(s["reference"].split()) for s in payload["samples"])
        if not math.isclose(payload["metrics"]["wer"], word_errors / reference_words, abs_tol=1e-12):
            raise ValueError("Stored WER does not match independently recomputed word edits")
    return {"schema": 2, "protocol": "Identical frozen crops/references; preserve visible quotes, case, accents and punctuation; normalize whitespace only; paired page-cluster bootstrap",
            "sample_count": len(fresh), "page_count": len(pages), "reference_characters": int(totals[2]),
            "surya": old["metrics"], "lighton3": new["metrics"],
            "historical_quote_stripping": {"surya": legacy_old, "lighton3": new["metrics"]["legacy_surya_postprocess"]},
            "surya_character_errors": int(totals[0]), "lighton3_character_errors": int(totals[1]),
            "relative_cer_reduction": 1 - totals[1] / totals[0],
            "new_minus_surya_cer_95": np.percentile(difference, [2.5, 97.5]).tolist(),
            "new_minus_surya_exact_match_95": np.percentile(exact_difference, [2.5, 97.5]).tolist(),
            "bootstrap_iterations": 5000, "fewer_edits_crops": wins, "more_edits_crops": losses, "equal_edits_crops": ties}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--surya", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.new, args.surya)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"surya", "lighton3", "historical_quote_stripping"}} |
        {"surya_cer": report["surya"]["cer"], "lighton3_cer": report["lighton3"]["cer"],
         "surya_wer": report["surya"]["wer"], "lighton3_wer": report["lighton3"]["wer"],
         "surya_exact_match": report["surya"]["exact_match"], "lighton3_exact_match": report["lighton3"]["exact_match"]}, indent=2))
