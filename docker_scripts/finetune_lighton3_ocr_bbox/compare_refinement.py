"""Compare frozen OCR predictions with paired page bootstrap, without inference."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from grounding import aggregate, page_metrics


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(before_path, after_path):
    before, after = load(before_path), load(after_path)
    old_pages = {str(p["page_id"]): p for p in before["pages"]}
    new_pages = {str(p["page_id"]): p for p in after["pages"]}
    if len(old_pages) != len(before["pages"]) or len(new_pages) != len(after["pages"]):
        raise ValueError("Duplicate page IDs in evaluation")
    if old_pages.keys() != new_pages.keys():
        raise ValueError("Comparisons require exactly the same pages")
    old_scores, new_scores, changes = [], [], []
    for page_id in old_pages:
        old, new = old_pages[page_id], new_pages[page_id]
        if old["reference"] != new["reference"] or old["image_file"] != new["image_file"]:
            raise ValueError(f"Reference/image mismatch on page {page_id}")
        a = page_metrics(old["reference"], old["prediction"], bool(old["metrics"].get("truncated")))
        b = page_metrics(new["reference"], new["prediction"], bool(new["metrics"].get("truncated")))
        old_scores.append(a)
        new_scores.append(b)
        changes.append({"page_id": page_id, "reference_chars": a["reference_chars"],
                        "old_character_edits": a["character_edits"], "new_character_edits": b["character_edits"],
                        "character_edit_delta": b["character_edits"] - a["character_edits"],
                        "old_cer": a["cer"], "new_cer": b["cer"],
                        "old_truncated": bool(a["truncated"]), "new_truncated": bool(b["truncated"])})
    old_metrics, new_metrics = aggregate(old_scores), aggregate(new_scores)
    # Resample the SAME page indices for both models. Ratios remain corpus weighted.
    samples = np.random.default_rng(42).integers(0, len(changes), (5000, len(changes)))
    bootstrap = {}
    for key, numerator, denominator in (("cer", "character_edits", "reference_chars"),
                                         ("wer", "word_edits", "reference_words")):
        a = np.array([s[numerator] for s in old_scores])
        b = np.array([s[numerator] for s in new_scores])
        d = np.array([s[denominator] for s in old_scores])
        deltas = (b[samples].sum(axis=1) - a[samples].sum(axis=1)) / d[samples].sum(axis=1)
        bootstrap[key] = {"delta_after_minus_before": new_metrics[key] - old_metrics[key],
                          "paired_page_bootstrap_ci95": np.quantile(deltas, [.025, .975]).tolist()}
    deltas = [p["character_edit_delta"] for p in changes]
    return {"before_file": str(before_path), "after_file": str(after_path),
            "before_sha256": hashlib.sha256(before_path.read_bytes()).hexdigest(),
            "after_sha256": hashlib.sha256(after_path.read_bytes()).hexdigest(),
            "pages": len(changes), "before": old_metrics, "after": new_metrics,
            "paired_bootstrap": bootstrap,
            "pages_improved_cer": sum(d < 0 for d in deltas),
            "pages_unchanged_cer": sum(d == 0 for d in deltas),
            "pages_regressed_cer": sum(d > 0 for d in deltas),
            "character_edits_saved": -sum(deltas), "page_changes": changes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    config = load(args.run / "run_config.json")
    parent_config = load(args.parent / "run_config.json")
    for split in ("train", "val", "test"):
        if config["dataset"][split]["metadata_sha256"] != parent_config["dataset"][split]["metadata_sha256"]:
            raise ValueError(f"Frozen dataset metadata changed for {split}")
    summary = load(args.run / "training_summary.json")
    report = {"source": "Frozen predictions only; no new inference, training or checkpoint selection.",
              "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "dataset_metadata_sha256": {split: config["dataset"][split]["metadata_sha256"] for split in ("train", "val", "test")},
              "metric_note": "Corpus CER/WER penalize missing/extra regions; NFC/whitespace normalization; case/punctuation retained.",
              "bootstrap_note": "5000 paired page resamples, seed 42. Negative delta favors refinement. Validation intervals are descriptive because checkpoints are selected on this validation set.",
              "test_previously_evaluated": bool(config.get("test_previously_evaluated")),
              "validation_by_step": {}, "test": None}
    baseline = args.run / "baseline_validation.json"
    paths = sorted(args.run.glob("validation_step_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    for path in paths:
        report["validation_by_step"][path.stem] = compare(baseline, path)
    if summary["status"] == "complete" and (args.run / "test_benchmark.json").exists():
        report["test"] = compare(args.parent / "test_benchmark.json", args.run / "test_benchmark.json")
    (args.run / "refinement_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not paths:
        return
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    steps = [0] + [int(p.stem.split("_")[-1]) for p in paths]
    scores = [load(baseline)["metrics"]] + [load(p)["metrics"] for p in paths]
    for ax, key, title in ((axes[0, 0], "cer", "CER de validation · 189 pages"),
                            (axes[0, 1], "wer", "WER de validation · 189 pages")):
        ax.plot(steps, [s[key] for s in scores], "o-", color="#2871a6")
        ax.set(title=title, xlabel="Étape de consolidation", ylabel="Taux d'erreur")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax = axes[1, 0]
    for key, label in (("exact_bubble_text_rate", "Bulles au texte exact"), ("page_text_exact", "Pages au texte exact")):
        ax.plot(steps, [s[key] for s in scores], "o-", label=label)
    ax.set(title="Texte exact · validation", xlabel="Étape de consolidation", ylabel="Taux exact")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    if summary.get("retained_initial_model"):
        chosen = None
    else:
        checkpoint = Path(summary.get("best_model_checkpoint") or paths[-1].stem)
        chosen = report["validation_by_step"].get("validation_step_" + checkpoint.name.split("-")[-1])
    chosen = chosen or report["validation_by_step"][paths[-1].stem]
    ax = axes[1, 1]
    counts = [chosen["pages_improved_cer"], chosen["pages_unchanged_cer"], chosen["pages_regressed_cer"]]
    if summary.get("retained_initial_model"):
        counts = [0, chosen["pages"], 0]
    bars = ax.bar(["Améliorées", "Identiques", "Dégradées"], counts, color=["#23815b", "#718096", "#bb7429"])
    ax.bar_label(bars)
    ax.set(title="CER page par page · candidat sélectionné / départ", ylabel="Nombre de pages")
    fig.suptitle("LightOnOCR-3 0.8B · consolidation orientée CER\nMême validation, même prompt grounding, mêmes images", fontsize=16)
    fig.savefig(args.run / "ocr_refinement_comparison.png", dpi=180)
    fig.savefig(args.run / "ocr_refinement_comparison.svg")
    plt.close(fig)
    compact = {name: {k: v for k, v in value.items() if k in ("before", "after", "paired_bootstrap", "pages_improved_cer", "pages_unchanged_cer", "pages_regressed_cer", "character_edits_saved")}
               for name, value in report["validation_by_step"].items()}
    print(json.dumps({"validation": compact, "test": {k: v for k, v in report["test"].items() if k != "page_changes"} if report["test"] else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
