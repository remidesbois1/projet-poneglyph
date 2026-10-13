"""Produce reproducible static training figures and a compact results manifest."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(run_dir):
    run_dir = Path(run_dir)
    summary = read_json(run_dir / "training_summary.json")
    config = read_json(run_dir / "run_config.json")
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_metrics = [record for record in records if record.get("event") == "metrics"]
    metrics = []
    resumptions = []
    for record in records:
        if record.get("event") == "start":
            if record["step"]:
                resumptions.append(record["step"])
            metrics = [entry for entry in metrics if entry["step"] <= record["step"]]
        elif record.get("event") == "metrics":
            metrics.append(record)
    baseline = read_json(run_dir / "baseline_validation.json")["metrics"] if (run_dir / "baseline_validation.json").exists() else None
    evaluations = [{"step": int(path.stem.split("_")[-1]), **read_json(path)["metrics"]}
                   for path in sorted(run_dir.glob("validation_step_*.json"), key=lambda p: int(p.stem.split("_")[-1]))]
    test = read_json(run_dir / "test_benchmark.json") if (run_dir / "test_benchmark.json").exists() else None
    final_model = run_dir / "final_merged"
    result = {"status": summary["status"], "model_id": config["model_id"], "revision": config["revision"],
              "dataset": config["dataset"], "hardware": config.get("hardware"), "trainable_parameters": config.get("trainable_parameters"),
              "total_parameters": config.get("total_parameters"), "steps": summary.get("step"),
              "best_checkpoint": summary.get("best_model_checkpoint"), "best_score": summary.get("best_metric"),
              "baseline_validation": baseline, "validation_by_step": evaluations,
              "test": test["metrics"] if test else None, "test_ci95": test["bootstrap_ci95"] if test else None,
              "final_model": str(final_model) if final_model.is_dir() else None,
              "measured_peak_gib_all_attempts": max((record.get("gpu_peak_gib", 0) for record in raw_metrics), default=0),
              "measured_peak_gib_retained_steps": max((record.get("gpu_peak_gib", 0) for record in metrics), default=0),
              "current_session_peak_gib": summary.get("gpu_peak_gib"), "resumption_steps": resumptions}
    result.update(selection_metric=config.get("selection_metric", "combined_score"), init_model=config.get("init_model"),
                  bbox_loss_weight=config.get("bbox_loss_weight", 1), eos_loss_weight=config.get("eos_loss_weight", 1),
                  retained_initial_model=summary.get("retained_initial_model"), test_previously_evaluated=config.get("test_previously_evaluated", False))
    for filename in ("model_verification.json", "export_precision_comparison.json", "ocr_metrics_extended.json", "refinement_comparison.json"):
        if (run_dir / filename).exists():
            result[Path(filename).stem] = read_json(run_dir / filename)
    extended = result.get("ocr_metrics_extended")
    if extended:
        result["test"].update(extended["test"])
        result["test_ci95"].update(extended["test_ci95_added_metrics"])
        for evaluation in result["validation_by_step"]:
            evaluation.update(extended["validation"].get(f"validation_step_{evaluation['step']}", {}))
    (run_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2, "figure.facecolor": "white"})
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), constrained_layout=True)
    status_label = "Terminé" if summary["status"] == "complete" else summary["status"]
    caption = f"LightOnOCR-3 0.8B · Poneglyph · RTX 5090\n{status_label} · {summary.get('step', 0)} étapes"
    if test:
        scores = result["test"]
        caption += f"\nTest {scores['num_samples']} pages : CER {scores['cer']:.2%}"
        if "wer" in scores:
            caption += f" · WER {scores['wer']:.2%}"
        if "exact_bubble_text_rate" in scores:
            caption += f" · Bulles au texte exact {scores['exact_bubble_text_rate']:.2%}"
        caption += f" · {scores['truncated_pages']} sortie tronquée"
    fig.suptitle(caption, fontsize=17)

    def trace(ax, source, key, label, scale=1, **style):
        values = {record["step"]: record[key] for record in source if isinstance(record.get(key), (float, int))}
        if values:
            ordered = sorted(values.items())
            ax.plot([step for step, _ in ordered], [value * scale for _, value in ordered], label=label, **style)

    trace(axes[0, 0], metrics, "loss", "Train", color="#2871a6")
    trace(axes[0, 0], metrics, "eval_loss", "Validation (189 pages)", marker="o", color="#bb7429")
    loss_title = "Loss pondérée sur tokens assistant" if config.get("bbox_loss_weight", 1) != 1 else "Loss sur tokens assistant"
    axes[0, 0].set(title=loss_title, ylabel="Cross-entropy")
    validation = ([{"step": 0, **baseline}] if baseline else []) + evaluations
    trace(axes[0, 1], validation, "cer", "CER (casse et ponctuation)", marker="o", color="#2871a6")
    trace(axes[0, 1], validation, "reading_order_cer", "CER avec ordre de lecture", marker="o", color="#bb7429")
    validation_count = baseline["num_samples"] if baseline else summary.get("eval_num_samples", 48)
    axes[0, 1].set(title=f"Transcription · {validation_count} pages de validation fixes", ylabel="Taux d'erreur")
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    for key, label, color in (("f1@0_5", "F1 / IoU ≥ 0,5", "#2871a6"), ("f1@0_75", "F1 / IoU ≥ 0,75", "#bb7429"), ("mean_iou", "IoU moyen (zones manquées pénalisées)", "#23815b")):
        trace(axes[2, 0], validation, key, label, marker="o", color=color)
    axes[2, 0].set(title=f"Localisation · {validation_count} pages de validation fixes", ylabel="Score", ylim=(0, 1.03))
    for key, label, color in (("wer", "WER", "#2871a6"), ("reading_order_wer", "WER avec ordre de lecture", "#bb7429")):
        trace(axes[1, 0], validation, key, label, marker="o", color=color)
    axes[1, 0].set(title="Transcription · erreurs sur les mots", ylabel="Taux d'erreur")
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    for key, label, color in (("exact_bubble_text_rate", "Bulles au texte exact", "#2871a6"), ("page_text_exact", "Pages au texte exact / ordre", "#bb7429")):
        trace(axes[1, 1], validation, key, label, marker="o", color=color)
    axes[1, 1].set(title="Texte exact · sans égalité littérale des coordonnées", ylabel="Taux exact", ylim=(0, 1.03))
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    for key, label in (("gpu_peak_gib", "Pic alloué"), ("gpu_reserved_gib", "Réservée PyTorch"), ("gpu_device_used_gib", "Utilisée sur toute la carte")):
        trace(axes[2, 1], metrics, key, label)
    axes[2, 1].set(title="Mémoire GPU", ylabel="GiB", ylim=(0, 33))
    trace(axes[3, 0], metrics, "learning_rate", "Décodeur", scale=4)
    trace(axes[3, 0], metrics, "learning_rate", "Merger", scale=2)
    trace(axes[3, 0], metrics, "learning_rate", "Vision", scale=1)
    axes[3, 0].set(title="Learning rates · warmup puis cosine", ylabel="LR")
    axes[3, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    trace(axes[3, 1], metrics, "gpu_utilization_percent", "Activité mesurée", color="#2871a6")
    axes[3, 1].set(title="Activité GPU", ylabel="%", ylim=(0, 102))
    for ax in axes.flat:
        ax.set_xlabel("Étape optimizer (0 = point de départ)")
        for step in sorted(set(resumptions)):
            ax.axvline(step, color="gray", linestyle=":", alpha=.45)
        if ax.lines:
            ax.legend(fontsize=9, loc="best")
    fig.savefig(run_dir / "training_curves.png", dpi=180)
    fig.savefig(run_dir / "training_curves.svg")
    plt.close(fig)
    print(json.dumps({"status": result["status"], "steps": result["steps"], "best_checkpoint": result["best_checkpoint"],
                      "test": result["test"], "figure": str(run_dir / "training_curves.png")}, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    write_report(parser.parse_args().run_dir)
