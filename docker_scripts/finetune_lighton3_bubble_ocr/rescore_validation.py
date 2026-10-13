"""Correct quote handling on saved native validation outputs, before test."""
import argparse
import json
from pathlib import Path

import train
from bubble_monitor import write_dashboard
from training_monitor import atomic_json


def rescore(directory):
    source = directory / "exported_validation_native.json"
    if not source.exists():
        source = directory / "exported_validation.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not payload["metrics"].get("native_forward_modules_verified") or len(payload["samples"]) != 1648:
        raise RuntimeError("Need a complete, verified native validation inference")
    samples = [{**s, "prediction": train.normalize_transcription(s["raw_prediction"])} for s in payload["samples"]]
    for s in samples:
        s["levenshtein"] = train.core.levenshtein_distance(s["prediction"], s["reference"])
        s["cer"] = train.core.sample_cer(s["prediction"], s["reference"])
    metrics = {**payload["metrics"], **train.core.compute_text_metrics(samples)}
    metrics["confidence_intervals"] = train.core.bootstrap_test_metrics(samples, iterations=5000)
    metrics["legacy_surya_postprocess"] = payload["metrics"]
    metrics["scoring_version"] = "text-v2-preserve-visible-quotes"
    corrected = directory / "exported_validation_corrected.json"
    train.core.save_benchmark(corrected, metrics, samples)
    new_payload = json.loads(corrected.read_text(encoding="utf-8"))
    new_payload.update(model_id="Remidesbois/LightOnOCR-3-Poneglyph", weights_source=payload.get("weights_source", str(directory / "final_merged")),
        rescored_from=source.name, inference_reused=True, no_labels_modified=True)
    atomic_json(corrected, new_payload)
    verification_path = directory / "model_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification.update(legacy_native_validation_cer=payload["metrics"]["cer"],
        fresh_native_bf16_validation_cer=metrics["cer"], fresh_native_bf16_validation_wer=metrics["wer"],
        scoring_version=metrics["scoring_version"], native_validation_benchmark=corrected.name,
        validation_rescoring="Same stored raw native outputs; visible quotes preserved; labels unchanged")
    atomic_json(verification_path, verification)
    config_path = directory / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(scoring_version=metrics["scoring_version"], training_curve_scoring_version="legacy-surya-quote-strip")
    atomic_json(config_path, config)
    summary_path = directory / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(verification=verification, best_metric=metrics["cer"], legacy_eval_cer=payload["metrics"]["cer"],
        **{"eval_"+k:v for k,v in metrics.items() if isinstance(v,(int,float))})
    atomic_json(summary_path, summary)
    history_path = directory / "metrics.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
               if json.loads(line).get("event") == "metrics"]
    write_dashboard(directory / "training_dashboard.html", summary, history)
    print(json.dumps({"run":directory.name,"legacy_cer":payload["metrics"]["cer"],"corrected_cer":metrics["cer"],
        "corrected_wer":metrics["wer"],"corrected_exact_match":metrics["exact_match"],"labels_modified":False}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories",nargs="+",type=Path)
    for directory in parser.parse_args().directories:
        rescore(directory)
