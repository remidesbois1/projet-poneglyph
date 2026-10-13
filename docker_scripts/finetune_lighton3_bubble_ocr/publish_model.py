"""Publish a verified native bubble checkpoint; never upload crops/predictions."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi

TARGET_REPO = "Remidesbois/LightOnOCR-3-Poneglyph"


def publish(source):
    root = Path(__file__).resolve().parents[2]
    for path in (root / ".env", root / "backend/.env", root / "docker_scripts/finetune_lighton_ocr_bbox/.env"):
        load_dotenv(path)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if api.whoami()["name"] != "Remidesbois":
        raise RuntimeError("Authenticated HF account does not own this repository")
    verification = json.loads((source.parent / "model_verification.json").read_text(encoding="utf-8"))
    summary = json.loads((source.parent / "training_summary.json").read_text(encoding="utf-8"))
    config = json.loads((source.parent / "run_config.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or not verification["selected_checkpoint_bf16_cast_exact"] or not verification["fresh_native_bf16_reload"] or not verification.get("native_forward_modules_verified"):
        raise RuntimeError("Native validation/export is incomplete")
    if config["prompt"] != "" or config["revision"] != "36d8636925327c671bd8e479c517a004f7566bd9":
        raise RuntimeError("Unexpected training contract")
    names = ["model.safetensors", "config.json", "generation_config.json", "processor_config.json",
             "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md"]
    optional = ["preprocessor_config.json", "video_preprocessor_config.json", "special_tokens_map.json", "added_tokens.json", "training_curves.png"]
    names += [name for name in optional if (source / name).is_file()]
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    with (source / "model.safetensors").open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != verification["weights_sha256"]:
        raise RuntimeError("Export was modified after verification")
    metrics = json.loads((source / "benchmark_test.json").read_text(encoding="utf-8"))["metrics"]
    if metrics.get("scoring_version") != "text-v2-preserve-visible-quotes" or verification.get("scoring_version") != metrics["scoring_version"] or not metrics.get("native_forward_modules_verified"):
        raise RuntimeError("Native test and validation must use the audited quote-preserving scoring")
    if metrics["sample_count"] != 1612 or metrics["token_limit_count"] or metrics["blank_rate"]:
        raise RuntimeError("Incomplete test benchmark or invalid outputs")
    public = {"schema": 1, "model_id": TARGET_REPO, "base_model": config["model_id"], "base_revision": config["revision"],
              "weights_sha256": digest, "evaluation": "Fresh native Transformers 5.14.1 / BF16 CUDA on RTX 5090",
              "date": datetime.now(timezone.utc).date().isoformat(), "dataset": config["dataset"], "test_previously_evaluated_for_surya": True,
              "protocol": {"prompt": "", "enable_thinking": False, "max_new_tokens": 256, "do_sample": False,
                           "image_pixels_min": 65536, "image_pixels_max": 1048576,
                           "normalization": "Strip surrounding whitespace and collapse whitespace; preserve visible quotes, case, accents and punctuation",
                           "scoring_version": "text-v2-preserve-visible-quotes"},
              "selection": {"metric": "full native validation CER", "step": verification["best_step"], "cer": verification["fresh_native_bf16_validation_cer"], "historical_training_cer": verification["selection_cer"]},
              "native_verification": {k: v for k, v in verification.items() if k != "selected_checkpoint"}, "metrics": metrics}
    experiment_path = source.parent / "experiment_summary.json"
    if experiment_path.exists():
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        public["paired_surya_comparison"] = experiment["comparison"]
        public["validation_candidates"] = experiment["validation_candidates"]
        if experiment["selected_run"] != min(experiment["validation_candidates"], key=lambda c: c["native_validation_cer"])["run"]:
            raise RuntimeError("Release does not have the lowest native validation CER")
        public["training_phases"] = [{"run": phase["run"], "learning_rate": phase["config"]["training_args"]["learning_rate"],
            "batch_size": phase["config"]["training_args"]["per_device_train_batch_size"],
            "accumulation": phase["config"]["training_args"]["gradient_accumulation_steps"],
            "maximum_epochs": phase["config"]["training_args"]["num_train_epochs"],
            "eos_loss_weight": phase["config"].get("eos_loss_weight", 1.0),
            "train_seconds": phase["training_metrics"]["train_runtime"],
            "selected": phase["run"] == experiment["selected_run"]} for phase in experiment["phases"]]
    remote = api.model_info(TARGET_REPO)
    if any(s.rfilename.endswith((".safetensors", ".onnx")) for s in remote.siblings):
        raise RuntimeError("Expected the empty newly recreated bubble repository; it already contains model weights")
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(source / name)) for name in names]
    operations.append(CommitOperationAdd(path_in_repo="benchmark_native.json", path_or_fileobj=(json.dumps(public, indent=2, ensure_ascii=False)+"\n").encode("utf-8")))
    commit = api.create_commit(TARGET_REPO, operations=operations, parent_commit=remote.sha,
        commit_message="Publish native CER-selected LightOnOCR-3 0.8B bubble transcription model")
    info = api.model_info(TARGET_REPO, revision=commit.oid, files_metadata=True)
    weights = next(s for s in info.siblings if s.rfilename == "model.safetensors")
    if not weights.lfs or weights.lfs.sha256 != digest or weights.size != verification["weights_bytes"]:
        raise RuntimeError("Remote weights failed hash/size verification")
    receipt = {"repo_id": TARGET_REPO, "revision": commit.oid, "commit_url": commit.commit_url,
               "weights_sha256": digest, "verified": True, "published_files": names + ["benchmark_native.json"]}
    (source.parent / "hf_publication.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    publish(parser.parse_args().source)
