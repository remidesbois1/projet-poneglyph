"""One validation-only, reproducible FP32 checkpoint interpolation candidate."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import train
from bubble_monitor import write_dashboard
from training_monitor import atomic_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(first, second, alpha):
    output = train.core.OUTPUT_DIR
    if output.exists():
        raise RuntimeError("Do not replace an existing averaging candidate")
    if not 0 < alpha < 1:
        raise ValueError("Interpolation weight must be between zero and one")
    configs = [read(path.parent / "run_config.json") for path in (first, second)]
    for config in configs:
        if config["model_id"] != train.BASE_ID or config["revision"] != train.BASE_REVISION or config["prompt"] != "":
            raise RuntimeError("Both parents must be verified bubble checkpoints")
    if configs[0]["dataset"] != configs[1]["dataset"]:
        raise RuntimeError("Parent splits differ")
    sources = [path / "model.safetensors" for path in (first, second)]
    parents = []
    for path, source in zip((first, second), sources):
        with source.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        parents.append({"run": path.parent.name, "checkpoint": path.name, "weights_sha256": digest})
    master, native = {}, {}
    with safe_open(sources[0], framework="pt", device="cpu") as left, safe_open(sources[1], framework="pt", device="cpu") as right:
        if set(left.keys()) != set(right.keys()):
            raise RuntimeError("Parent parameter sets differ")
        for key in left.keys():
            a, b = left.get_tensor(key), right.get_tensor(key)
            if a.shape != b.shape or a.dtype != b.dtype or a.dtype != torch.float32:
                raise RuntimeError(f"Expected identically shaped FP32 masters: {key}")
            value = torch.lerp(a, b, alpha)
            if not torch.isfinite(value).all():
                raise FloatingPointError(key)
            master[key] = value.contiguous()
            native[key] = value.to(torch.bfloat16)
    if len(master) != 473:
        raise RuntimeError("Unexpected native parameter count")
    checkpoint = output / "checkpoint-average"
    checkpoint.mkdir(parents=True)
    train.core.FINAL_DIR.mkdir()
    save_file(master, checkpoint / "model.safetensors", metadata={"format": "pt"})
    save_file(native, train.core.FINAL_DIR / "model.safetensors", metadata={"format": "pt"})
    # Native config/processor are unchanged. Each saved tensor is checked against
    # the persisted FP32 interpolation, not just its in-memory cast.
    metadata_source = first.parent / "final_merged"
    for source in metadata_source.iterdir():
        if source.suffix in {".json", ".jinja"} and not source.name.startswith("benchmark"):
            shutil.copyfile(source, train.core.FINAL_DIR / source.name)
            shutil.copyfile(source, checkpoint / source.name)
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as source, safe_open(train.core.FINAL_DIR / "model.safetensors", framework="pt", device="cpu") as target:
        for key in source.keys():
            if not torch.equal(source.get_tensor(key).to(torch.bfloat16), target.get_tensor(key)):
                raise RuntimeError(f"Saved native cast differs: {key}")
    with (train.core.FINAL_DIR / "model.safetensors").open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    verification = {"selected_checkpoint": str(checkpoint), "selection_metric": "full native validation CER",
        "selection_cer": None, "best_step": 0, "all_native_tensors_verified": len(master),
        "selected_checkpoint_bf16_cast_exact": True, "weights_sha256": digest,
        "weights_bytes": (train.core.FINAL_DIR / "model.safetensors").stat().st_size,
        "checkpoint_average": {"method": "FP32 torch.lerp then BF16 cast", "second_parent_weight": alpha, "parents": parents}}
    atomic_json(output / "model_verification.json", verification)
    config = {**configs[0], "init_model": "FP32 interpolation of two bubble checkpoints",
        "checkpoint_average": verification["checkpoint_average"], "selection": "Full native BF16 validation CER; test remains closed"}
    atomic_json(output / "run_config.json", config)
    shutil.copyfile(first.parent / "dataset_audit.json", output / "dataset_audit.json")
    shutil.copyfile(first.parent / "baseline_validation.json", output / "baseline_validation.json")
    print(json.dumps({"candidate": output.name, "tensors_verified": len(master), "parents": parents, "second_parent_weight": alpha}), flush=True)


def validate():
    train.core.configure_torch_runtime()
    output = train.core.OUTPUT_DIR
    verification = read(output / "model_verification.json")
    metrics, _ = train.benchmark(str(train.core.FINAL_DIR), "val", output / "exported_validation_native.json")
    verification.update(selection_cer=metrics["cer"], fresh_native_bf16_validation_cer=metrics["cer"],
        fresh_native_bf16_validation_wer=metrics["wer"], fresh_native_bf16_reload=True,
        native_forward_modules_verified=metrics["native_forward_modules_verified"], separate_process_native_validation=True)
    atomic_json(output / "model_verification.json", verification)
    summary = {"status": "validation_complete", "verification": verification, "best_metric": metrics["cer"],
        "best_model_checkpoint": verification["selected_checkpoint"], "step": 0, "max_steps": 0,
        "dataset": read(output / "run_config.json")["dataset"],
        **{"eval_" + k: v for k, v in metrics.items() if isinstance(v, (int, float))}}
    atomic_json(output / "training_summary.json", summary)
    (output / "metrics.jsonl").write_text(json.dumps({"event": "metrics", "step": 0, **summary})+"\n", encoding="utf-8")
    write_dashboard(output / "training_dashboard.html", summary, [])
    print(json.dumps({"candidate": output.name, "native_validation_cer": metrics["cer"], "native_validation_wer": metrics["wer"], "exact_match": metrics["exact_match"], "token_limit_count": metrics["token_limit_count"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "validate"), required=True)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--second-weight", type=float, default=.25)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.first, args.second, args.second_weight)
    else:
        validate()
