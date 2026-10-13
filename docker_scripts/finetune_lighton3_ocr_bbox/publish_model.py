"""Publish the selected native checkpoint and aggregate metrics, never pages or predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi


def publish(source, repo_id):
    root = Path(__file__).resolve().parents[2]
    for path in [root / ".env", root / "backend/.env", Path(__file__).parent / ".env",
                 root / "docker_scripts/finetune_lighton_ocr_bbox/.env"]:
        load_dotenv(path)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if api.whoami()["name"] != repo_id.split("/")[0]:
        raise RuntimeError("Authenticated account does not own the requested repository")
    names = ["model.safetensors", "config.json", "generation_config.json", "processor_config.json",
             "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md"]
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    with (source / "model.safetensors").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    verification = json.loads((source.parent / "model_verification.json").read_text(encoding="utf-8"))
    if digest != verification["export"]["sha256"] or not verification["export"]["selected_checkpoint_bf16_cast_exact"]:
        raise RuntimeError("Weights differ from the verified CER-selected checkpoint")
    summary = json.loads((source.parent / "training_summary.json").read_text(encoding="utf-8"))
    metrics = {key.removeprefix("test_"): value for key, value in summary.items() if key.startswith("test_")}
    public = {"schema": 1, "model_id": repo_id, "weights_sha256": digest,
              "evaluation_runtime": "Transformers 5.14.1, BF16 CUDA, native model (not ONNX/WebGPU)",
              "hardware": "NVIDIA GeForce RTX 5090 32 GB", "date": "2026-10-12",
              "dataset": "Poneglyph validated full pages, frozen 882/189/189 train/validation/test split",
              "split": "test", "test_previously_inspected": True,
              "selection": {"metric": "validation CER", "step": 111, "value": summary["best_metric"]},
              "protocol": "grounding prompt, thinking disabled, greedy 1280 tokens, maximum image long edge 2048 px; NFC and normalized whitespace, case and punctuation preserved; one-to-one spatial assignment at IoU >= 0.1, missed/extra zones count as deletions/insertions; page exact ignores coordinates and preserves reading order",
              "metrics": metrics}
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(source / name)) for name in names]
    operations.append(CommitOperationAdd(path_in_repo="benchmark_native.json",
        path_or_fileobj=(json.dumps(public, ensure_ascii=False, indent=2) + "\n").encode("utf-8")))
    head = api.model_info(repo_id).sha
    commit = api.create_commit(repo_id, operations=operations, parent_commit=head,
        commit_message="Publish CER-selected LightOnOCR-3 0.8B full-page grounding checkpoint")
    remote = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
    weights = next(s for s in remote.siblings if s.rfilename == "model.safetensors")
    if not weights.lfs or weights.lfs.sha256 != digest or weights.size != (source / "model.safetensors").stat().st_size:
        raise RuntimeError("Published weights failed SHA-256/size verification")
    receipt = {"repo_id": repo_id, "revision": commit.oid, "commit_url": commit.commit_url,
               "weights_sha256": digest, "verified": True, "published_files": names + ["benchmark_native.json"]}
    (source.parent / "hf_publication.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo-id", default="Remidesbois/LightOnOCR-3-Poneglyph-bbox")
    args = parser.parse_args()
    publish(args.source, args.repo_id)
