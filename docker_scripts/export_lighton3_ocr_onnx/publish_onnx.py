"""Publish only hash-verified distribution assets after native and browser validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


def main(directory, repo_id, model_card=None):
    root = Path(__file__).resolve().parents[2]
    for path in [root / ".env", root / "backend/.env", root / "docker_scripts/finetune_lighton_ocr_bbox/.env"]:
        load_dotenv(path)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if api.whoami()["name"] != repo_id.split("/")[0]:
        raise RuntimeError("Authenticated account does not own the requested model repository")
    manifest = json.loads((directory / "browser_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "webgpu_metrics.json").read_text(encoding="utf-8"))
    if manifest["task"] == "bubble-ocr":
        if repo_id != "Remidesbois/LightOnOCR-3-Poneglyph" or not report["verified"] or report["metrics"]["sample_count"] != 1612 or report["metrics"]["token_limit_count"] or report["metrics"]["blank_count"]:
            raise RuntimeError("Bubble WebGPU validation is not complete")
    elif not report["verified"] or report["metrics"]["num_samples"] != 189 or report["metrics"]["truncated_pages"] or report["metrics"]["invalid_blocks"]:
        raise RuntimeError("Full-page WebGPU validation is not complete")
    if report["weights_sha256"] != manifest["source_sha256"]:
        raise RuntimeError("Browser report belongs to another native checkpoint")
    for name, expected in manifest["files"].items():
        with (directory / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected["bytes"] != (directory / name).stat().st_size or expected["sha256"] != digest:
            raise RuntimeError(f"Artifact changed after validation: {name}")
        if report["files"][name] != expected:
            raise RuntimeError(f"Browser validation did not use this artifact: {name}")
    names = list(manifest["files"]) + ["browser_manifest.json", "webgpu_metrics.json", "README.md"]
    if manifest["task"] == "bubble-ocr":
        names.append("quality_comparison.png")
    if any("/" in name or "\\" in name for name in names):
        raise ValueError("Only root distribution files may be published; no fixtures")
    operations = [CommitOperationAdd(path_in_repo=f"onnx/{name}", path_or_fileobj=str(directory / name)) for name in names]
    if model_card:
        operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(model_card)))
    native = api.model_info(repo_id, files_metadata=True)
    weights = next(item for item in native.siblings if item.rfilename == "model.safetensors")
    if not weights.lfs or weights.lfs.sha256 != manifest["source_sha256"]:
        raise RuntimeError("The target repository contains a different native model")
    commit = api.create_commit(repo_id, operations=operations, parent_commit=native.sha,
        commit_message="Add LightOnOCR-3 bubble ONNX/WebGPU, verified on all 1612 frozen test crops" if manifest["task"] == "bubble-ocr" else "Add full-page LightOnOCR-3 ONNX/WebGPU, scored on all 189 frozen test pages")
    remote = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
    metadata = {item.rfilename: item for item in remote.siblings}
    publication_hashes = {}
    for name in names:
        path = directory / name
        with path.open("rb") as stream:
            local_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        expected = {"bytes": path.stat().st_size, "sha256": local_hash}
        publication_hashes[f"onnx/{name}"] = expected
        item = metadata[f"onnx/{name}"]
        if item.size != expected["bytes"] or (item.lfs and item.lfs.sha256 != expected["sha256"]):
            raise RuntimeError(f"Remote asset verification failed: {name}")
        if not item.lfs:
            local = Path(hf_hub_download(repo_id, f"onnx/{name}", revision=commit.oid, token=api.token))
            if hashlib.sha256(local.read_bytes()).hexdigest() != expected["sha256"]:
                raise RuntimeError(f"Downloaded asset verification failed: {name}")
    if model_card:
        expected = hashlib.sha256(model_card.read_bytes()).hexdigest()
        downloaded = Path(hf_hub_download(repo_id, "README.md", revision=commit.oid, token=api.token))
        if hashlib.sha256(downloaded.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Published root model card failed verification")
        publication_hashes["README.md"] = {"bytes": model_card.stat().st_size, "sha256": expected}
    if metadata["model.safetensors"].lfs.sha256 != manifest["source_sha256"]:
        raise RuntimeError("Native weights changed during ONNX publication")
    receipt = {"repo_id": repo_id, "revision": commit.oid, "commit_url": commit.commit_url,
               "native_weights_sha256": manifest["source_sha256"], "published_files": names, "verified": True,
               "published_file_hashes": publication_hashes}
    (directory / "hf_publication.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repo-id", default="Remidesbois/LightOnOCR-3-Poneglyph-bbox")
    parser.add_argument("--model-card", type=Path, help="Updated native model card linking the validated ONNX assets")
    args = parser.parse_args()
    main(args.directory, args.repo_id, args.model_card)
