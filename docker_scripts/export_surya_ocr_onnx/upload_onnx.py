"""Publish only hash-verified ONNX release files, never private browser fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def publish(directory, repo_id):
    manifest = json.loads((directory / "browser_manifest.json").read_text())
    files = list(manifest["files"]) + ["browser_manifest.json", "tokenizer.json", "tokenizer_config.json"]
    for name, expected in manifest["files"].items():
        path = directory / name
        if path.name != name:
            raise ValueError("Only flat release filenames are allowed")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"Artifact changed: {name}")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.model_info(repo_id, files_metadata=True)
    native = next(s for s in info.siblings if s.rfilename == "model.safetensors")
    if native.lfs.sha256 != manifest["source_sha256"]:
        raise RuntimeError("Native Hub checkpoint differs from the ONNX source")
    operations = [CommitOperationAdd(path_in_repo=f"onnx/{name}", path_or_fileobj=str(directory / name)) for name in files]
    operations.append(CommitOperationAdd(path_in_repo="onnx/README.md", path_or_fileobj=str(Path(__file__).with_name("README.md"))))
    result = api.create_commit(repo_id, operations=operations,
        commit_message="Update verified Surya WebGPU artifacts and optimized prompt graphs")
    info = api.model_info(repo_id, revision=result.oid, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    for name, expected in manifest["files"].items():
        sibling = remote[f"onnx/{name}"]
        if sibling.size != expected["bytes"] or (sibling.lfs and sibling.lfs.sha256 != expected["sha256"]):
            raise RuntimeError(f"Hub artifact mismatch: {name}")
    receipt = {"repo_id": repo_id, "revision": result.oid, "commit_url": result.commit_url, "verified": True}
    (directory / "hf_publication.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repo-id", default="Remidesbois/surya-bubble-ocr-poneglyph")
    args = parser.parse_args()
    publish(args.directory, args.repo_id)
