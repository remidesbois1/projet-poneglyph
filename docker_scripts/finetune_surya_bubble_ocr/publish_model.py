"""Atomically publish matched native and WebGPU Surya weights, without private data."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download


NATIVE_FILES = (
    "model.safetensors", "config.json", "generation_config.json",
    "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def public_model_card(metrics, digest):
    return f"""---
license: openrail
pipeline_tag: image-text-to-text
tags:
- ocr
- french
- manga
- surya
---

# Surya Bubble OCR · Poneglyph

Reconnaissance du texte français dans les bulles de bande dessinée : modèle Surya OCR 2 ajusté sur des bulles Poneglyph. Les poids natifs sont en BF16 et nécessitent un environnement compatible Qwen3.5. La version [ONNX/WebGPU](onnx) est exportée à partir de ces mêmes poids en précision mixte FP16/FP32.

Évaluation native BF16 sur {metrics['sample_count']} bulles de test, séparées par page des données d'entraînement : CER {metrics['cer'] * 100:.3f} %, WER {metrics['wer'] * 100:.3f} %, transcription exacte {metrics['exact_match'] * 100:.2f} %. Aucune sortie n'a atteint la limite de 256 tokens. Les détails et intervalles de confiance sont dans [benchmark_test.json](benchmark_test.json). Ces résultats concernent l'inférence native, pas l'export WebGPU ; les images et prédictions individuelles ne sont pas publiées.

SHA-256 du fichier `model.safetensors` : `{digest}`.
"""


def publish(source, repo_id, onnx_dir, parent_revision):
    source, onnx_dir = source.resolve(), onnx_dir.resolve()
    for name in NATIVE_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    digest = sha256(source / "model.safetensors")
    benchmark = json.loads((source / "benchmark_test.json").read_text(encoding="utf-8"))
    benchmark = {k: v for k, v in benchmark.items() if k not in {"samples", "top_20_worst_errors"}}
    benchmark.update(model_id=repo_id, weights_sha256=digest, evaluation_runtime="Transformers BF16 CUDA; not ONNX")
    if not benchmark["metrics"]["sample_count"] or benchmark["metrics"]["token_limit_count"]:
        raise ValueError("A complete test evaluation without token truncation is required")

    manifest_path = onnx_dir / "browser_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_sha256"] != digest or manifest["model_id"] != repo_id:
        raise ValueError("ONNX export does not match the native checkpoint or repository")
    for name, metadata in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError(f"Unsafe ONNX filename: {name}")
        path = onnx_dir / name
        if not path.is_file() or path.stat().st_size != metadata["bytes"] or sha256(path) != metadata["sha256"]:
            raise ValueError(f"Missing or incorrect ONNX file: {name}")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.model_info(repo_id, files_metadata=True)
    if parent_revision and info.sha != parent_revision:
        raise RuntimeError(f"Repository changed since preparation: {info.sha} != {parent_revision}")
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(source / name)) for name in NATIVE_FILES]
    operations.extend([
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=public_model_card(benchmark["metrics"], digest).encode()),
        CommitOperationAdd(path_in_repo="benchmark_test.json", path_or_fileobj=json.dumps(benchmark, ensure_ascii=False, indent=2).encode()),
        CommitOperationAdd(path_in_repo="onnx/browser_manifest.json", path_or_fileobj=str(manifest_path)),
        CommitOperationAdd(path_in_repo="onnx/README.md", path_or_fileobj=(
            "# Surya Bubble OCR WebGPU\n\nExport FP16/FP32 des poids BF16 de cette révision. "
            "Les métriques à la racine décrivent le moteur natif, pas cet export. "
            "Poneglyph utilise les graphes vision/préremplissage et son décodeur WebGPU.\n"
        ).encode()),
    ])
    for name in manifest["files"]:
        operations.append(CommitOperationAdd(path_in_repo=f"onnx/{name}", path_or_fileobj=str(onnx_dir / name)))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        operations.append(CommitOperationAdd(path_in_repo=f"onnx/{name}", path_or_fileobj=str(source / name)))
    if any(s.rfilename == "onnx/webgpu_metrics.json" for s in info.siblings):
        operations.append(CommitOperationDelete(path_in_repo="onnx/webgpu_metrics.json"))
    result = api.create_commit(repo_id, operations=operations, parent_commit=info.sha,
                               commit_message="Publish matched BF16 and WebGPU bubble OCR checkpoint")
    remote = api.model_info(repo_id, revision=result.oid, files_metadata=True)
    files = {s.rfilename: s for s in remote.siblings}
    for path, expected_hash, expected_size in [
        ("model.safetensors", digest, (source / "model.safetensors").stat().st_size),
        *[(f"onnx/{name}", meta["sha256"], meta["bytes"]) for name, meta in manifest["files"].items() if meta["bytes"] > 10_000_000],
    ]:
        item = files[path]
        if not item.lfs or item.lfs.sha256 != expected_hash or item.size != expected_size:
            raise RuntimeError(f"Published file verification failed: {path}")
    downloaded = hf_hub_download(repo_id, "onnx/browser_manifest.json", revision=result.oid, token=os.environ.get("HF_TOKEN"))
    if json.loads(Path(downloaded).read_text(encoding="utf-8"))["source_sha256"] != digest:
        raise RuntimeError("Published ONNX manifest has the wrong source")
    receipt = {"repo_id": repo_id, "revision": result.oid, "commit_url": result.commit_url,
               "weights_sha256": digest, "verified": True}
    (source.parent / "hf_publication.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="Remidesbois/surya-bubble-ocr-poneglyph")
    parser.add_argument("--parent-revision", help="Abort rather than overwrite a newer repository revision")
    args = parser.parse_args()
    publish(args.source, args.repo_id, args.onnx_dir, args.parent_revision)
