"""Publish a finished Surya checkpoint, without private crops or predictions."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def publish(source, repo_id):
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    names = ["model.safetensors", "config.json", "generation_config.json",
             "processor_config.json", "tokenizer.json", "tokenizer_config.json",
             "chat_template.jinja"]
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    with (source / "model.safetensors").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    readme = (source / "README.md").read_text(encoding="utf-8")
    readme = readme.replace(
        "Le fichier `benchmark_test.json` contient toutes les\nprédictions, les sous-catégories et les intervalles de confiance par bootstrap\nde pages. La normalisation des espaces et le post-traitement sont décrits dans\nce fichier.",
        "Les crops et les prédictions individuelles restent privés. Les métriques\nutilisent une normalisation des espaces et retirent une paire de guillemets\nextérieurs. Les scores ci-dessus concernent les poids BF16 évalués en Python.")
    readme = readme.split("La recette, les versions des packages")[0]
    readme += f"\nEmpreinte SHA-256 des poids : `{digest}`.\n"
    if (source.parent / "onnx-decode" / "hf_publication.json").exists():
        readme += ("\n## ONNX / WebGPU\n\nUne version navigateur validée est disponible dans [onnx/](onnx). "
                   "Elle contient un encodeur visuel, un graphe de préremplissage et un décodeur récurrent, "
                   "en précision mixte FP16/FP32. Voir les [instructions et contrôles](onnx/README.md). "
                   "Le dépôt Poneglyph propose la page `/sandbox/surya` et le moteur **Surya · WebGPU**.\n")
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(source / name)) for name in names]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")))
    benchmark = json.loads((source / "benchmark_test.json").read_text(encoding="utf-8"))
    benchmark = {k: v for k, v in benchmark.items() if k not in {"samples", "top_20_worst_errors"}}
    benchmark["weights_sha256"] = digest
    benchmark["evaluation_runtime"] = "Transformers BF16 CUDA; not ONNX"
    operations.append(CommitOperationAdd(path_in_repo="benchmark_test.json",
        path_or_fileobj=json.dumps(benchmark, ensure_ascii=False, indent=2).encode("utf-8")))
    result = api.create_commit(repo_id, operations=operations,
        commit_message="Publish RTX 5090 Surya bubble OCR fine-tune, best validation checkpoint")
    info = api.model_info(repo_id, revision=result.oid, files_metadata=True)
    remote = next(s for s in info.siblings if s.rfilename == "model.safetensors")
    if not remote.lfs or remote.lfs.sha256 != digest:
        raise RuntimeError("Remote weights do not match the selected local checkpoint")
    receipt = {"repo_id": repo_id, "revision": result.oid, "commit_url": result.commit_url,
               "weights_sha256": digest, "verified": True}
    (source.parent / "hf_publication.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo-id", default="Remidesbois/surya-bubble-ocr-poneglyph")
    args = parser.parse_args()
    publish(args.source, args.repo_id)
