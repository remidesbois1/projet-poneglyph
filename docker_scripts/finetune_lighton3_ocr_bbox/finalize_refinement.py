"""Document a completed CER refinement and a separately reloaded native export."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct

from grounding import page_metrics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    summary, config, completion = (read(run / name) for name in ("training_summary.json", "run_config.json", "completion.json"))
    if summary["status"] != "complete" or completion["status"] != "complete":
        raise RuntimeError("The training run must be complete before finalization")
    reload_check = read(run / "native_reload_check.json")
    if reload_check["invalid_blocks"] or reload_check["truncated"] or not reload_check["items"]:
        raise RuntimeError("The fresh native reload did not produce a valid complete grounding response")
    selected_step = Path(summary["best_model_checkpoint"]).name.split("-")[-1]
    baseline = read(run / "baseline_validation.json")
    selected = baseline if summary.get("retained_initial_model") else read(run / f"validation_step_{selected_step}.json")
    test = read(run / "test_benchmark.json")
    image_name = Path(reload_check["image"]).name
    page = next(p for p in selected["pages"] if Path(p["image_file"]).name == image_name)
    native_metrics = page_metrics(page["reference"], reload_check["prediction"], reload_check["truncated"])
    weights = run / "final_merged/model.safetensors"
    with weights.open("rb") as source:
        header = json.loads(source.read(struct.unpack("<Q", source.read(8))[0]))
    dtypes = dict(Counter(tensor["dtype"] for name, tensor in header.items() if name != "__metadata__"))
    if set(dtypes) != {"BF16"}:
        raise RuntimeError(f"Unexpected export tensor dtypes: {dtypes}")
    import torch
    from safetensors import safe_open
    torch.set_num_threads(8)
    selected_weights = Path(summary["best_model_checkpoint"]) / "model.safetensors"
    with safe_open(selected_weights, framework="pt", device="cpu") as source, safe_open(weights, framework="pt", device="cpu") as exported:
        if set(source.keys()) != set(exported.keys()):
            raise RuntimeError("Export and selected checkpoint have different tensor keys")
        for key in source.keys():
            final_tensor = exported.get_tensor(key)
            if not torch.equal(source.get_tensor(key).to(final_tensor.dtype), final_tensor):
                raise RuntimeError(f"Export differs from selected checkpoint after BF16 casting: {key}")
        verified_tensors = len(list(source.keys()))
    verification = {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
                    "base_model": config["model_id"], "base_revision": config["revision"],
                    "initial_checkpoint": config["init_model"], "selected_checkpoint": summary["best_model_checkpoint"],
                    "selection_metric": "Minimum generation CER on 189 frozen validation pages; test never selects a checkpoint.",
                    "retained_initial_model": bool(summary.get("retained_initial_model")),
                    "baseline_validation_cer": baseline["metrics"]["cer"], "selected_validation": selected["metrics"],
                    "export": {"file": str(weights), "bytes": weights.stat().st_size,
                               "sha256": sha256(weights), "tensor_dtypes": dtypes,
                               "selected_checkpoint_bf16_cast_exact": True, "verified_tensor_count": verified_tensors,
                               "native_loader": "Qwen3_5ForConditionalGeneration + AutoProcessor, local_files_only=True"},
                    "fresh_native_reload": {"source": "native_reload_check.json", "image": reload_check["image"],
                                            "page_id": page["page_id"], "boxes": len(reload_check["items"]),
                                            "invalid_blocks": reload_check["invalid_blocks"], "truncated": reload_check["truncated"],
                                            "generated_tokens": reload_check["generated_tokens"],
                                            "generation_seconds": reload_check["generation_seconds"],
                                            "embedding_head_tied": reload_check["embedding_head_tied"],
                                            "metrics_against_validation_gt": native_metrics,
                                            "prediction_identical_to_validation_batch": reload_check["prediction"] == page["prediction"]},
                    "test_previously_evaluated": bool(config["test_previously_evaluated"]),
                    "test": test["metrics"], "test_ci95": test["bootstrap_ci95"],
                    "postprocessing_sources_sha256": {name: sha256(Path(__file__).parent / name)
                        for name in ("grounding.py", "infer.py", "compare_refinement.py", "finalize_refinement.py", "report.py", "monitor.py", "rescore_saved.py")}}
    (run / "model_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    a, b, t = baseline["metrics"], selected["metrics"], test["metrics"]
    card = f"""---
base_model: {config['model_id']}
license: apache-2.0
pipeline_tag: image-text-to-text
language:
- fr
tags:
- ocr
- manga
- grounding
---

# Poneglyph · LightOnOCR-3 0.8B · consolidation CER

OCR de pages complètes de manga en français : transcription et bbox normalisées
0–1000, dans le format natif `![text](x1,y1,x2,y2)` suivi du texte.
Prompt utilisateur **`grounding`**, template natif avec **`enable_thinking=False`**.
Les annotations concernent les zones de texte validées, pas les descriptions de cases.

Base : [{config['model_id']}](https://huggingface.co/{config['model_id']})
à la révision `{config['revision']}`. Architecture native Qwen3.5, poids BF16.

## Entraînement et sélection

Départ : `{config['init_model']}` ; checkpoint retenu : `{summary['best_model_checkpoint']}`.
Cette phase utilise un nouvel AdamW/scheduler : LR décodeur 3e-6, merger 1,5e-6,
vision 7,5e-7 ; trois époques maximum, batch physique 1 × accumulation 8.
793 333 056 paramètres entraînés sur 852 985 920 : décodeur, merger et quatre
derniers blocs vision. Poids/états Adam FP32, autocast BF16, kernels Blackwell SM120.
Images pleine page, bord long maximum 2048 pixels, sans augmentation dans cette phase.

Loss sur la réponse uniquement : texte/séparateurs 1, marqueurs/coordonnées 0,2,
EOS 2. Les chiffres contenus dans la transcription gardent le poids 1.
La sélection utilise exclusivement le **CER minimum par génération sur les 189
pages de validation**, avec protection du modèle initial si le gain est insuffisant.
Le WER et les bbox sont suivis. Les données restent figées : 882 pages train,
189 validation, 189 test ; aucune page de test ne participe au train ou à la sélection.

## Résultats

| Mesure | Départ · validation | Retenu · validation | Test final |
|---|---:|---:|---:|
| CER | {a['cer']:.3%} | {b['cer']:.3%} | {t['cer']:.3%} |
| WER | {a['wer']:.3%} | {b['wer']:.3%} | {t['wer']:.3%} |
| Texte exact par bulle | {a['exact_bubble_text_rate']:.2%} | {b['exact_bubble_text_rate']:.2%} | {t['exact_bubble_text_rate']:.2%} |
| Texte de page exact, dans l'ordre | {a['page_text_exact']:.2%} | {b['page_text_exact']:.2%} | {t['page_text_exact']:.2%} |
| F1 bbox / IoU ≥ 0,5 | {a['f1@0_5']:.2%} | {b['f1@0_5']:.2%} | {t['f1@0_5']:.2%} |
| Pages tronquées | {a['truncated_pages']} | {b['truncated_pages']} | {t['truncated_pages']} |

CER/WER pondérés par les caractères/mots de référence ; NFC et espaces normalisés,
casse et ponctuation conservées. Appariement spatial un à un à IoU ≥ 0,1 :
zones manquées/ajoutées pénalisées par suppressions/insertions.
Le texte exact par bulle porte sur toutes les bulles annotées ; l'exact par page
ignore les coordonnées mais conserve l'ordre de lecture. L'égalité littérale de
la réponse entière, bbox comprises, est une autre mesure (`exact_match`).

**Le test avait déjà été évalué et inspecté lors du premier run.** Ces résultats
mesurent un progrès sur le même test réutilisé, pas sur un nouveau jeu indépendant.
Toutes les sorties, y compris les échecs, sont incluses. Intervalles bootstrap,
comparaisons appariées et anciens modèles : voir les JSON du dossier parent.

## Utilisation locale et traçabilité

Charger ce dossier avec `AutoProcessor.from_pretrained` et
`Qwen3_5ForConditionalGeneration.from_pretrained`, en BF16 sur CUDA, attention SDPA.
Appliquer le template natif avec le prompt `grounding`, thinking désactivé,
et générer avec `do_sample=False`, `use_cache=True`, `max_new_tokens=1280`.
Le script `infer.py` de ce pipeline fournit l'inférence complète et les bbox en pixels.

Poids : `{verification['export']['bytes']}` octets ; SHA-256
`{verification['export']['sha256']}`. Export rechargé dans un nouveau processus :
{len(reload_check['items'])} zones valides, sortie complète, aucune zone mal formée
sur la page de validation `{page['page_id']}`. Ce contrôle vérifie le fonctionnement
de l'export ; les scores du tableau proviennent des évaluations complètes.

Le dossier parent conserve `run_config.json`, `execution_manifest.json`,
`source_snapshot/`, les checkpoints, les prédictions, `model_verification.json`,
`results.json`, les courbes PNG/SVG, TensorBoard, Trackio local et le dashboard.
"""
    (run / "final_merged/README.md").write_text(card, encoding="utf-8")
    from report import write_report
    from monitor import write_dashboard
    write_report(run)
    history = []
    for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "start":
            history = [entry for entry in history if entry["step"] <= record["step"]]
        elif record.get("event") == "metrics":
            history.append(record)
    write_dashboard(run / "training_dashboard.html", summary, history)
    print(json.dumps({"weights_sha256": verification["export"]["sha256"], "dtypes": dtypes,
                      "selected_validation_cer": b["cer"], "test_cer": t["cer"],
                      "native_reload": verification["fresh_native_reload"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
