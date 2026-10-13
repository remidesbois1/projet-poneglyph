"""Write public model documentation from verified aggregate measurements only."""
import argparse
import json
from pathlib import Path


def main(directory, model_card):
    report = json.loads((directory / "webgpu_metrics.json").read_text(encoding="utf-8"))
    if not report["verified"]:
        raise RuntimeError("Complete native ONNX and browser validation before documenting a release")
    native, browser = report["native_reference_metrics"], report["metrics"]
    rows = [("CER ↓", "cer", 3), ("WER ↓", "wer", 3),
            ("Texte exact par zone ↑", "exact_bubble_text_rate", 2),
            ("Texte de page exact, dans l’ordre ↑", "page_text_exact", 2),
            ("F1 bbox / IoU ≥ 0,5 ↑", "f1@0_5", 2)]
    table = "| Mesure · 189 pages de test | PyTorch BF16 | Navigateur WebGPU |\n|---|---:|---:|\n"
    for label, key, precision in rows:
        table += f"| {label} | {native[key] * 100:.{precision}f} % | {browser[key] * 100:.{precision}f} % |\n"
    table += f"| Pages tronquées / blocs invalides | {native['truncated_pages']} / {native['invalid_blocks']} | {browser['truncated_pages']} / {browser['invalid_blocks']} |\n"
    delta = report["native_comparison"]
    latency = report["latency"]
    summary = (table + f"\nLe navigateur totalise {delta['character_edits_webgpu']} erreurs de caractères, "
        f"contre {delta['character_edits_native']} en BF16 : +{delta['cer_delta'] * 100:.3f} point de CER. "
        "Ces sorties ne sont pas identiques bit à bit. Les poids FP16, calculs mixtes FP16/FP32 "
        "et le redimensionnement Canvas peuvent modifier les textes et coordonnées.\n\n"
        f"Sur RTX 5090 32 Go / Blackwell : médiane **{latency['median_ms'] / 1000:.2f} s/page**, "
        f"p95 **{latency['p95_ms'] / 1000:.2f} s/page**, modèle chargé, téléchargement exclu. "
        "Les temps dépendent de la taille de l’image, du texte et du matériel.\n\n"
        "Le test est le même jeu figé de **189 pages / 1646 zones** déjà évalué et inspecté "
        "pendant l’entraînement. Il mesure le portage, pas une nouvelle évaluation indépendante. "
        "CER/WER : appariement spatial un à un à IoU ≥ 0,1, zones manquées/ajoutées pénalisées ; "
        "NFC et espaces normalisés, casse et ponctuation conservées. L’exact par zone porte sur "
        "toutes les zones annotées ; l’exact par page ignore les coordonnées et conserve l’ordre.\n")
    file_table = "| Artefact | Usage |\n|---|---|\n"
    for name, purpose in [
        ("vision.onnx + vision.onnx.data", "Encodeur de l’image entière, features visuelles"),
        ("decoder.onnx + decoder.onnx.data", "Prefill portable par blocs, avec logits et argmax"),
        ("decode.onnx + decoder.onnx.data", "Décodage portable, un token et états explicites"),
        ("prefill.onnx + decoder.onnx.data", "Prefill portable, états seuls"),
        ("prefill-fast.onnx + decoder.onnx.data", "Prefill WebGPU, RMSNormalization standard"),
        ("decode_weights.json", "Index des poids pour le décodeur WGSL fusionné de Poneglyph"),
        ("browser_manifest.json", "Contrat image/prompt/EOS/configuration et SHA-256 des artefacts"),
        ("tokenizer.json + tokenizer_config.json", "Vocabulaire complet, 248 320 tokens"),
        ("webgpu_metrics.json", "Agrégats, intervalles bootstrap, contrôle ONNX, latences et intégrité"),
    ]:
        file_table += f"| `{name}` | {purpose} |\n"
    cuda = report["onnx_cuda_validation"]
    documentation = f"""# LightOnOCR-3 Poneglyph · ONNX / WebGPU · full-page bbox

Export du checkpoint 111 du [modèle natif](../README.md), sélectionné sur le CER de validation.
Poids BF16 source SHA-256 : `{report['weights_sha256']}`.
Évaluation du {report['date']}. Aucune quantification INT8/INT4.

## Mesures sur le moteur réel de Poneglyph

{summary}
## Format et exécution

Prompt **`grounding`**, template natif avec **thinking désactivé**. Sortie :

```text
![text](x1,y1,x2,y2)
transcription de la zone
```

Coordonnées entières dans [0,1000] ; multiplier x par la largeur **originale** / 1000
et y par la hauteur originale / 1000. Images RGB : bord long ≤ 2048 px, sans agrandir
à cette étape ; grille Qwen de 32 px, budget 65 536–4 194 304 pixels.
Génération gloutonne, 1280 tokens maximum, EOS 248046, image token 248056.
Le modèle prédit les zones de texte, pas les descriptions de cases.

Poids/projections FP16 ; résidus, normalisations et état récurrent DeltaNet FP32.
12 blocs vision, 24 blocs texte Qwen3.5. Attention globale visuelle par 16 groupes
de requêtes : **chaque requête voit toutes les clés et valeurs**. Hiérarchie Split/Concat
de quatre groupes pour respecter les limites de buffers WebGPU, sans attention locale.
États DeltaNet et caches KV explicites ; prefill par blocs de 32. Aucun Loop/Scan,
opérateur Triton/FLA ou domaine ONNX personnalisé.

{file_table}
Les graphes portables utilisent l’opset 17. Le graphe `prefill-fast.onnx` utilise
RMSNormalization standard et l’opset 23. Les données externes sont relatives aux
graphes : conserver les noms `.onnx.data` lors du téléchargement.

Dans Poneglyph : **Page entière → LightOnOCR 3 · WebGPU → Charger le modèle → Lire la page entière**.
Les zones et textes générés deviennent des annotations modifiables ; calcul local,
streaming, annulation et cache des poids. La page `/sandbox/lighton3` affiche les bbox
et la sortie grounding. Le chemin rapide utilise ONNX Runtime Web **1.27.0** pour
vision/prefill et le décodeur WGSL Qwen3.5 partagé avec Surya ; features et états
restent sur le GPU, fallback CPU désactivé. Chrome/Edge avec WebGPU et `shader-f16`
requis. Le chemin intégral ONNX est disponible avec `load(progress, {{ mode: 'onnx' }})`.

## Vérification de l’export

Les cinq graphes passent `onnx.checker`. Génération portable complète sur **{cuda['pages']} contrôles**
avec **{cuda['runtime']}** : états/features finis, comparaison numérique au BF16,
EOS atteint, bbox valides, ancienne page qui bouclait et répétition de la première page.
La répétition produit une sortie identique après réinitialisation des caches.
Maximum de l’erreur RMS relative des features vision face au BF16 : **{cuda['maximum_vision_relative_rmse']:.4f}**.
La validation native utilise le runtime CUDA 12.8 compatible avec l’image PyTorch ;
le runtime WebGPU est vérifié séparément dans le navigateur, sur les **189 pages complètes**.

Les empreintes et tailles de chaque artefact sont dans `browser_manifest.json`.
Le rapport contient les contrôles, agrégats natifs et navigateur ainsi que 5000 tirages
bootstrap par page. Les SHA des sources moteur décrivent la version du benchmark local ;
la publication est ensuite épinglée par commit dans l’application.
**Aucune image, annotation, fixture ou transcription individuelle n’est publiée.**

Le modèle peut manquer des textes et onomatopées ; les transcriptions et bbox restent
à vérifier selon l’usage. FP16/WebGPU conserve ici un CER proche du BF16, sans garantie
d’identité des sorties ni de précision sur d’autres documents.

Références : [contrat natif LightOnOCR-3](https://huggingface.co/lightonai/LightOnOCR-3-0.8B),
[ORT WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html).
"""
    (directory / "README.md").write_text(documentation, encoding="utf-8")
    card = model_card.read_text(encoding="utf-8").split("\n## ONNX et WebGPU\n")[0]
    card = card.replace("Toutes les sorties, y compris les échecs, sont incluses. Intervalles bootstrap,\n", "Toutes les sorties, y compris les échecs, sont incluses.\n")
    card += "\n## ONNX et WebGPU\n\n[Graphes, contrat et validation](onnx/README.md) · [Rapport agrégé du navigateur](onnx/webgpu_metrics.json).\n\n" + summary
    card += "\nIntégré à Poneglyph sous **Page entière → LightOnOCR 3 · WebGPU**. Export sans quantification, poids FP16 et calculs sensibles FP32. Les mesures navigateur ci-dessus proviennent de l’exécution WebGPU réelle, sans fallback CPU.\n"
    model_card.write_text(card, encoding="utf-8")
    print(json.dumps({"verified": True, "documentation": str(directory / "README.md"), "model_card": str(model_card)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--model-card", type=Path, required=True)
    args = parser.parse_args()
    main(args.directory, args.model_card)
