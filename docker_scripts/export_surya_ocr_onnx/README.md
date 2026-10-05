# Surya OCR Poneglyph — ONNX / WebGPU

Export du fine-tuning non-bbox du 5 octobre 2026, checkpoint sélectionné sur le
CER de validation. Poids source SHA-256 :
`67fa1733d2541d6305cffef1e5202e47930a5f5393d1901adda52e7daab05629`.

La page `/sandbox/surya` et le moteur **Surya · WebGPU** des sélecteurs OCR
exécutent la transcription entièrement dans le navigateur. Le premier
chargement télécharge environ 1,33 Go ; les poids restent en cache local.
Chrome/Edge avec WebGPU et `shader-f16` sont nécessaires. Les images restent
sur l'appareil. Une seule bulle RGB est traitée par appel.

## Graphes

| Fichier | Fonction |
| --- | --- |
| `vision.onnx` + `vision.onnx.data` | Encodeur visuel 12 blocs, patches 16×16, fusion spatiale 2×2 |
| `decoder.onnx` | Préremplissage du prompt par blocs de 1 à 32 tokens |
| `decode.onnx` | Décodage récurrent spécialisé à un token |
| `decoder.onnx.data` | Poids partagés par les deux graphes de texte |
| `browser_manifest.json` | Tailles, SHA-256, architecture et prompt tokenisé |

Opérateurs ONNX standard, opset 17. Poids, matmuls et cache KV en FP16 ;
résidus, normalisations et états DeltaNet en FP32. Aucun noyau CUDA/Triton ni
backend Python pendant l'inférence WebGPU. Les états des 18 blocs DeltaNet
et le cache des 6 blocs d'attention restent sur GPU entre les appels. Seul
l'identifiant du token est lu par le CPU. Le vocabulaire n'est projeté qu'à
la dernière position. Les deux sessions texte partagent les fichiers de
poids téléchargés mais ORT peut les allouer séparément en mémoire GPU.

Le préremplissage résout exactement la récurrence triangulaire de chaque
bloc de 32 tokens par une série de Neumann finie : la matrice strictement
triangulaire est nilpotente. Le graphe de décodage applique directement la
récurrence à un token, sans calcul des matrices de préremplissage.

Le worker utilise ONNX Runtime Web 1.27.0, `executionProviders: ['webgpu']`
et `disable_cpu_ep_fallback: '1'`. Les sessions et buffers sont libérés lors
du démontage du worker ; les états sont réinitialisés pour chaque image.
Le téléchargement reprend par blocs de 4 Mio et vérifie le SHA-256 complet.

## Prétraitement

Même prompt français, budget de 65 536 à 1 048 576 pixels, grille de 32 px,
normalisation [-1,1], ordre des patches, interpolation des positions et
RoPE multimodal intercalé que le processor Transformers sauvegardé.
Les frames temporelles sont dupliquées pour les images fixes. Le
redimensionnement Canvas haute qualité peut différer légèrement de Pillow.
Génération gloutonne, EOS 2, limite 256 tokens. Les espaces sont normalisés
et une paire de guillemets extérieurs est retirée, comme à l'entraînement.
La limite de tokens est signalée explicitement à l'appelant.

## Vérifications du 5 octobre 2026

Sur RTX 5090 / Blackwell dans le navigateur Chromium du poste :

- 24 bulles de validation choisies à intervalles réguliers de longueur,
  de 1 à 139 tokens générés : 24 générations complètes et conformes à la
  référence native, aucune erreur WebGPU et aucune limite atteinte.
- Durée par bulle, modèle chargé : médiane **1 064 ms**, plage **350–4 311 ms**.
  Le téléchargement et le chargement initial des poids sont exclus.
- Une image redimensionnée à 864×1 184 px (3 996 patches) : transcription
  complète en **6 606 ms**, dont **876 ms** pour l'image et **2 447 ms** jusqu'au
  premier token. Ce dernier temps inclut l'image.
- Vérifications indépendantes des tokens, positions, grilles et rotations
  du JavaScript contre les sorties du processor Transformers.

Ces contrôles vérifient le portage ; ils ne constituent pas une nouvelle
évaluation complète de généralisation. Les scores du test BF16 de 1 612
bulles ne doivent pas être attribués automatiquement à ONNX. Les crops,
références individuelles et entrées de contrôle restent locaux.

## Reproduire

Depuis la racine du dépôt, PowerShell :

```powershell
docker build -t poneglyph/surya-onnx:20261005 docker_scripts/export_surya_ocr_onnx
$source = (Resolve-Path 'docker_scripts/finetune_surya_bubble_ocr/outputs_surya_bubble_ocr/rtx5090-20261005/final_merged').Path
New-Item -ItemType Directory -Force frontend/public/models/surya-ocr | Out-Null
$output = (Resolve-Path 'frontend/public/models/surya-ocr').Path
docker run --rm -v "${source}:/source:ro" -v "${output}:/output" poneglyph/surya-onnx:20261005 --source /source --output /output
npm.cmd run dev --workspace frontend
```

L'export lui-même fonctionne sur CPU. Pour les fixtures de contrôle, monter
le dataset sous `/dataset`, ajouter `--gpus all --entrypoint python` et lancer
`make_fixtures.py --source /source --dataset /dataset --output /output --count 24 --stress`.
Les fixtures sont des données locales ignorées par Git ; ne pas les distribuer.
`validate_onnx.py --model /output --count 6` exécute aussi les graphes avec
ONNX Runtime Python. La validation CPU est beaucoup plus lente que WebGPU.

En développement, le worker détecte `/models/surya-ocr/browser_manifest.json`.
Sinon il télécharge la version Hugging Face épinglée dans `suryaWebgpu.js`.
Révision de distribution : `16efe756422b73054b97ea1fe082a0724627eadb`.
`NEXT_PUBLIC_SURYA_MODEL_BASE` permet d'imposer une URL de dossier ; redémarrer
Next.js après modification. Le modèle fonctionne sans les fixtures.

`upload_onnx.py --directory <dossier>` publie uniquement les graphes, leurs
poids, les deux fichiers tokenizer, le manifeste et cette documentation,
avec `HF_TOKEN` dans l'environnement. Il vérifie aussi l'empreinte du modèle
source publié. Après publication, épingler la révision retournée dans le worker.

Références techniques :
[Qwen3.5 / Transformers](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5),
[opérateurs WebGPU ORT](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md),
[ONNX Runtime WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html).
