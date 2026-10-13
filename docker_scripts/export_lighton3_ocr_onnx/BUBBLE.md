# LightOnOCR-3 Poneglyph · bulles, ONNX et WebGPU

Source : `docker_scripts/finetune_lighton3_bubble_ocr/outputs/rtx5090-20261012-bubble/final_merged`,
checkpoint **1175**, révision native HF **3d8ee1c3f32ad7e662a2543cc77e75240801fd26**.
SHA-256 safetensors : `45c7aa6054eddd1a879715fdb5a3f208dbfedefe06ec270a2f83d7032e61820f`.

Artefacts publiés et vérifiés : [révision dd066c12](https://huggingface.co/Remidesbois/LightOnOCR-3-Poneglyph/tree/dd066c12c4dcc990d71ebc78136e9c7c21c36522/onnx),
épinglée dans le frontend. Test WebGPU : **CER 0,393746 %**, **WER 1,409561 %**,
**exact match 92,2457 %** ; 205 erreurs de caractères contre 202 en natif,
1 600/1 612 transcriptions identiques, aucune sortie vide ou tronquée.
Sur RTX 5090 : **176 ms** médiane et **238 ms** p95, modèle chargé.

Cette variante prend **un crop RGB de bulle**, avec un message utilisateur qui ne
contient que l'image. Prompt vide, thinking désactivé, génération gloutonne,
EOS 248046, **256 tokens maximum**. Elle retourne du texte ; les rectangles viennent
du flux d'annotation de l'app. Le modèle de pages complètes reste distinct.

## Export reproductible

L'image `poneglyph/lighton3-onnx:5090` est construite avec le Dockerfile de ce dossier.
Elle utilise Transformers 5.14.1, PyTorch 2.8.0 / CUDA 12.8 et ORT GPU 1.26.0.
Le navigateur utilise ORT Web 1.27.0, indépendamment de la validation CUDA.

Depuis la racine du dépôt, PowerShell :

```powershell
docker build -t poneglyph/lighton3-onnx:5090 docker_scripts/export_lighton3_ocr_onnx
$bubbleSource = (Resolve-Path 'docker_scripts/finetune_lighton3_bubble_ocr/outputs/rtx5090-20261012-bubble/final_merged').Path
$bubbleScripts = (Resolve-Path 'docker_scripts').Path
$bubbleDataset = (Resolve-Path 'docker_scripts/finetune_surya_bubble_ocr/surya_bubble_dataset').Path
New-Item -ItemType Directory -Force frontend/public/models/lighton3-bubble | Out-Null
$bubbleOutput = (Resolve-Path 'frontend/public/models/lighton3-bubble').Path
docker run --rm -v "${bubbleSource}:/source:ro" -v "${bubbleOutput}:/output" -v "${bubbleScripts}:/scripts:ro" poneglyph/lighton3-onnx:5090 /scripts/export_lighton3_ocr_onnx/export_lighton3.py --source /source --output /output --task bubble-ocr --source-revision 3d8ee1c3f32ad7e662a2543cc77e75240801fd26
docker run --rm --gpus all --ipc=host -v "${bubbleSource}:/source:ro" -v "${bubbleDataset}:/dataset:ro" -v "${bubbleOutput}:/output" -v "${bubbleScripts}:/scripts:ro" poneglyph/lighton3-onnx:5090 /scripts/export_lighton3_ocr_onnx/make_fixtures.py --source /source --dataset /dataset --output /output --count 12 --task bubble-ocr
docker run --rm --gpus all -v "${bubbleSource}:/source:ro" -v "${bubbleOutput}:/output" -v "${bubbleScripts}:/scripts:ro" poneglyph/lighton3-onnx:5090 /scripts/export_lighton3_ocr_onnx/check_vision_precision.py --source /source --model /output
docker run --rm --gpus all -v "${bubbleOutput}:/output" -v "${bubbleScripts}:/scripts:ro" poneglyph/lighton3-onnx:5090 /scripts/export_lighton3_ocr_onnx/validate_onnx.py --model /output --provider cuda --count 16
docker run --rm -v "${bubbleSource}:/source:ro" -v "${bubbleOutput}:/output" -v "${bubbleScripts}:/scripts:ro" poneglyph/lighton3-onnx:5090 /scripts/export_lighton3_ocr_onnx/make_resize_fixtures.py --source /source --directory /output
node docker_scripts/export_lighton3_ocr_onnx/verify_resize.mjs frontend/public/models/lighton3-bubble
python docker_scripts/export_lighton3_ocr_onnx/make_bubble_benchmark.py --dataset docker_scripts/finetune_surya_bubble_ocr/surya_bubble_dataset --native-report "$bubbleSource/benchmark_test.json" --output frontend/public/models/lighton3-bubble
npm.cmd run dev --workspace frontend -- --port 3001
```

Ouvrir `/sandbox/lighton3-bubble` : charger, vérifier les 16 bulles fixes puis
évaluer les 1 612 bulles du test local. Sauvegarder les résultats dans
`validation_webgpu_fixtures.json` et `validation_webgpu_test.json` dans le dossier
de sortie. Les contrôles incluent guillemets visibles, accents, majuscules,
différentes longueurs et une répétition pour vérifier la réinitialisation des états.

## Prétraitement et fidélité

Le processor natif sauvegardé utilise le backend **Torchvision** avec resize
**BICUBIC, antialias=True, uint8 CPU**. Canvas « high » ne reproduit pas ce filtre.
Le port JavaScript utilise le noyau Keys a=-0,5, des coefficients double normalisés,
une précision entière adaptative et deux passes uint8, comme PyTorch 2.8.
22 images de contrôle et cas synthétiques correspondent octet pour octet ; quatre
goldens synthétiques indépendants sont aussi conservés dans les tests Vitest.
Le budget est **65 536–1 048 576 pixels** sur une grille de 32 px, sans thumbnail
préalable de page. Guillemets, casse, accents et ponctuation restent intacts.

Les poids/projections sont FP16, les résidus, normalisations et états DeltaNet
FP32. Le vocabulaire de 248 320 tokens est complet. Aucun INT8/INT4.
Les graphes portables ONNX exposent les caches KV et états récurrents ;
le décodeur rapide WGSL réutilise ces mêmes poids avec l'ORT GPUDevice.
WebGPU est exigé, avec shader-f16 et fallback CPU désactivé. Les requêtes du worker
sont sérialisées, annulables, et leurs états sont libérés après chaque lecture.

Le contrôle d'encodeur compare aussi à une référence **native FP32** : cela
distingue une erreur d'export de l'arrondi du modèle BF16. La comparaison finale
de texte reste face au modèle BF16 réellement publié. Le test figé déjà inspecté
mesure la fidélité de conversion, pas un nouveau benchmark indépendant.

## Scorer, documenter et publier

```powershell
python docker_scripts/export_lighton3_ocr_onnx/score_bubble_results.py --directory frontend/public/models/lighton3-bubble --date 2026-10-13
docker run --rm -v "${bubbleSource}:/source:ro" -v "${bubbleOutput}:/output" -v "${bubbleScripts}:/scripts:ro" poneglyph/lighton3-onnx:5090 /scripts/export_lighton3_ocr_onnx/write_bubble_docs.py --directory /output --model-card /source/README.md
python docker_scripts/export_lighton3_ocr_onnx/publish_onnx.py --directory frontend/public/models/lighton3-bubble --repo-id Remidesbois/LightOnOCR-3-Poneglyph --model-card frontend/public/models/lighton3-bubble/model_card.md
```

La publication est conditionnée par le test complet, les 16 contrôles CUDA et
WebGPU, les contrôles pixels, aucune sortie vide/tronquée, et une augmentation de
CER ≤ 0,01 point. Le rapport agrégé et la courbe distinguent explicitement natif et
navigateur. La liste de publication exclut toutes les fixtures, images et sorties
individuelles. SHA-256 et tailles de tous les fichiers publiés sont vérifiés sur
la révision retournée, puis cette révision doit être épinglée dans
`frontend/src/lib/lighton3BubbleWebgpu.js`.

Dans l'app, **LightOnOCR 3 · bulles** s'utilise à côté de Surya et Falcon dans le
sélecteur OCR de bulles de l'annotation et de la sandbox. Chargement, progression,
cache, comparaison de plusieurs modèles et édition utilisent le même flux.
`NEXT_PUBLIC_LIGHTON3_BUBBLE_MODEL_BASE` permet une source alternative au build.
La page `/sandbox/lighton3-bubble` affiche la transcription et sa latence, ainsi
que les mesures WebGPU lorsque le rapport est disponible. Les contrôles locaux
sont réservés au mode développement.
