# Surya OCR Poneglyph — ONNX et WebGPU

Fine-tuning non-bbox du 5 octobre 2026, sélectionné sur le CER de validation.
SHA-256 du checkpoint natif :
`67fa1733d2541d6305cffef1e5202e47930a5f5393d1901adda52e7daab05629`.

La page `/sandbox/surya` et le moteur **Surya · WebGPU** exécutent l'OCR dans
un worker du navigateur. Les images restent sur l'appareil. Chrome/Edge
avec WebGPU et `shader-f16` sont nécessaires. Une bulle RGB par appel.
Les téléchargements reprennent par blocs de 4 Mio, sont mis en cache et
vérifiés par SHA-256. Environ 1,34 Go au premier chargement.

## Chemin optimisé

ONNX Runtime Web 1.27.0 exécute la vision et le préremplissage du prompt.
Un décodeur WGSL spécialisé prend ensuite le relais sur le **même GPUDevice**.
Les états du prompt sont copiés directement entre buffers GPU.

- Matrices FP16, accumulations des projections et états DeltaNet FP32.
- Poids ONNX réutilisés sans seconde archive : un index décrit leurs
  offsets ; les matrices sont transposées une fois au chargement.
- Normalisations, convolution causale, activation et récurrence DeltaNet
  fusionnées ; ajout du résidu fusionné avec la normalisation suivante.
- GEMV avec sous-groupes lorsqu'ils sont disponibles, réduction en mémoire
  partagée sinon.
- Caches KV de capacité fixe, mis à jour sur place ; attention sur la seule
  partie active, sans recopier l'historique à chaque token.
- Attention sur les valeurs répartie entre huit groupes de travail, puis
  réduction et gate ; rotations RoPE préparées une fois par transcription.
- Sélection du token et incrément des positions sur GPU. Jusqu'à huit
  étapes autorégressives par soumission, avec lecture CPU groupée. Les
  premiers tokens sont lus individuellement pour l'affichage progressif.
- Arrêt à EOS même au milieu d'une soumission ; les tokens calculés après
  EOS ne sont jamais renvoyés. La limite de 256 tokens borne également les
  soumissions. États réinitialisés à chaque image.
- Préremplissage sans projection inutile du vocabulaire entre les blocs,
  avec 79 normalisations RMS regroupées dans des opérateurs ONNX standard.

Le vocabulaire complet et les poids sont conservés : aucune quantification
INT8/INT4, aucun raccourcissement du prompt ni réduction du budget d'image.
Le chemin ONNX d'origine reste disponible avec `load(progress, { mode: 'onnx' })`.
Les anciens manifestes restent compatibles : sans index WGSL, le moteur
utilise ONNX. Les sessions imposent `disable_cpu_ep_fallback: '1'`.

## Fichiers de distribution

| Fichier | Usage |
| --- | --- |
| `vision.onnx` + `vision.onnx.data` | Encodeur visuel, 12 blocs |
| `prefill-fast.onnx` | Prompt avec sorties d'état et RMSNormalization, opset 23 |
| `prefill.onnx` | Variante sans fusion RMS, opset 17 |
| `decoder.onnx` | Prompt ONNX historique, avec logits, opset 17 |
| `decode.onnx` | Décodage ONNX à un token, opset 17 |
| `decoder.onnx.data` | Poids communs à tous les chemins de texte |
| `decode_weights.json` | Index des matrices et petites constantes WGSL |
| `browser_manifest.json` | Architecture, prompt, tailles et empreintes |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer du modèle |

Le décodeur est dans `frontend/src/lib/suryaFusedDecoder.js`, l'orchestration
dans `suryaWebgpu.js`, le worker dans `frontend/src/workers/surya.worker.js`.
Le prompt résout la récurrence triangulaire DeltaNet par une série de
Neumann finie sur des blocs de 32 tokens. Les petites matrices de
récurrence et les résidus restent en FP32.

## Prétraitement et précision

Même prompt français, budget de 65 536 à 1 048 576 pixels, grille de 32 px,
normalisation [-1,1], ordre des patches, interpolation des positions et
RoPE multimodal que le processor sauvegardé. Frames temporelles dupliquées
pour les images fixes. Génération gloutonne, EOS 2, 256 tokens maximum.
Espaces normalisés et paire de guillemets extérieurs retirée.

Le redimensionnement Canvas peut différer de Pillow. Les réductions GPU,
les activations FP16 et la référence native BF16 ne sont pas identiques bit
à bit. Les résultats du modèle natif ne sont donc pas attribués au moteur
WebGPU : le banc mesure ses propres CER, WER et transcriptions exactes.

## Mesures vérifiées sur RTX 5090

Application Next.js compilée, worker de production, artefacts chargés depuis
la révision HF `1d3698b88de708cc7b7cb8d349b4255715b70bc5`, modèle chaud.
Trois passes sur 24 bulles de validation fixes et deux cas de résistance :
78 transcriptions conformes. Les cas de résistance sont exclus des agrégats
de vitesse ci-dessous ; les téléchargements et le chargement sont exclus.

| Mesure | Résultat |
| --- | ---: |
| Latence médiane / p95 par bulle | 350 / 529 ms |
| Premier token médian | 237 ms |
| Débit de décodage agrégé | 285,4 tokens/s |
| Test complet indépendant | 1 612 bulles |
| CER / WER du moteur optimisé | 0,492 % / 1,893 % |
| Transcriptions exactement correctes | 91,07 % |
| Sorties vides / tronquées | 0 / 0 |

Douze sorties du test complet diffèrent du benchmark natif enregistré.
Les mesures de qualité portent sur les sorties WebGPU réelles. Ces chiffres
valident la RTX 5090 testée ; ils ne prédisent pas les performances d'autres GPU.
Le [résumé JSON](webgpu_metrics.json) conserve les agrégats et les empreintes
des artefacts sans aucune image ni transcription individuelle.

## Reproduire

Depuis la racine du dépôt, PowerShell :

```powershell
docker build -t poneglyph/surya-onnx:optimized docker_scripts/export_surya_ocr_onnx
$source = (Resolve-Path 'docker_scripts/finetune_surya_bubble_ocr/outputs_surya_bubble_ocr/rtx5090-20261005/final_merged').Path
New-Item -ItemType Directory -Force frontend/public/models/surya-ocr | Out-Null
$output = (Resolve-Path 'frontend/public/models/surya-ocr').Path
docker run --rm -v "${source}:/source:ro" -v "${output}:/output" poneglyph/surya-onnx:optimized --source /source --output /output
npm.cmd run dev --workspace frontend
```

L'export fonctionne sur CPU. Pour compléter un export existant :
`export_decode_weights.py --source /source --output /output`, puis
`export_prefill.py --directory /output --fused-norms`.
`test_optimizations.py` vérifie par exécution ORT que la suppression des
logits et la fusion RMS préservent les états récurrents.

En développement, le worker détecte `/models/surya-ocr/browser_manifest.json`.
Sinon il utilise la révision HF épinglée dans `suryaWebgpu.js`.
`NEXT_PUBLIC_SURYA_MODEL_BASE` permet d'imposer une URL de dossier ; il faut
reconstruire/redémarrer Next.js après changement.

## Banc de mesure et contrôle complet

```powershell
node frontend/bench/serve-surya.mjs
# Ouvrir http://127.0.0.1:3002/bench/surya.html
```

Le banc utilise le même worker que l'application. Il sépare l'échauffement,
les répétitions, les 26 cas (dont deux cas de résistance), le contrôle de
128 bulles et le test complet de 1 612 bulles. Il propose un balayage des
GEMV et tailles de soumission, et un profil GPU séparé avec des requêtes
de timestamps. Désactiver le profilage pour mesurer le débit.

Les fixtures sont privées et ignorées par Git :

- `make_fixtures.py --source /source --dataset /dataset --output /output --count 24 --stress`
  produit des références natives et des entrées de contrôle.
- Ajouter `--split test --count 128 --no-arrays` pour le contrôle étendu.
- `make_test_fixtures.py --dataset /dataset --benchmark /source/benchmark_test.json --output /output/full1612`
  réutilise l'évaluation complète après vérification des IDs et des textes,
  sans nouvelle sélection des cas.

Les données JSON visibles alimentent
`create_optimization_report.py --directory <dossier>` : métriques JSON,
mesures CSV, erreurs détaillées locales et neuf graphiques PNG/SVG dans un
rapport HTML autonome. Modèle déjà chargé, téléchargement et initialisation
exclus. Le délai du premier token inclut prétraitement, vision, prompt et
préparation du décodeur. Le débit inclut la détection d'EOS et les calculs
supplémentaires des soumissions groupées : somme des tokens texte divisée
par la somme des durées après le premier token.

`upload_onnx.py --directory <dossier>` publie uniquement les fichiers du
manifeste, le tokenizer et cette documentation. Il vérifie le checkpoint
natif et les empreintes distantes. **Les crops, prédictions individuelles,
diagnostics et rapports contenant des références ne sont jamais publiés.**
Le résumé agrégé `webgpu_metrics.json` peut être publié séparément avec
cette documentation après vérification des mesures de production.

Références : [ORT WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html),
[RMSNormalization](https://onnx.ai/onnx/operators/onnx__RMSNormalization.html),
[WGSL](https://www.w3.org/TR/WGSL/).
