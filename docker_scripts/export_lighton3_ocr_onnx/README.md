# LightOnOCR-3 Poneglyph — pages complètes, ONNX et WebGPU

La variante pour les bulles et son protocole distinct sont documentés dans [BUBBLE.md](BUBBLE.md).

Export du checkpoint **111**, retenu exclusivement sur le CER de validation.
Modèle natif : [Remidesbois/LightOnOCR-3-Poneglyph-bbox](https://huggingface.co/Remidesbois/LightOnOCR-3-Poneglyph-bbox).
Artefacts ONNX : [révision b3ae1dac](https://huggingface.co/Remidesbois/LightOnOCR-3-Poneglyph-bbox/tree/b3ae1dac8ae48a0f43a7b78dcc7aa8d48b97a63d/onnx), épinglée dans l'application après renommage ; poids inchangés.
SHA-256 BF16 : `3e7f25dbbdd87a0f1896af27aad5ee07f66364a24da4151d14ef6d125a98d14b`.

## Reproduire

Depuis la racine du dépôt, PowerShell :

```powershell
docker build -t poneglyph/lighton3-onnx:5090 docker_scripts/export_lighton3_ocr_onnx
$source = (Resolve-Path 'docker_scripts/finetune_lighton3_ocr_bbox/outputs/rtx5090-20261011-cer-refinement/final_merged').Path
$scripts = (Resolve-Path 'docker_scripts').Path
New-Item -ItemType Directory -Force frontend/public/models/lighton3-ocr | Out-Null
$output = (Resolve-Path 'frontend/public/models/lighton3-ocr').Path
docker run --rm -v "${source}:/source:ro" -v "${output}:/output" -v "${scripts}:/scripts:ro" poneglyph/lighton3-onnx:5090 /scripts/export_lighton3_ocr_onnx/export_lighton3.py --source /source --output /output
npm.cmd run dev --workspace frontend
```

Ouvrir `/sandbox/lighton3` dans Chrome/Edge avec WebGPU et `shader-f16`.
L'annotation et la sandbox proposent **LightOnOCR 3 · WebGPU** dans **Page entière**.
Un premier clic prépare le modèle ; le suivant lit l'image entière et ajoute ses
zones, avec les bbox générées par le modèle, dans l'ordre de sortie. Les textes
et coordonnées restent modifiables. L'annulation ou un changement de page empêche
l'import des résultats obsolètes.

En développement, l'export local est utilisé s'il existe. En production, les
artefacts sont chargés à une révision Hugging Face figée dans `lighton3Webgpu.js`.
`NEXT_PUBLIC_LIGHTON3_MODEL_BASE` permet de remplacer la source au build.
Les téléchargements reprennent par blocs, utilisent le cache du navigateur et
vérifient les tailles et SHA-256 avant l'initialisation.

## Contrat et architecture

- Qwen3.5 : 12 blocs vision, 24 blocs texte, attention globale + Gated DeltaNet,
  vocabulaire complet **248 320**, embedding et projection de sortie partagés.
- Prompt natif **`grounding`**, template sauvegardé avec **thinking désactivé**.
  EOS 248046 ; génération gloutonne, **1280 tokens maximum**.
- Images RGB entières : bord long maximum 2048 px, sans agrandir à cette étape,
  puis grille Qwen de 32 px et budget **65 536–4 194 304 pixels**.
  Frames temporelles dupliquées, normalisation [-1,1], interpolation des positions
  et RoPE multimodal natifs.
- Sortie native `![text](x1,y1,x2,y2)` puis texte. Coordonnées entières [0,1000],
  remises en pixels avec les dimensions **originales** de la page.
- Poids et projections FP16 ; résidus, normalisations et état récurrent FP32.
  Aucune quantification INT8/INT4 ni réduction du vocabulaire ou du budget image.
- Attention visuelle globale exacte par 16 groupes de requêtes : chaque groupe
  voit toutes les clés et valeurs. Les Split/Concat utilisent une hiérarchie de
  quatre entrées pour respecter les limites de buffers des shaders WebGPU.
- `vision.onnx`, `decoder.onnx`, `decode.onnx`, `prefill.onnx` : opérateurs ONNX
  standard, opset 17 ; `prefill-fast.onnx` avec RMSNormalization standard, opset 23.
  États DeltaNet et caches KV explicites, prefill par blocs de 32, sans Loop/Scan,
  sans opérateur Triton/FLA ou domaine ONNX personnalisé.
- Le chemin rapide réutilise le décodeur WGSL de Surya sur le GPUDevice d'ORT.
  Caches de capacité fixe, calcul de normalisation/conv/récurrence fusionné,
  GEMV avec sous-groupes ou réduction partagée, argmax hiérarchique du vocabulaire
  complet et lectures CPU groupées. Le chemin ONNX intégral reste disponible
  avec `load(progress, { mode: 'onnx' })`.
- ONNX Runtime Web **1.27.0** ; sessions WebGPU avec fallback CPU désactivé,
  features et états conservés sur GPU. Chaque requête réinitialise ses caches et
  les libère à la fin ; les demandes d'un même worker sont exécutées en série.

## Vérifier

`make_fixtures.py` produit six pages de validation stratifiées par longueur,
la page difficile 1491, l'ancien cas de boucle 1233 et une répétition de la
première page. Il conserve les sorties BF16 et les entrées numériques natives.
`validate_onnx.py --model /output --provider cuda --count 9` vérifie tous les
graphes, les états finis, les features visuelles, EOS et la génération complète.
Le fournisseur CUDA est exigé lorsqu'il est demandé ; aucun repli silencieux.
L'image de validation utilise ONNX Runtime GPU **1.26.0 / CUDA 12.8** pour
correspondre aux bibliothèques PyTorch de l'image de base ; le runtime du
navigateur reste **1.27.0 / WebGPU**.

`make_benchmark.py` expose les **189 pages de test figées** localement, en PNG RGB
sans nouvelle compression JPEG. La page de contrôle lance le véritable worker
de l'app sur ces images ; ses sorties sont scorées avec `grounding.py`, le même
protocole que le modèle natif. Le test était déjà inspecté pendant l'entraînement :
cette vérification mesure la conversion, pas un nouveau benchmark indépendant.

Les sorties BF16, FP16/FP32 et WebGPU ne sont pas identiques bit à bit. Le
redimensionnement Canvas peut également différer du processor Python. Les
métriques du modèle natif ne sont jamais attribuées au moteur navigateur.
Les mesures et limites propres au navigateur sont publiées dans
`onnx/webgpu_metrics.json` et la fiche `onnx/README.md` après vérification complète.

La publication utilise une liste explicite de fichiers. **Aucune page,
annotation, fixture ou transcription individuelle n'est publiée.**

Le dernier portage vérifié couvre 189 pages : **CER WebGPU 1,096 %** contre
**1,066 % BF16**, WER 2,195 %, texte exact par zone 89,67 %, par page 38,62 %,
aucune troncature ni sortie invalide. Médiane 3,51 s et p95 4,27 s sur la 5090,
modèle chargé. Les 9 contrôles ONNX CUDA passent aussi, avec répétition identique.
Voir le rapport agrégé publié pour le protocole et les intervalles bootstrap.

Après avoir sauvegardé les sorties navigateur dans `validation_webgpu_test.json`
et `validation_webgpu_fixtures.json`, scorer puis documenter avant publication :

```powershell
$frontend = (Resolve-Path 'frontend').Path
docker run --rm -v "${output}:/output" -v "${scripts}:/scripts:ro" -v "${frontend}:/frontend:ro" poneglyph/lighton3-onnx:5090 /scripts/export_lighton3_ocr_onnx/score_results.py --directory /output
python docker_scripts/export_lighton3_ocr_onnx/write_docs.py --directory frontend/public/models/lighton3-ocr --model-card "$source/README.md"
python docker_scripts/export_lighton3_ocr_onnx/publish_onnx.py --directory frontend/public/models/lighton3-ocr --model-card "$source/README.md"
```

Le script exige le compte propriétaire, les poids natifs attendus, un rapport
vérifié et toutes les empreintes avant de créer le commit. Il vérifie également
les métadonnées et les petits fichiers téléchargés depuis la révision publiée.
Le reçu `hf_publication.json` fournit la révision à épingler dans le frontend.
Le build de production et les 45 tests ciblés passent. Les 26 contrôles Surya
produisent toujours exactement leurs références après l'adaptation du vocabulaire
dans le décodeur partagé. Une image synthétique au budget maximal de 2048² passe
aussi : ce contrôle porte sur les limites GPU, pas sur la précision OCR.

Références : [contrat LightOnOCR-3](https://huggingface.co/lightonai/LightOnOCR-3-0.8B),
[ORT WebGPU et gestion des tenseurs GPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html).
