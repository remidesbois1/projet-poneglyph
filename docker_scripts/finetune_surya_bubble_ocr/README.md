# Surya OCR sur crops de bulles — RTX 5090

Fine-tuning de `datalab-to/surya-ocr-2` avec la transcription seule comme cible.
Le prompt partagé `ocr_surya_bubble` reste celui de l'inférence desktop.

## Entraînement

- Langage complet, merger multimodal et quatre derniers blocs vision entraînables :
  606 048 576 paramètres sur 665 701 440.
- BF16, TF32, AdamW fusionné, sans quantification ni gradient checkpointing.
- Profil `auto` : batch physique 32 / accumulation 1 sur SM120, sinon 16 / 2.
- LR langage `1.2e-5`, merger `6e-6`, vision `3e-6` ; warmup 80 étapes,
  cosine jusqu'à 10 % du LR initial, cinq époques, patience 2 sur le CER.
- Loss par bulle, pondérations légères des textes courts et onomatopées,
  augmentations manga conservatrices. Normalisation exacte sur la fenêtre
  d'accumulation, y compris si le dernier microbatch est incomplet.
- Validation loss et génération sur toutes les bulles de validation par défaut.
  Checkpoint retenu sur le CER de génération. Pour une exploration rapide,
  `SURYA_GEN_EVAL_MAX_SAMPLES=512` utilise un sous-ensemble stratifié fixe.
- Test réservé complet après sélection du checkpoint. Aucune comparaison avec
  d'autres modèles n'est exécutée ou ajoutée au rapport.

## Accélérations et lien avec l'inférence

Les modules desktop `surya_mlp_kernels.py` et `surya_hybrid_flash_kvcache.py`
spécialisent le décodage mono-token / batch 1, sans backward. Leur réutilisation
brute casserait l'entraînement. `training_acceleration.py` reprend les fusions
SwiGLU et RMSNorm avec les kernels autograd de Liger 0.8.0 ; les grandes
projections restent des GEMM cuBLAS.

La tête vocabulaire reçoit **uniquement les états cachés qui prédisent un token
supervisé**. Les positions décalées, les cibles et leurs poids sont calculés sur
CPU dans le collator : pas de `.item()` CUDA dans la loss, ni de logits sur le
prompt, l'image ou le padding. Cross-entropy fusionnée, conservation d'EOS même
si PAD partage son identifiant, vérification exacte du suffixe assistant.

Les 18 couches DeltaNet utilisent FLA + causal-conv1d avec backward ; les couches
d'attention complète utilisent SDPA. Le regroupement par longueur réduit le
padding. La validation génère par lots regroupés ; le cache KV est activé puis
l'état entraînement est restauré même en cas d'erreur. Les kernels desktop de
cache Flash/CUDA Graphs ne sont pas installés dans le modèle entraînable.
Le collator garde aussi en mémoire les templates et leurs tokens déterministes
entre les époques (`SURYA_TEXT_CACHE_ITEMS`, 16 384 par défaut, 0 pour désactiver).
Les images augmentées ne sont pas mises en cache : l'augmentation reste aléatoire.
Les transferts de lots sont non bloquants quand la mémoire épinglée et CUDA sont
disponibles (`SURYA_NON_BLOCKING_TRANSFERS=0` pour les désactiver).

La recette rapide conserve les paramètres en BF16. Pour éviter d'arrondir les
petites mises à jour, `SURYA_WEIGHT_DTYPE=float32` garde les poids et états Adam en
FP32 tout en utilisant BF16 pour les calculs. Cette variante consomme davantage de
VRAM : commencer par `SURYA_TRAIN_BATCH=16` et `SURYA_GRAD_ACCUM=2`, puis mesurer
le débit et la qualité avant de modifier le batch. `SURYA_MODEL_REVISION` permet de
figer la révision Hugging Face du modèle initial ; elle est notée dans `run_config.json`.

`torch.compile` global reste opt-in : les formes image variables et les kernels
DeltaNet provoquent des coûts de compilation qui doivent être mesurés avant de
l'activer. Aucun fallback silencieux de compilation.

Références : [Liger 0.8.0](https://github.com/linkedin/Liger-Kernel/tree/v0.8.0),
[SDPA PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

## Données

L'export lit les annotations validées de Supabase et les pages HTTP ou privées
R2. Il conserve un split strict par page, refuse les téléchargements incomplets
et les splits vides. Les hashes SHA-256 des métadonnées sont dans `run_config.json`.
Les credentials sont chargés depuis les fichiers `.env` locaux ; ils ne sont
pas copiés dans l'image.

Pour actualiser un export sans déplacer ses anciennes pages de test ou de validation
dans l'entraînement, définir `SURYA_SPLIT_REFERENCE_DIR` vers l'ancien dossier de
dataset et `SURYA_DATASET_DIR` vers un nouveau dossier distinct. Les nouvelles pages
sont affectées de manière déterministe ; un split manquant ou incohérent interrompt
l'export. Sans référence explicite, le split initial reste aléatoire par page.

Variables nécessaires : `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` et, pour R2,
`R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_PAGES_BUCKET_NAME`.
L'export utilise la vue Supabase `ocr_training_bubbles` : elle doit être présente
et accessible au rôle de service, afin d'exclure les pages contenant des textes
issus uniquement d'un consensus automatique.

## Lancement local

Depuis ce dossier, sous Windows :

```bat
build_image.bat
run_pipeline.bat
```

Le build prend **la racine du dépôt** comme contexte. PyTorch 2.8 / CUDA 12.8,
Transformers 5.14.1, FLA 0.5.2, causal-conv1d 1.6.2.post1 compilé SM120,
Liger 0.8.0. Les caches Triton/Inductor restent sur un volume Linux natif.
L'image désactive l'upload (`SURYA_SKIP_UPLOAD=1`). Pour le demander explicitement,
passer `SURYA_SKIP_UPLOAD=0` et configurer `HF_TOKEN` / `HF_REPO`.

Lancement direct dans l'environnement CUDA déjà préparé :

```bash
python export_dataset.py
python -m unittest test_training -v
SURYA_RUN_GPU_TESTS=1 python -m unittest test_training.KernelTests -v
SURYA_OUTPUT_DIR=/workspace/smoke python train_surya_bubble_ocr.py --train-smoke-steps 3 --smoke-eval-samples 16
SURYA_GEN_EVAL_BATCH=32 SURYA_IMAGE_CACHE_ITEMS=8192 python train_surya_bubble_ocr.py
```

Le smoke consomme de vrais batches, exécute forward/backward/AdamW puis loss et
génération sur validation. Utiliser un dossier distinct du run principal.
`SURYA_RESUME_FROM_CHECKPOINT=auto` reprend le dernier checkpoint ; `0` repart du
modèle de base. Choisir un nouveau `SURYA_OUTPUT_DIR` pour préserver un ancien run.
Les poids exportés gardent le format Transformers standard.

Pour comparer plusieurs recettes sans utiliser le test pour choisir le modèle,
`SURYA_DEFER_FINAL_TEST=1` termine chaque entraînement après la sélection sur
validation. `--benchmark-only --benchmark-split val --benchmark-output ...` évalue
un modèle existant sur la validation ; une fois le modèle choisi,
`--benchmark-only --benchmark-split test --benchmark-output ... --write-report`
exécute le test réservé et écrit le rapport dans `SURYA_OUTPUT_DIR`.

Pour une 3090, compiler avec `--build-arg TORCH_CUDA_ARCH_LIST=8.6
--build-arg CAUSAL_CONV_CUDA_ARCH=86`, puis ajuster le batch de génération.

## Rapport et mesures

- `training_dashboard.html` : 12 graphiques, rafraîchissement pendant le run.
- `report.html` : rapport autonome avec figures intégrées et erreurs à examiner.
- `plots/` : figures d'apprentissage, ressources GPU, qualité, erreurs ; PNG et SVG.
- `metrics.jsonl`, `metrics.csv` : historique exploitable, loss, CER, WER, exact match,
  LR, gradients, débit, temps/étape, VRAM allouée/réservée/pic, utilisation,
  puissance et température GPU.
- `validation_step_*.json` : métriques et prédictions de chaque validation.
- `final_merged/benchmark_test.json`, `predictions_test.csv`, `error_analysis.json` :
  détail du test, CER par longueur/catégorie, caractères insérés/supprimés/substitués,
  confusions fréquentes, sorties vides et limite de génération, débit de génération,
  intervalles de confiance à 95 % par bootstrap de pages.
- `training_summary.json`, `run_config.json`, `trainer_state.json` : état, recette,
  empreintes des données et checkpoint sélectionné. TensorBoard via `SURYA_REPORT_TO=tensorboard`.

Le `learning_rate` standard du Trainer correspond au **premier groupe (vision)** ;
les autres groupes suivent le même multiplicateur cosine. La VRAM instantanée est
mesurée après l'étape, le pic couvre aussi les activations. Le débit par étape
utilise le batch nominal et exclut validation/sauvegarde ; le dernier batch peut
être incomplet. Les mesures GPU sont échantillonnées, pas intégrées en continu.
Les catégories OCR se recoupent et le bootstrap est groupé par page.

Pour ouvrir le rapport sans dépendance web :

```powershell
node serve_report.cjs outputs_surya_bubble_ocr/rtx5090-20261005 8765
```

Le serveur écoute uniquement `127.0.0.1`. Ouvrir `http://127.0.0.1:8765`.
Les fichiers HTML fonctionnent aussi hors ligne.

## Mesure réelle du chemin GPU — 5 octobre 2026

Sur le batch de 32 bulles les plus coûteuses du nouvel export (séquence 352),
forward + backward + clipping + AdamW : **0,701 s** sur le dernier pas chaud,
**23,70 Gio** alloués au pic et **24,27 Gio** réservés. Le premier pas a pris
42,93 s pour compiler les kernels. Cette mesure utilise trois répétitions du même
batch ; ce n'est pas le débit de tout le pipeline.

Les tests vérifient les cibles/EOS sur les deux côtés de padding, l'égalité
loss/gradients avec la formulation dense, l'accumulation inégale, la restauration
d'état après échec de génération et l'absence de comparaisons dans les exports.
Les deux tests CUDA vérifient le forward/backward de SwiGLU, RMSNorm Qwen et de la
cross-entropy pondérée. Le smoke réel couvre l'intégration Transformers.

## Run terminé le 5 octobre 2026

Sortie : `outputs_surya_bubble_ocr/rtx5090-20261005/`.
7 496 bulles / 860 pages d'entraînement, 1 648 / 184 de validation,
1 612 / 185 de test. Cinq époques, 1 175 mises à jour ; meilleur checkpoint
`checkpoint-705` (époque 3). Durée Trainer, validations comprises : 15 min 44 s ;
cycle complet avec export final, test et rapport : 17 min 03 s (hors préparation).
Pic VRAM allouée pendant le run : 23,54 Gio.

| Test réservé | Résultat |
| --- | ---: |
| CER | 0,471 % |
| WER | 1,843 % |
| Exact match | 91,32 % |
| Distance de Levenshtein moyenne | 0,1520 |
| Sorties vides / tronquées | 0 / 0 |
| IC 95 % du CER, bootstrap par page | 0,350–0,602 % |

Les poids `final_merged/model.safetensors` ont été rechargés avec Transformers
standard, sans patch Liger : aucune clé manquante/inattendue/incompatible et
transcriptions de contrôle générées (`reload_check.json`).
Le rapport contient 18 graphiques répartis en quatre planches PNG/SVG, plus le
tableau de bord, toutes les prédictions CSV et l'analyse d'erreurs.
