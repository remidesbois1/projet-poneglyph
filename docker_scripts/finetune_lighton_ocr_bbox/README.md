# LightOnOCR-2 BBox — RTX 5090

Fine-tuning pleine page de `lightonai/LightOnOCR-2-1B-bbox-base` pour Poneglyph.
Les images sont plafonnées à **1500 px côté long** et la sortie supervisée reste
strictement une zone par ligne :

```text
Texte exact [x1,y1,x2,y2]
```

Les coordonnées sont des entiers normalisés dans `[0,1000]`. LightOn-BBox reste
conditionné **image-only**, comme le modèle de base, l'export historique et le
runtime desktop Poneglyph. Le prompt textuel bbox partagé sert de contrat de
validation/documentation mais n'est pas injecté à LightOn pendant le SFT.

## Profil RTX 5090

- PyTorch 2.8 CUDA 12.8, BF16, TF32 et SDPA ;
- AdamW fusionné ;
- profil large par défaut : rsLoRA `r=128`, `alpha=256`, dropout `0`, sans `lm_head`
  ni DoRA, sur les projections attention+MLP du vision encoder et du language model ;
- `vision_projection` (le pont vision→langage, ~6,3 M paramètres) est entraîné en
  entier via `modules_to_save`, pour environ **159 M paramètres entraînables** au
  total (~13,7 % du modèle) ;
- selective logits LightOn : le loss ne matérialise que la fin supervisée quand
  `logits_to_keep` est disponible ;
- calibration automatique des batchs physiques `1,2,4,8` sur les pages les plus
  coûteuses, avec une limite par défaut de 90 % de VRAM et extrapolation du pic
  avant chaque batch supérieur pour éviter de faire tomber le contexte CUDA ;
- batch effectif conservé à 8 (`gradient_accumulation` est recalculé si le batch
  physique maximal ne rentre pas) ;
- gradient checkpointing désactivé par défaut et activé uniquement comme fallback
  si aucun batch utile ne tient nativement ;
- group-by-length, mémoire pinnée, 2 workers, prefetch 1 et workers non persistants
  pour exploiter la 5090 sans reproduire les OOM RAM/WSL ;
- validation loss bornée, génération bbox déterministe sur un sous-ensemble fixe,
  benchmark final complet sur le split test ;
- meilleur checkpoint sélectionné sur `eval_combined_score` ;
- checkpoints resumables, `--resume auto`, `--smoke-steps`, `--diagnose` et
  `--benchmark-only` ;
- dashboard HTML + JSON/JSONL/TensorBoard avec loss, LR, ETA, GPU/VRAM et métriques.

Il n'y a **aucune comparaison de modèle ni quality gate final**. Le pipeline fait
uniquement export si nécessaire → entraînement/reprise → merge → benchmark LightOn
→ upload optionnel.

## Lancement Windows

```powershell
cd docker_scripts\finetune_lighton_ocr_bbox
.\build_image.bat
.\run_pipeline.bat
```

Le profil réellement retenu est enregistré dans
`outputs_lighton_bbox/hardware_profile.json`. Le suivi principal se trouve dans
`outputs_lighton_bbox/training_dashboard.html` et le modèle final dans
`outputs_lighton_bbox/final_merged/`.

## Commandes utiles

```powershell
# Diagnostic CUDA/Blackwell sans charger le modèle
docker run --rm --gpus all lighton-ocr-bbox-finetune:latest `
  python train_lighton_bbox.py --diagnose --profile auto

# Petit run technique, sans benchmark final
docker run --rm --gpus all lighton-ocr-bbox-finetune:latest `
  python train_lighton_bbox.py --smoke-steps 2

# Reprendre automatiquement le dernier checkpoint sain
docker run --rm --gpus all lighton-ocr-bbox-finetune:latest `
  python train_lighton_bbox.py --resume auto
```

## Variables principales

- `LIGHTON_BBOX_TRAIN_BATCH`, `LIGHTON_BBOX_EFFECTIVE_BATCH=8` ;
- `LIGHTON_BBOX_BATCH_CANDIDATES=1,2,4,8` ;
- `LIGHTON_BBOX_CALIBRATION_MAX_VRAM_RATIO=0.90` ;
- `LIGHTON_BBOX_EVAL_BATCH=4`, `LIGHTON_BBOX_GEN_BATCH=4` ;
- `LIGHTON_BBOX_DATALOADER_WORKERS=2`, `LIGHTON_BBOX_PREFETCH_FACTOR=1` ;
- `LIGHTON_BBOX_GRADIENT_CHECKPOINTING=0` ;
- `LIGHTON_BBOX_LORA_R=128`, `LIGHTON_BBOX_LORA_ALPHA=256` ;
- `LIGHTON_BBOX_MODULES_TO_SAVE=vision_projection` ;
- `LIGHTON_BBOX_LR=1e-5`, `LIGHTON_BBOX_EPOCHS=3` ;
- `LIGHTON_BBOX_GEN_EVAL_MAX_SAMPLES=16`, `LIGHTON_BBOX_LOSS_EVAL_MAX_SAMPLES=64` ;
- `LIGHTON_BBOX_SKIP_UPLOAD=1` pour ne jamais publier ;
- `LIGHTON_BBOX_REQUIRE_UPLOAD=1` pour rendre une erreur d'upload fatale ;
- `LIGHTON_BBOX_FORCE_EXPORT=1` pour refaire l'export gelé ;
- l'export privé `r2://` utilise `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`,
  `R2_SECRET_ACCESS_KEY` et `R2_PAGES_BUCKET_NAME` (les mêmes variables que Surya) ;
- `LIGHTON_BBOX_TORCH_COMPILE=1` uniquement pour tester : les formes multimodales
  dynamiques peuvent rendre Inductor moins rapide que l'eager sur ce modèle.

Un échec d'upload Hugging Face **n'invalide pas** un entraînement et un benchmark
déjà terminés, sauf si `LIGHTON_BBOX_REQUIRE_UPLOAD=1` est explicitement activé.
