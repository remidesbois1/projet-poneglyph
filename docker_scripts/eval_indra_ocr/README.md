# Indra OCR — évaluation, fine-tuning et publication (5 novembre 2026)

Résultats et décisions : [`documentation/indra_ocr_bubble.md`](../../documentation/indra_ocr_bubble.md). Valeurs exactes : [`results-20261105.json`](results-20261105.json). Publication vérifiée : [`published-20261105.json`](published-20261105.json).

Les données privées (crops, prédictions, archives, checkpoints) restent dans `outputs/`, et les modèles Indra d'origine ainsi que le code source cloné dans `cache/` : les deux dossiers sont ignorés par Git.

## Environnement

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe indra-ocr==1.1.2 onnx==1.23.2 onnxruntime==1.31.1 pillow torch
.venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download as s; s('datalab-to/indra-ocr', revision='v1.1.0', local_dir='cache/models-v1.1.0', allow_patterns=['*.onnx','meta.json','lm.npz','LICENSE'])"
```

Les scripts Node réutilisent les dépendances de `../retrain_ppocr_pipeline` et `../audit_ppocr_pipeline` (runtime de production, `@napi-rs/canvas`, `onnxruntime-node`).

## Étapes

1. `node run_ppocr.cjs --split val|test|train` : crops de bulles depuis `dataset-v5`, boîtes YOLO de production et, pour val et test, le pipeline A (YOLO + PP-OCRv6) via les fonctions du worker.
2. `python sweep_val.py`, puis `node score.cjs --split val` : réglages zero-shot (mode de lecture, échelle du détecteur, modèle de langue), choisis sur la validation uniquement.
3. `python prepare_training.py --det-scale 0.75` : boîtes du détecteur Indra pour train et val, et archive `outputs/indra_rec_training.tar`. Le test n'entre jamais dans l'archive.
4. RunPod, avec un seul pod à la fois : `python runpod_job.py create --gpu "<id>"`, puis `guard` en arrière-plan, `upload`, `exec "bash code/run_training.sh"` (reconnaissance) ou `run_det_training.sh` (détection), `download`, `delete`. La création enregistre le pod avant d'attendre SSH, et le supprime s'il n'est pas joignable au bout de 10 minutes.
5. `python evaluate.py --split val|test --det indra --det-scale 0.75 --mode stitched --rec <rec.onnx> --det-model <det.onnx> --name <run>`, puis `node score.cjs`. Bootstrap par page et vitesse : `benchmark_speed.py --limit 300`.
6. `python release.py build`, `node run_js_pipeline.mjs --split val|test` pour la parité de l'implémentation navigateur, puis `python release.py publish` : envoi de la liste exacte des fichiers et vérification des SHA-256 téléchargés.
7. `python make_test_fixtures.py` régénère les références OpenCV et indra_ocr de `frontend/src/lib/indraOcr.test.js`.

`train_rec.py` et `train_det.py` entraînent les graphes ONNX publiés via `onnx_torch.py`, puis réécrivent les poids dans une copie du graphe d'origine. Un essai rapide sur CPU se lance avec `--limit 96 --epochs 2 --workers 2 --models cache/models-v1.1.0 --data outputs`.

## Coût

RunPod, une RTX PRO 6000 Blackwell : environ 0,30 $ pour la reconnaissance et environ 0,15 $ pour la détection. S'y ajoutent environ 0,48 $ de deux pods créés par erreur et supprimés : la boucle de création réessayait un autre GPU alors qu'un `--wait` expiré avait laissé un pod actif. `runpod_job.py` empêche désormais ce cas.
