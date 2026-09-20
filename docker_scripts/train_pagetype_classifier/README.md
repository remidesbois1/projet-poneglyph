# Page-type classifier training

Fine-tunes a light `MobileNetV3-Small` classifier for the page types labelled
by `scripts/pagetype_labelizer`:

- `cover`
- `story_page`
- `annexe`
- `summary`

The training set is read directly from the labelizer dataset. Original CBZ
archives are never read during training: only the extracted page images and
their `labels.json` are used.

## First baseline

From the repository root:

```powershell
python docker_scripts/train_pagetype_classifier/train.py `
  --held-out-volume "One Piece T07" `
  --epochs 25 `
  --batch-size 64
```

The held-out volume is never used for gradient updates. T07 is permanently
reserved as an unseen validation set and must not be used by `train_final.py`.

## Outputs

Each run is written to `runs/<timestamp>/`:

- `best.pt`: selected checkpoint (highest macro F1, then cover F1, on the held-out tome)
- `last.pt`: checkpoint at the final epoch
- `metrics.json`: full validation report and training configuration
- `history.json`: per-epoch losses and metrics
- `page_type_classifier.onnx`: static FP32 ONNX model
- `page_type_classifier.metadata.json`: class order and preprocessing contract

The exported ONNX graph has a single `input` tensor `[1, 3, 224, 224]` and a
`logits` output `[1, 4]`. Preprocessing is RGB, resize to `256`, center crop
to `224`, then ImageNet mean/std normalization. This model remains deliberately
small enough for a later `onnxruntime-web` WebGPU/WASM worker.

## Important interpretation

This baseline has only a few dozen `cover` examples. Read the held-out cover
recall and confusion matrix as an early signal, not a release metric. Keep
labelling diverse tomes, covers, color pages, title spreads and annexes before
freezing a browser threshold.

## Final model after validation

Once a held-out-volume run has selected an epoch count, train the deployment
candidate on every labelled page except the held-out test tome. This deliberately
has no new validation claim: its evidence remains the referenced held-out run.

```powershell
python docker_scripts/train_pagetype_classifier/train_final.py `
  --selection-metrics docker_scripts/train_pagetype_classifier/runs/<validation-run>/metrics.json `
  --epochs <best_epoch>
```

## Dataset & Benchmark (One Piece Tomes 1 à 7)

- **Hugging Face Model:** [Remidesbois/Poneglyph-Classifier](https://huggingface.co/Remidesbois/Poneglyph-Classifier)

### Dataset Annoté (1 415 pages)

| Tome | Total Pages | Story Page | Annexe | Cover | Summary |
| :--- | :---: | :---: | :---: | :---: | :---: |
| One Piece T01 | 210 | 187 | 15 | 7 | 1 |
| One Piece T02 | 211 | 167 | 33 | 10 | 1 |
| One Piece T03 | 213 | 172 | 31 | 9 | 1 |
| One Piece T04 | 196 | 165 | 21 | 9 | 1 |
| One Piece T05 | 196 | 164 | 22 | 9 | 1 |
| One Piece T06 | 193 | 158 | 25 | 9 | 1 |
| **One Piece T07 (Held-out Test)** | **196** | **164** | **22** | **9** | **1** |
| **TOTAL** | **1 415** | **1 177** | **169** | **62** | **7** |

### Held-out Test Metrics (Tome 7 - 196 pages)

- **Accuracy:** **99.49%** (195 / 196)
- **Macro F1:** **0.9845**
- **Loss:** **0.0476**

| Classe | Support | Précision | Rappel | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| `annexe` | 22 | 100.0% | 100.0% | 1.000 |
| `summary` | 1 | 100.0% | 100.0% | 1.000 |
| `story_page` | 164 | 99.39% | 100.0% | 0.997 |
| `cover` | 9 | 100.0% | 88.89% | 0.941 |

#### Matrice de Confusion

```
               Classes Prédites
               cover   story_page   annexe   summary
cover            8         1           0        0
story_page       0       164           0        0
annexe           0         0          22        0
summary          0         0           0        1
```

