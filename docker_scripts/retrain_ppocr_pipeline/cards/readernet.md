---
license: agpl-3.0
library_name: onnxruntime
tags:
- onnx
- comic
- manga
- object-detection
- image-segmentation
- reading-order
---

# Poneglyph ReaderNet

ReaderNet detects manga bubbles and panels, assigns each bubble to a panel, then ranks panels and bubbles inside each panel. The 2026-11-03 training release adds **actual retrained YOLO candidates and an inference-equivalent recovered native checkpoint**. Production keeps the existing bubble ONNX weights with confidence calibrated to **0.25** on validation. Panel and ordering weights are unchanged.

## Bubble detector selection, 2026-11-03

1,193 annotated validation bubbles on 129 pages, actual production preprocessing and one-to-one IoU >= 0.5 matching:

| Detector | Confidence | Missed GT | Unmatched predictions |
|---|---:|---:|---:|
| Existing production | 0.45 | 16 | 2 |
| **Existing production, selected** | **0.25** | **7** | **2** |
| Retrained YOLO26n, 800 px | 0.25 | 8 | 14 |
| Recovered detector, trained head, 800 px | 0.25 | 7 | 5 |
| Retrained YOLO26n, 1024 px refinement | 0.25 | 7 | 8 |

Selection minimized missed annotations without increasing unmatched predictions above the previous validation operating point. Candidates did not improve this criterion, so none replaces active weights. Unmatched predictions are not verified false positives: annotation completeness is unproven. All candidates include SFX training examples without a class-based skip.

Training used RunPod RTX 4090, seed 42, 1,065 train and 127 validated detector-validation pages, with historical page splits preserved. Five pages with unvalidated annotations were excluded from detector training/validation; two out-of-bounds train boxes were clipped and recorded. Thresholds and candidate selection were frozen before the selected final full-page test.

With the [new PP-OCR recognizer](https://huggingface.co/Remidesbois/pp-ocrv6-one-piece-bubble-line-rec), the selected pipeline on 129 historical test pages / 1,221 bubbles reduces GT-conditioned final CER from **1.6245% to 1.0141%** and missed annotations from **12 to 4**. Unmatched detections increase from 6 to 9 on test. Text CER does not penalize these extra predictions or measure reading order. This historical holdout has already been inspected; it is not blind. The active detector also passed 20 real-page Chrome WASM checks.

## Artifacts

- `bubble_detector.onnx`: active YOLO26n, unchanged SHA256 `fa28ece56ba9e5ccf4361fbb4d2533e088906b0a0d1cd187c02422ba7d6f5688`, input `[1,3,800,800]`, output `[1,300,6]`. Letterbox in RGB with gray fill, production confidence **0.25**.
- `bubble_detector.pt`: recovered trainable native graph using **Ultralytics 8.4.48**, saved with PyTorch 2.8.0. All 366 parameter tensors were recovered; missing training-head tensors were copied from the exported head. Original batch-normalization statistics cannot be recovered; neutral BN preserves inference, and running BN statistics must stay frozen during further training. This is a reconstruction, not the original training checkpoint. Parity on 4 real pages / 39 detections: maximum box difference 0.00003052 px, maximum score difference 0.00000303.
- `retraining/2026-11-03/yolo800/`: newly trained **Ultralytics 8.4.201** candidate `.pt` + `.onnx`, input 800 px, output `[1,300,6]`.
- `retraining/2026-11-03/yolo1024/`: 1024 px refinement of that candidate, **Ultralytics 8.4.201**, `.pt` + `.onnx`, input `[1,3,1024,1024]`, output `[1,300,6]`.
- `retraining/2026-11-03/legacy-head800/`: trained head candidate from the recovered native model, **Ultralytics 8.4.48**, input 800 px. Do not confuse it with the retained initial checkpoint.
- `panel_detector.onnx`: unchanged YOLO11n-seg, input `[1,3,1504,1504]`. The application represents its detected bounding boxes as four-point rectangular polygons for ordering.
- `ordering.onnx`: unchanged independent MLP heads, `panel_features[*,96] -> panel_logits[*]` and `bubble_features[*,102] -> bubble_logits[*]`.
- `model_manifest.json`: active tensor contracts, calibration, file hashes and provenance.
- `metrics/retraining_20261103.json`: current aggregate paired results, full validation threshold curves, limitations and evidence hashes. Candidate `training.json` files contain sanitized configs, actual completed epochs and native validation metrics.

Use an isolated Ultralytics **8.4.48** installation for the recovered/head models; 8.4.201 changes the YOLO26 architecture. Native mAP and production operating-point measurements use different preprocessing/protocols and must not be interchanged.

## Reading algorithm and historical ordering evidence

Detect panels/bubbles; rank panels by summed directed pairwise sigmoid probabilities; assign each bubble by center containment (smallest containing panel, otherwise nearest boundary); rank bubbles independently inside each panel. There is no global bubble sorter.

The following earlier ordering benchmarks are inherited evidence, not newly retrained ordering results:

- Annotated polygons: 51/51 validation panel-order pages exact.
- Shared bubble detections and annotated panels: 132/138 exact multi-bubble panels (95.65%).
- Browser-compatible rectangular panel polygons: 131/139 exact multi-bubble panels (94.24%).
- Predicted panels and reference bubble boxes: 125/139 exact multi-bubble panels (89.93%), assignment 428/438 (97.72%).

Historical protocols remain in `metrics/`. The November report separately measures exploratory global order of matched annotations and clearly excludes misses/extras from that score. Detector exports retain Ultralytics AGPL-3.0 licensing.
