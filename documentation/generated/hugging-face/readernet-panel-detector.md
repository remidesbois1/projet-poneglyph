## Projet Poneglyph benchmark (2026-09-09)

Registry ID: `readernet-panel-detector`
Pinned revision: `d97d4cd2903a7ebe49276a5269c4f3b7df608be7`

| Metric | Value |
|---|---:|
| mAP50 | 99,44 % |
| mAP50-95 | 98,39 % |

- Dataset: Poneglyph weighted polygon panel dataset
- Split: validation held out by page
- Date: 2026-08-21
- Samples: 257
- Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb
- Protocol: Ultralytics YOLO11n-seg validation at 1504 px with box and mask metrics over polygon panel annotations.
- Evidence: https://huggingface.co/Remidesbois/Poneglyph-ReaderNet/blob/d97d4cd2903a7ebe49276a5269c4f3b7df608be7/metrics/panel_detector_metrics.json

> Generated from `shared/model-registry.json`; update the registry and rerun `node scripts/render-model-registry.mjs`.
