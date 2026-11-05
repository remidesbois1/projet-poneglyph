## Projet Poneglyph benchmark (2026-11-05)

Registry ID: `readernet-reading-order`
Pinned revision: `0f3cc90737d90d8c3810b7397c923a2fc7be24e9`

| Metric | Value |
|---|---:|
| Exact panel | 95,65 % |
| Bubble position accuracy | 96,62 % |
| Bubble assignment accuracy | 100,00 % |

- Dataset: Poneglyph polygon panels with bubbles from the shared YOLO26n detector
- Split: validation held out by page
- Date: 2026-08-27
- Samples: 138
- Hardware: CPU offline ONNX scoring
- Protocol: Shared bubble detections and annotated panels; pairwise Borda ranking independently inside each panel, with no global bubble sorter. Historical ordering benchmark inherited unchanged. Active detector weights retained; confidence calibrated to 0.25 on 2026-11-03. Fresh detector/text results are separate in metrics/retraining_20261103.json.
- Evidence: https://huggingface.co/Remidesbois/Poneglyph-ReaderNet/blob/0f3cc90737d90d8c3810b7397c923a2fc7be24e9/metrics/shared_detector_comparison.json

> Generated from `shared/model-registry.json`; update the registry and rerun `node scripts/render-model-registry.mjs`.
