## Projet Poneglyph benchmark (2026-09-09)

Registry ID: `readernet-reading-order`
Pinned revision: `d97d4cd2903a7ebe49276a5269c4f3b7df608be7`

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
- Protocol: Shared bubble detections and annotated panels; pairwise Borda ranking independently inside each panel, with no global bubble sorter.
- Evidence: https://huggingface.co/Remidesbois/Poneglyph-ReaderNet/blob/d97d4cd2903a7ebe49276a5269c4f3b7df608be7/metrics/shared_detector_comparison.json

> Generated from `shared/model-registry.json`; update the registry and rerun `node scripts/render-model-registry.mjs`.
