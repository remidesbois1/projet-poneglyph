## Projet Poneglyph benchmark (2026-09-09)

Registry ID: `surya-bbox`
Pinned revision: `671cfe63672286ccfe629079a8d57f7ed967f3d5`

| Metric | Value |
|---|---:|
| CER | 1,49 % |
| WER | 3,44 % |
| Mean IoU | 92,36 % |
| Detection rate | 98,38 % |
| F1@IoU 0.5 | 98,36 % |
| F1@IoU 0.75 | 96,67 % |
| Combined score | 97,220 % |
| Avg inference | 2,75 |

- Dataset: Poneglyph validated full pages with text-zone bounding boxes
- Split: test held-out by page
- Date: 2026-09-09
- Samples: 166
- Hardware: NVIDIA GeForce RTX 5090 32 GB
- Protocol: Full-page multimodal generation of one exact text zone per line as Texte exact [x1,y1,x2,y2], with normalized integer coordinates in [0,1000]; OCR text and bbox localization scored jointly on the held-out page split.
- Evidence: https://huggingface.co/Remidesbois/surya-ocr-2-poneglyph-bbox/blob/671cfe63672286ccfe629079a8d57f7ed967f3d5/benchmark_surya_bbox.json

> Generated from `shared/model-registry.json`; update the registry and rerun `node scripts/render-model-registry.mjs`.
