## Projet Poneglyph benchmark (2026-09-09)

Registry ID: `lighton-bbox`
Pinned revision: `fb35b7f4f99b282d5c2867d5cf23e66df0d03925`

| Metric | Value |
|---|---:|
| Page CER | 5,61 % |
| Global mean IoU | 65,00 % |
| F1@IoU 0.3 | 91,29 % |
| F1@IoU 0.5 | 79,62 % |
| F1@IoU 0.75 | 41,75 % |
| Exact bubble text | 90,63 % |
| Bubble-text CER | 1,15 % |
| Corrected combined score | 82,571 % |
| Avg inference | 6,18 |

- Dataset: Poneglyph validated full pages with text-zone bounding boxes
- Split: test held-out by page
- Date: 2026-09-09
- Samples: 221
- Hardware: NVIDIA GeForce RTX 5090 32 GB
- Protocol: Full-page image-only LightOnOCR generation of one exact text zone per line as Texte exact [x1,y1,x2,y2], with normalized integer coordinates in [0,1000]. Publication metrics use full-page CER, one-to-one global IoU assignment including unmatched GT zones at IoU 0, and micro-aggregated detection metrics over the complete held-out test split.
- Evidence: https://huggingface.co/Remidesbois/LightonOCR-2-1b-poneglyph-bbox/blob/fb35b7f4f99b282d5c2867d5cf23e66df0d03925/benchmark_lighton_bbox_corrected.json

> Generated from `shared/model-registry.json`; update the registry and rerun `node scripts/render-model-registry.mjs`.
