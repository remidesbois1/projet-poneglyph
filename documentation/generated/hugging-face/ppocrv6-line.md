## Projet Poneglyph benchmark (2026-11-05)

Registry ID: `ppocrv6-line`
Pinned revision: `a6dddec384c76a1ac3c2125f92403e775a3929d7`

| Metric | Value |
|---|---:|
| CER | 0,904 % |
| Exact match | 85,01 % |

- Dataset: Poneglyph validated reference bubble crops reconstructed from production-detected text lines
- Split: historical test held out by page, previously evaluated and inspected
- Date: 2026-11-03
- Samples: 1221
- Hardware: NVIDIA GeForce RTX 4090 24 GB, native PyTorch 2.8.0 CUDA FP32
- Protocol: Reference bubble crops; production YOLO line detection at 0.10, horizontal stitching, 48 px height with gray padding and dynamic width 960-3200, BGR normalization, PP-OCRv6 CTC, frozen train-derived lexicon/spacing/case rules. NFKC, straight apostrophes, expanded ellipses and collapsed whitespace; case and punctuation retained. Native corpus results; Chrome WebGPU validated separately on 100 validation crops. Not a full-page or blind benchmark.
- Evidence: https://huggingface.co/Remidesbois/pp-ocrv6-one-piece-bubble-line-rec/blob/a6dddec384c76a1ac3c2125f92403e775a3929d7/metrics/retraining_20261103.json

> Generated from `shared/model-registry.json`; update the registry and rerun `node scripts/render-model-registry.mjs`.
