## Projet Poneglyph benchmark (2026-10-13)

Registry ID: `lighton3-bbox`
Pinned revision: `41570a243115f8ed723c8fc8b761a5aa7735aa03`

| Metric | Value |
|---|---:|
| Native CER | 1,07 % |
| Native WER | 2,22 % |
| Exact bubble text | 90,22 % |
| Exact page text | 40,21 % |
| F1@IoU 0.5 | 98,75 % |

- Dataset: Poneglyph validated full pages with text-zone bounding boxes
- Split: frozen test, previously evaluated and inspected
- Date: 2026-10-12
- Samples: 189
- Hardware: NVIDIA GeForce RTX 5090 32 GB
- Protocol: Native Transformers 5.14.1 BF16 CUDA, grounding prompt, thinking disabled, greedy 1280 tokens, long edge <=2048 px; NFC and normalized whitespace with case and punctuation preserved; one-to-one spatial assignment at IoU >= 0.1, missed/extra zones penalized. Native results, not ONNX/WebGPU results.
- Evidence: https://huggingface.co/Remidesbois/LightOnOCR-3-Poneglyph-bbox/blob/41570a243115f8ed723c8fc8b761a5aa7735aa03/benchmark_native.json

> Generated from `shared/model-registry.json`; update the registry and rerun `node scripts/render-model-registry.mjs`.
