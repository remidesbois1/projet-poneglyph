## Projet Poneglyph benchmark (2026-10-30)

Registry ID: `surya-bubble`
Pinned revision: `24a589633c9c4b57663bfc019b132f11b45f9eb8`

| Metric | Value |
|---|---:|
| CER | 0,465 % |
| WER | 1,823 % |
| Exact match | 91,19 % |
| Token limit | 0,00 % |

- Dataset: Poneglyph validated single-bubble crops
- Split: test held-out by page
- Date: 2026-10-30
- Samples: 1691
- Hardware: NVIDIA RTX 5090 32 GB
- Protocol: Exhaustive generative evaluation with a 256-token budget, collapsed whitespace and explicit blank, hallucination and token-limit accounting.
- Evidence: https://huggingface.co/Remidesbois/surya-bubble-ocr-poneglyph/blob/24a589633c9c4b57663bfc019b132f11b45f9eb8/benchmark_test.json

> Generated from `shared/model-registry.json`; update the registry and rerun `node scripts/render-model-registry.mjs`.
