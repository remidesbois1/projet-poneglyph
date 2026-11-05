## Projet Poneglyph benchmark (2026-11-05)

Registry ID: `indra-bubble`
Pinned revision: `fea60b5793979c05b483dd53e6536b920db6a17a`

| Metric | Value |
|---|---:|
| CER | 0,693 % |
| Exact match | 88,53 % |

- Dataset: Poneglyph validated reference bubble crops, same 1,221-bubble test as PP-OCRv6 Bubble Line
- Split: historical test held out by page, previously evaluated and inspected
- Date: 2026-11-05
- Samples: 1221
- Hardware: CPU, onnxruntime-node 1.24.3 running the browser implementation (frontend/src/lib/indraOcr.js)
- Protocol: Fine-tuned Indra OCR v1.1.0 det.onnx (YOLO line pseudo-labels) and rec.onnx; bubble crop at 0.75x for line detection, Indra tall line windows side by side, one greedy CTC pass, then the same train-derived lexicon/spacing/case rules as PP-OCRv6. NFKC, straight apostrophes, expanded ellipses and collapsed whitespace; case and punctuation retained. Settings chosen on validation only. Python reference 0.690% CER (1219/1221 identical raw texts); PP-OCRv6 production pipeline 0.902% on the same crops (paired page bootstrap difference -0.21 points, 95% CI -0.41 to -0.005). Not a full-page or blind benchmark.
- Evidence: https://huggingface.co/Remidesbois/indra-ocr-poneglyph-bubble/blob/fea60b5793979c05b483dd53e6536b920db6a17a/metrics/evaluation_20261105.json

> Generated from `shared/model-registry.json`; update the registry and rerun `node scripts/render-model-registry.mjs`.
