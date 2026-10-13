## Projet Poneglyph benchmark (2026-10-13)

Registry ID: `lighton3-bubble`
Pinned revision: `dd066c12c4dcc990d71ebc78136e9c7c21c36522`

| Metric | Value |
|---|---:|
| CER WebGPU | 0,394 % |
| WER WebGPU | 1,410 % |
| Exact match WebGPU | 92,25 % |

- Dataset: Poneglyph frozen Surya single-bubble crops
- Split: test held out by page, previously evaluated for Surya and native LightOnOCR-3
- Date: 2026-10-13
- Samples: 1612
- Hardware: NVIDIA GeForce RTX 5090 32 GB / Chromium WebGPU (Blackwell)
- Protocol: Real app worker, ONNX Runtime Web 1.27.0 and fused WGSL decoder, mixed FP16/FP32, no quantization or CPU EP fallback. Image-only empty prompt, thinking disabled, greedy 256 tokens, 65536–1048576 pixels; native Torchvision uint8 antialiased bicubic resize reproduced in JavaScript. Whitespace collapsed; visible quotes, case, accents and punctuation preserved. Native BF16 CER 0.387984%; browser CER 0.393746% (205 vs 202 character errors), 1600/1612 literal outputs identical, no empty/truncated outputs. Conversion fidelity on the already inspected frozen test, not a new independent benchmark.
- Evidence: https://huggingface.co/Remidesbois/LightOnOCR-3-Poneglyph/blob/dd066c12c4dcc990d71ebc78136e9c7c21c36522/onnx/webgpu_metrics.json

> Generated from `shared/model-registry.json`; update the registry and rerun `node scripts/render-model-registry.mjs`.
