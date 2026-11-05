---
pipeline_tag: image-to-text
license: apache-2.0
tags:
- ocr
- manga
- ppocrv6
- onnx
- webgpu
---

# PP-OCRv6 One Piece Bubble Line Recognition

Fine-tuned PP-OCRv6 recognizer for Poneglyph, retrained on RunPod RTX 4090 on 2026-11-03 using the current production image pipeline. Native trainable weights and dynamic-width ONNX exports are included. The bundled YOLO line detector is unchanged; YOLO artifacts retain their Ultralytics AGPL-3.0 licensing.

## Paired evaluation

The same historical test split was evaluated before and after training: 129 pages, 1,221 annotated bubbles, 40,136 normalized reference characters. This split has already been evaluated and inspected; it is not a new blind benchmark.

| Test scope | Previous production | Selected release |
|---|---:|---:|
| Recognition from reference bubble crops, final CER | 1.0689% (429 errors) | **0.9044% (363 errors)** |
| Reference crops, exact text | 1,015 / 1,221 | **1,038 / 1,221** |
| Reference crops, empty outputs | 3 | **0** |
| Full pages, final CER | 1.6245% (652 errors) | **1.0141% (407 errors)** |
| Full pages, missed annotated bubbles | 12 | **4** |
| Full pages, unmatched detections | 6 | 9 |

Crop predictions use native PyTorch CUDA FP32. Full-page results use the actual production worker functions through Skia Canvas and ONNX Runtime Node CPU. Text is matched to annotations one-to-one at IoU >= 0.5; missed bubbles receive empty text. Unmatched predictions are counted separately, not included in CER. Annotation completeness is unproven, so these counts do not establish precision. Case and punctuation are scored after NFKC, apostrophe, ellipsis and whitespace normalization.

The full-page configuration uses the existing [ReaderNet bubble detector](https://huggingface.co/Remidesbois/Poneglyph-ReaderNet), confidence **0.25**, and this new recognizer. Newly retrained page-level YOLO candidates did not improve the selected validation operating point and are published separately in ReaderNet rather than replacing active detector weights.

## Image and text contract

1. Crop the bubble; detect lines at confidence **0.10** with `onnx/bubble_line_detector_yolo26n.onnx`.
2. Sort lines and stitch horizontally with **2 px** crop padding and **8 px** gaps. Keep the full crop when no line is detected.
3. Resize to height **48**, preserve aspect, clamp/pad width to **960–3200**, fill padding with **#808080**. Normalize **BGR** channels by `(uint8 / 255 - 0.5) / 0.5`.
4. Greedy CTC: collapse repeated tokens and remove blank ID 0.
5. Apply the frozen train-derived lexicon, spacing and case rules in `onnx/browser_manifest.json#postprocess` once. Preserve quotes and literal Roman `I`; do not add the former generic French cleanup.

Inputs are `pixel_values: float32[batch,3,48,width]`; output is `logits`. The production browser loads `onnx/ppocrv6_bubble_line_rec_webgpu.onnx`. Use the character list and exact model filenames from `onnx/browser_manifest.json`.

## Training and export

Starting revision: `10b932d4aadca2830850ccf5951116597404bef8` of this repository. Historical page splits are preserved. The dataset contains 1,326 pages and 11,670 bubbles, with 9,256 train, 1,193 validation and 1,221 test bubbles. Twenty-eight visually reviewed **train-only** pairs were excluded because the line detector had omitted visible text, leaving **9,228** recognition training pairs. No validation/test pairs were excluded. SFX and all 24 no-line fallbacks remain included; there is no SFX classifier or automatic SFX skip.

Five full-backbone epochs in FP32, seed 42, batch 1 with accumulation 16, learning rates 6e-6 (head) / 6e-7 (backbone), cosine schedule, warmup 0.05, frozen batch-normalization running statistics. Epoch **3** was selected by raw validation CER; initial checkpoint validation and non-finite gradient checks prevent silent regressions. Zero optimizer steps were skipped. Training took approximately 927 seconds.

Native and both ONNX variants decoded identically on 9 real validation examples; maximum absolute numeric difference was 0.0001194. Dynamic widths 960, 1200, 1600, 2400 and 3200 were checked. The WebGPU export changes ceil-mode only for stride-one MaxPool, where it is equivalent.

Chrome WebGPU on AMD RDNA3 also ran 100 fixed validation crop inputs with actual Canvas normalization: 29 -> **22** character errors; native/browser raw text agreed on **99/100** examples. The active page detector passed 20 full-page WASM checks. These browser subsets do not establish full-corpus browser CER or exact parity for every input.

## Files and remaining limits

`model.safetensors`, `config.json` and `preprocessor_config.json` are the selected native checkpoint. `onnx/` contains both recognizers, the inherited line detector, dictionaries and runtime manifests. `metrics/retraining_20261103.json` contains the complete aggregate protocol, threshold curves, comparisons and evidence hashes. `training_source/` preserves training/export source layout; private training images and prediction rows are not distributed.

The leading identified data issue is **line omission before recognition**. A recognizer cannot restore pixels absent from its input. Reliable progress requires complete line annotations, a separately labeled ordinary/SFX evaluation set, and unseen volumes. The tiny inspected SFX diagnostic subset does not support a general SFX score. Text CER also does not score reading order; the aggregate report provides separate exploratory matched-order counts.

The previous June release and its CSV metrics remain accessible in repository history. Current summaries describe this November run and are not directly comparable with those older exports.
