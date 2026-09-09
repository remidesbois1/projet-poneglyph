# Model benchmark release notes

Generated from model registry v1 on 2026-09-09.

## PP-OCRv6 Bubble Line

- Registry ID: `ppocrv6-line`
- Published version: `Remidesbois/pp-ocrv6-one-piece-bubble-line-rec@10b932d4aadca2830850ccf5951116597404bef8`
- Result: CER 1,451 % · Exact match 75,96 %
- Evaluation: Poneglyph validated bubbles reconstructed from detected text lines, test held-out by page, 1219 samples, 2026-06-29
- Hardware: Not recorded; offline scoring over pinned predictions
- Protocol and evidence: YOLO26n line detection, horizontal line stitching, PP-OCRv6 CTC decoding, then spacing and case rules learned only from the training split. https://huggingface.co/Remidesbois/pp-ocrv6-one-piece-bubble-line-rec/blob/10b932d4aadca2830850ccf5951116597404bef8/postprocess_official_metrics.json

## LightOnOCR Poneglyph

- Registry ID: `lighton-bubble`
- Published version: `Remidesbois/LightonOCR-2-1b-poneglyph@3d5181ce138e7d92132a741f1e54c3a9e602e129`
- Result: CER 0,424 % · WER 1,405 % · Exact match 92,55 %
- Evaluation: Poneglyph validated single-bubble crops, test held-out by page, 1128 samples, 2026-07-01
- Hardware: Modal NVIDIA H100
- Protocol and evidence: Strict full-generation transcription with the published prompt; whitespace-normalized CER and WER, exact match, blank-rate and multiline-rate over every held-out sample. https://huggingface.co/Remidesbois/LightonOCR-2-1b-poneglyph/blob/3d5181ce138e7d92132a741f1e54c3a9e602e129/benchmark_test.json

## Surya OCR 2 Poneglyph

- Registry ID: `surya-bubble`
- Published version: `Remidesbois/surya-bubble-ocr-poneglyph@7d7b358c545cfe757329f780da6ed4100bb5909f`
- Result: CER 0,451 % · WER 1,656 % · Exact match 90,65 % · Token limit 0,00 %
- Evaluation: Poneglyph validated single-bubble crops, test held-out by page, 1423 samples, 2026-07-30
- Hardware: NVIDIA RTX 3090 24 GB
- Protocol and evidence: Exhaustive generative evaluation with a 256-token budget, collapsed whitespace and explicit blank, hallucination and token-limit accounting. https://huggingface.co/Remidesbois/surya-bubble-ocr-poneglyph/blob/7d7b358c545cfe757329f780da6ed4100bb5909f/benchmark_test.json

## Surya OCR 2 Poneglyph BBox

- Registry ID: `surya-bbox`
- Published version: `Remidesbois/surya-ocr-2-poneglyph-bbox@671cfe63672286ccfe629079a8d57f7ed967f3d5`
- Result: CER 1,49 % · WER 3,44 % · Mean IoU 92,36 % · Detection rate 98,38 % · F1@IoU 0.5 98,36 % · F1@IoU 0.75 96,67 % · Combined score 97,220 % · Avg inference 2,75
- Evaluation: Poneglyph validated full pages with text-zone bounding boxes, test held-out by page, 166 samples, 2026-09-09
- Hardware: NVIDIA GeForce RTX 5090 32 GB
- Protocol and evidence: Full-page multimodal generation of one exact text zone per line as Texte exact [x1,y1,x2,y2], with normalized integer coordinates in [0,1000]; OCR text and bbox localization scored jointly on the held-out page split. https://huggingface.co/Remidesbois/surya-ocr-2-poneglyph-bbox/blob/671cfe63672286ccfe629079a8d57f7ed967f3d5/benchmark_surya_bbox.json

## LightOnOCR 2 Poneglyph BBox

- Registry ID: `lighton-bbox`
- Published version: `Remidesbois/LightonOCR-2-1b-poneglyph-bbox@fb35b7f4f99b282d5c2867d5cf23e66df0d03925`
- Result: Page CER 5,61 % · Global mean IoU 65,00 % · F1@IoU 0.3 91,29 % · F1@IoU 0.5 79,62 % · F1@IoU 0.75 41,75 % · Exact bubble text 90,63 % · Bubble-text CER 1,15 % · Corrected combined score 82,571 % · Avg inference 6,18
- Evaluation: Poneglyph validated full pages with text-zone bounding boxes, test held-out by page, 221 samples, 2026-09-09
- Hardware: NVIDIA GeForce RTX 5090 32 GB
- Protocol and evidence: Full-page image-only LightOnOCR generation of one exact text zone per line as Texte exact [x1,y1,x2,y2], with normalized integer coordinates in [0,1000]. Publication metrics use full-page CER, one-to-one global IoU assignment including unmatched GT zones at IoU 0, and micro-aggregated detection metrics over the complete held-out test split. https://huggingface.co/Remidesbois/LightonOCR-2-1b-poneglyph-bbox/blob/fb35b7f4f99b282d5c2867d5cf23e66df0d03925/benchmark_lighton_bbox_corrected.json

## ReaderNet Panel Detector

- Registry ID: `readernet-panel-detector`
- Published version: `Remidesbois/Poneglyph-ReaderNet@d97d4cd2903a7ebe49276a5269c4f3b7df608be7`
- Result: mAP50 99,44 % · mAP50-95 98,39 %
- Evaluation: Poneglyph weighted polygon panel dataset, validation held out by page, 257 samples, 2026-08-21
- Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb
- Protocol and evidence: Ultralytics YOLO11n-seg validation at 1504 px with box and mask metrics over polygon panel annotations. https://huggingface.co/Remidesbois/Poneglyph-ReaderNet/blob/d97d4cd2903a7ebe49276a5269c4f3b7df608be7/metrics/panel_detector_metrics.json

## Poneglyph ReaderNet

- Registry ID: `readernet-reading-order`
- Published version: `Remidesbois/Poneglyph-ReaderNet@d97d4cd2903a7ebe49276a5269c4f3b7df608be7`
- Result: Exact panel 95,65 % · Bubble position accuracy 96,62 % · Bubble assignment accuracy 100,00 %
- Evaluation: Poneglyph polygon panels with bubbles from the shared YOLO26n detector, validation held out by page, 138 samples, 2026-08-27
- Hardware: CPU offline ONNX scoring
- Protocol and evidence: Shared bubble detections and annotated panels; pairwise Borda ranking independently inside each panel, with no global bubble sorter. https://huggingface.co/Remidesbois/Poneglyph-ReaderNet/blob/d97d4cd2903a7ebe49276a5269c4f3b7df608be7/metrics/shared_detector_comparison.json
