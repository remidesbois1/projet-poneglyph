# Benchmarks modèles publiés

Ce document est généré depuis `shared/model-registry.json` (v1, 2026-09-09). Ne pas modifier manuellement.

Les résultats ne sont comparables qu’à protocole et tâche identiques. Un matériel non consigné est indiqué explicitement plutôt que supposé.

## PP-OCRv6 Bubble Line

- Identifiant registre : `ppocrv6-line`
- Tâche : `bubble-line-ocr`
- Version : [`Remidesbois/pp-ocrv6-one-piece-bubble-line-rec@10b932d4aadca2830850ccf5951116597404bef8`](https://huggingface.co/Remidesbois/pp-ocrv6-one-piece-bubble-line-rec/tree/10b932d4aadca2830850ccf5951116597404bef8)
- Dataset : Poneglyph validated bubbles reconstructed from detected text lines
- Split : test held-out by page
- Date : 2026-06-29
- Échantillons : 1 219
- Matériel : Not recorded; offline scoring over pinned predictions
- Protocole : YOLO26n line detection, horizontal line stitching, PP-OCRv6 CTC decoding, then spacing and case rules learned only from the training split.
- Preuve : [artefact figé](https://huggingface.co/Remidesbois/pp-ocrv6-one-piece-bubble-line-rec/blob/10b932d4aadca2830850ccf5951116597404bef8/postprocess_official_metrics.json)

| Clé | Valeur | Sens favorable |
|---|---:|---|
| `cer` | CER 1,451 % | plus bas |
| `exact_match` | Exact match 75,96 % | plus haut |

## LightOnOCR Poneglyph

- Identifiant registre : `lighton-bubble`
- Tâche : `bubble-ocr`
- Version : [`Remidesbois/LightonOCR-2-1b-poneglyph@3d5181ce138e7d92132a741f1e54c3a9e602e129`](https://huggingface.co/Remidesbois/LightonOCR-2-1b-poneglyph/tree/3d5181ce138e7d92132a741f1e54c3a9e602e129)
- Dataset : Poneglyph validated single-bubble crops
- Split : test held-out by page
- Date : 2026-07-01
- Échantillons : 1 128
- Matériel : Modal NVIDIA H100
- Protocole : Strict full-generation transcription with the published prompt; whitespace-normalized CER and WER, exact match, blank-rate and multiline-rate over every held-out sample.
- Preuve : [artefact figé](https://huggingface.co/Remidesbois/LightonOCR-2-1b-poneglyph/blob/3d5181ce138e7d92132a741f1e54c3a9e602e129/benchmark_test.json)

| Clé | Valeur | Sens favorable |
|---|---:|---|
| `cer` | CER 0,424 % | plus bas |
| `wer` | WER 1,405 % | plus bas |
| `exact_match` | Exact match 92,55 % | plus haut |

## Surya OCR 2 Poneglyph

- Identifiant registre : `surya-bubble`
- Tâche : `bubble-ocr`
- Version : [`Remidesbois/surya-bubble-ocr-poneglyph@7d7b358c545cfe757329f780da6ed4100bb5909f`](https://huggingface.co/Remidesbois/surya-bubble-ocr-poneglyph/tree/7d7b358c545cfe757329f780da6ed4100bb5909f)
- Dataset : Poneglyph validated single-bubble crops
- Split : test held-out by page
- Date : 2026-07-30
- Échantillons : 1 423
- Matériel : NVIDIA RTX 3090 24 GB
- Protocole : Exhaustive generative evaluation with a 256-token budget, collapsed whitespace and explicit blank, hallucination and token-limit accounting.
- Preuve : [artefact figé](https://huggingface.co/Remidesbois/surya-bubble-ocr-poneglyph/blob/7d7b358c545cfe757329f780da6ed4100bb5909f/benchmark_test.json)

| Clé | Valeur | Sens favorable |
|---|---:|---|
| `cer` | CER 0,451 % | plus bas |
| `wer` | WER 1,656 % | plus bas |
| `exact_match` | Exact match 90,65 % | plus haut |
| `token_limit_rate` | Token limit 0,00 % | plus bas |

## Surya OCR 2 Poneglyph BBox

- Identifiant registre : `surya-bbox`
- Tâche : `page-ocr-bbox`
- Version : [`Remidesbois/surya-ocr-2-poneglyph-bbox@671cfe63672286ccfe629079a8d57f7ed967f3d5`](https://huggingface.co/Remidesbois/surya-ocr-2-poneglyph-bbox/tree/671cfe63672286ccfe629079a8d57f7ed967f3d5)
- Dataset : Poneglyph validated full pages with text-zone bounding boxes
- Split : test held-out by page
- Date : 2026-09-09
- Échantillons : 166
- Matériel : NVIDIA GeForce RTX 5090 32 GB
- Protocole : Full-page multimodal generation of one exact text zone per line as Texte exact [x1,y1,x2,y2], with normalized integer coordinates in [0,1000]; OCR text and bbox localization scored jointly on the held-out page split.
- Preuve : [artefact figé](https://huggingface.co/Remidesbois/surya-ocr-2-poneglyph-bbox/blob/671cfe63672286ccfe629079a8d57f7ed967f3d5/benchmark_surya_bbox.json)

| Clé | Valeur | Sens favorable |
|---|---:|---|
| `cer` | CER 1,49 % | plus bas |
| `wer` | WER 3,44 % | plus bas |
| `mean_iou` | Mean IoU 92,36 % | plus haut |
| `detection_rate` | Detection rate 98,38 % | plus haut |
| `f1_iou_0_5` | F1@IoU 0.5 98,36 % | plus haut |
| `f1_iou_0_75` | F1@IoU 0.75 96,67 % | plus haut |
| `combined_score` | Combined score 97,220 % | plus haut |
| `avg_inference_time` | Avg inference 2,75 | plus bas |

## LightOnOCR 2 Poneglyph BBox

- Identifiant registre : `lighton-bbox`
- Tâche : `page-ocr-bbox`
- Version : [`Remidesbois/LightonOCR-2-1b-poneglyph-bbox@fb35b7f4f99b282d5c2867d5cf23e66df0d03925`](https://huggingface.co/Remidesbois/LightonOCR-2-1b-poneglyph-bbox/tree/fb35b7f4f99b282d5c2867d5cf23e66df0d03925)
- Dataset : Poneglyph validated full pages with text-zone bounding boxes
- Split : test held-out by page
- Date : 2026-09-09
- Échantillons : 221
- Matériel : NVIDIA GeForce RTX 5090 32 GB
- Protocole : Full-page image-only LightOnOCR generation of one exact text zone per line as Texte exact [x1,y1,x2,y2], with normalized integer coordinates in [0,1000]. Publication metrics use full-page CER, one-to-one global IoU assignment including unmatched GT zones at IoU 0, and micro-aggregated detection metrics over the complete held-out test split.
- Preuve : [artefact figé](https://huggingface.co/Remidesbois/LightonOCR-2-1b-poneglyph-bbox/blob/fb35b7f4f99b282d5c2867d5cf23e66df0d03925/benchmark_lighton_bbox_corrected.json)

| Clé | Valeur | Sens favorable |
|---|---:|---|
| `page_cer` | Page CER 5,61 % | plus bas |
| `mean_iou` | Global mean IoU 65,00 % | plus haut |
| `f1_iou_0_3` | F1@IoU 0.3 91,29 % | plus haut |
| `f1_iou_0_5` | F1@IoU 0.5 79,62 % | plus haut |
| `f1_iou_0_75` | F1@IoU 0.75 41,75 % | plus haut |
| `exact_bubble_text` | Exact bubble text 90,63 % | plus haut |
| `bubble_text_cer` | Bubble-text CER 1,15 % | plus bas |
| `combined_score` | Corrected combined score 82,571 % | plus haut |
| `avg_inference_time` | Avg inference 6,18 | plus bas |

## ReaderNet Panel Detector

- Identifiant registre : `readernet-panel-detector`
- Tâche : `panel-detection`
- Version : [`Remidesbois/Poneglyph-ReaderNet@d97d4cd2903a7ebe49276a5269c4f3b7df608be7`](https://huggingface.co/Remidesbois/Poneglyph-ReaderNet/tree/d97d4cd2903a7ebe49276a5269c4f3b7df608be7)
- Dataset : Poneglyph weighted polygon panel dataset
- Split : validation held out by page
- Date : 2026-08-21
- Échantillons : 257
- Matériel : NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb
- Protocole : Ultralytics YOLO11n-seg validation at 1504 px with box and mask metrics over polygon panel annotations.
- Preuve : [artefact figé](https://huggingface.co/Remidesbois/Poneglyph-ReaderNet/blob/d97d4cd2903a7ebe49276a5269c4f3b7df608be7/metrics/panel_detector_metrics.json)

| Clé | Valeur | Sens favorable |
|---|---:|---|
| `map50` | mAP50 99,44 % | plus haut |
| `map50_95` | mAP50-95 98,39 % | plus haut |

## Poneglyph ReaderNet

- Identifiant registre : `readernet-reading-order`
- Tâche : `reading-order`
- Version : [`Remidesbois/Poneglyph-ReaderNet@d97d4cd2903a7ebe49276a5269c4f3b7df608be7`](https://huggingface.co/Remidesbois/Poneglyph-ReaderNet/tree/d97d4cd2903a7ebe49276a5269c4f3b7df608be7)
- Dataset : Poneglyph polygon panels with bubbles from the shared YOLO26n detector
- Split : validation held out by page
- Date : 2026-08-27
- Échantillons : 138
- Matériel : CPU offline ONNX scoring
- Protocole : Shared bubble detections and annotated panels; pairwise Borda ranking independently inside each panel, with no global bubble sorter.
- Preuve : [artefact figé](https://huggingface.co/Remidesbois/Poneglyph-ReaderNet/blob/d97d4cd2903a7ebe49276a5269c4f3b7df608be7/metrics/shared_detector_comparison.json)

| Clé | Valeur | Sens favorable |
|---|---:|---|
| `exact_panel_match` | Exact panel 95,65 % | plus haut |
| `bubble_position_accuracy` | Bubble position accuracy 96,62 % | plus haut |
| `assignment_accuracy` | Bubble assignment accuracy 100,00 % | plus haut |
