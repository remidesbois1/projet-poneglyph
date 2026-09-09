---
license: apache-2.0
language:
- fr
library_name: transformers
pipeline_tag: image-text-to-text
base_model: lightonai/LightOnOCR-2-1B-bbox-base
tags:
- lighton_ocr
- vision-language-model
- ocr
- manga
- one-piece
- bbox-detection
- fine-tuned
---

# LightOnOCR-2-1B Poneglyph BBox

Fine-tuned version of [`lightonai/LightOnOCR-2-1B-bbox-base`](https://huggingface.co/lightonai/LightOnOCR-2-1B-bbox-base) for **full-page French manga OCR with text-zone bounding boxes**.

The model receives **only an image** and generates one text zone per line:

```text
Texte exact [x1,y1,x2,y2]
```

Coordinates are integers normalized to **`[0,1000]`** in the image reference frame. The model is trained to preserve Japanese manga reading order and to return no JSON, Markdown, prefix or commentary.

## Final held-out benchmark

Evaluation was run on the complete held-out **test split: 221 pages, 1,920 annotated text zones**. The model generated 1,903 zones (99.11% of the GT count).

The table below reports the corrected global metrics. These are recomputed from every raw page prediction in `benchmark_lighton_bbox.json`; the full values and definitions are stored in [`benchmark_lighton_bbox_corrected.json`](benchmark_lighton_bbox_corrected.json).

| Metric | Final score |
|---|---:|
| **Corrected combined score** | **0.8257** |
| Page CER, macro | **5.61%** |
| Page CER, character-weighted | **4.48%** |
| Global mean IoU | **65.00%** |
| F1 @ IoU 0.3, micro | **91.29%** |
| **F1 @ IoU 0.5, micro** | **79.62%** |
| Precision @ IoU 0.5, micro | **79.98%** |
| Recall @ IoU 0.5, micro | **79.27%** |
| F1 @ IoU 0.75, micro | **41.75%** |
| F1 @ IoU 0.9, micro | **4.71%** |
| Exact bubble text | **90.63%** |
| Bubble-text CER, character-weighted | **1.15%** |
| Average generation time | **6.18 s/page** |

### Metric definitions

- **Page CER (macro)**: CER computed on the complete serialized page output, then averaged over pages. Reading-order mistakes therefore count as OCR sequence errors.
- **Page CER (character-weighted)**: the same full-page comparison, aggregated by reference character count.
- **Global mean IoU**: one-to-one maximum-IoU assignment over all predictions and GT zones; unmatched GT zones contribute IoU `0`. This avoids inflating IoU by averaging only detections that already pass a threshold.
- **Precision / recall / F1**: micro-aggregated across the complete test split at the stated IoU threshold.
- **Exact bubble text**: exact text-content matches after one-to-one text assignment, independent of bbox quality and serialized page order.
- **Bubble-text CER**: OCR error after bubble-text assignment, weighted by reference character count.

The corrected combined score keeps the training benchmark weighting but uses the corrected global metrics:

```text
0.4 * (1 - Page_CER_macro)
+ 0.3 * F1@0.5_micro
+ 0.2 * Global_Mean_IoU
+ 0.1 * Recall@0.5_micro
= 0.8257147803
```

The original `metrics_version=2` fields remain available in `benchmark_lighton_bbox.json` for reproducibility. In particular, its legacy `mean_iou` averages only IoU>=0.5 matches and should not be interpreted as a global mean IoU.

## Dataset

Human-validated Poneglyph full manga pages were split by page:

| Split | Pages | Annotated zones |
|---|---:|---:|
| Train | 775 | 6,962 |
| Validation | 111 | 1,008 |
| Test | 221 | 1,920 |
| **Total** | **1,107** | **9,890** |

Images are processed with a **1,500 px longest edge** while keeping their aspect ratio. Bboxes are normalized to `[0,1000]`.

## Training

The final run was trained locally on an **NVIDIA GeForce RTX 5090 32 GB**.

| Setting | Value |
|---|---|
| Base model | `lightonai/LightOnOCR-2-1B-bbox-base` |
| Epochs | 3 |
| Optimizer steps | 291 |
| Learning rate | `1e-5` |
| Scheduler | cosine |
| Warmup | 15 steps |
| Optimizer | fused AdamW |
| Weight decay | `0.01` |
| Precision | BF16 + TF32 |
| Physical batch | 1 |
| Gradient accumulation | 8 |
| Effective batch | 8 |
| Gradient checkpointing | disabled |
| Adapter | rsLoRA, `r=128`, `alpha=256`, dropout `0` |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` in vision + language |
| Fully trained bridge | `vision_projection` |
| Trainable parameters | 159,384,576 (~13.68%) |
| Peak VRAM during calibration | 26.68 GiB |

The LoRA adapters were merged into the final weights published in this repository.

## Inference

```python
import torch
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

MODEL_ID = "Remidesbois/LightonOCR-2-1b-poneglyph-bbox"
IMAGE_PATH = "page.jpg"

processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)
processor.image_processor.default_to_square = False

model = LightOnOcrForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")

# Image-only conditioning: do not add a textual user prompt.
messages = [
    {
        "role": "user",
        "content": [{"type": "image"}],
    }
]

prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)

inputs = processor(
    text=[prompt],
    images=[image],
    size={"longest_edge": 1500},
    return_tensors="pt",
)

device = next(model.parameters()).device
inputs = {
    key: value.to(device=device, dtype=torch.bfloat16)
    if value.is_floating_point()
    else value.to(device)
    for key, value in inputs.items()
}

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1280,
        do_sample=False,
        use_cache=True,
    )

generated = output_ids[:, inputs["input_ids"].shape[1]:]
text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
print(text)
```

Example output:

```text
Je serai le roi des pirates !! [684,102,919,191]
Allons-y. [112,713,251,769]
```

To map a normalized bbox back to pixels for an image of width `W` and height `H`:

```python
x1_px = x1 * W / 1000
y1_px = y1 * H / 1000
x2_px = x2 * W / 1000
y2_px = y2 * H / 1000
```

## Benchmark artifacts

- `benchmark_lighton_bbox.json`: raw final benchmark with all 221 page predictions and the legacy metrics emitted by the training pipeline.
- `benchmark_lighton_bbox_corrected.json`: corrected global/micro metrics derived from those same predictions, including metric definitions and the combined-score formula.

## Limitations

- Fine-tuned primarily on French **One Piece** manga pages; generalization to other manga, languages or layouts is not guaranteed.
- Very strict bbox localization remains the main limitation: F1 is 79.62% at IoU 0.5 but 41.75% at IoU 0.75.
- Full-page CER is sensitive to reading-order mistakes even when individual bubble transcription is correct.
- The model may occasionally miss, duplicate or reorder small text zones.

## Base model

Fine-tuned from **LightOnOCR-2-1B-bbox** by LightOn.

```bibtex
@misc{lightonocr2_2026,
  title        = {LightOnOCR: A 1B End-to-End Multilingual Vision-Language Model for State-of-the-Art OCR},
  author       = {Said Taghadouini and Adrien Cavailles and Baptiste Aubertin},
  year         = {2026},
  howpublished = {https://arxiv.org/abs/2601.14251}
}
```

## License

Apache 2.0, following the base model license.
