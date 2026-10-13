"""Load the exported native model and OCR one complete page with grounding."""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from grounding import PROMPT, parse_grounding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    args = parser.parse_args()
    if not args.model.is_dir() or not args.image.is_file():
        raise FileNotFoundError("Both the native model directory and page image must exist")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This inference profile requires CUDA/BF16")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    with Image.open(args.image) as source:
        original = source.convert("RGB")
    page = original.copy()
    page.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = processor(text=[prompt], images=[page], return_tensors="pt")
    page.close()
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=args.max_new_tokens)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    tokens = generated[0, inputs["input_ids"].shape[1]:].cpu().tolist()
    prediction = processor.decode(tokens, skip_special_tokens=True).strip()
    items, invalid = parse_grounding(prediction)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    for item in items:
        item["bbox_pixels"] = [round(value * (original.width if i % 2 == 0 else original.height) / 1000) for i, value in enumerate(item["bbox"])]
    result = {"model": str(args.model), "image": str(args.image), "image_size": list(original.size),
              "embedding_head_tied": model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr(),
              "prompt": PROMPT, "thinking": False, "prediction": prediction, "items": items,
              "invalid_blocks": invalid, "generated_tokens": len(tokens),
              "truncated": not any(token in eos_ids for token in tokens) and len(tokens) >= args.max_new_tokens,
              "generation_seconds": seconds, "tokens_per_second": len(tokens) / seconds}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.overlay:
        draw = ImageDraw.Draw(original)
        for i, item in enumerate(items):
            box = item["bbox_pixels"]
            draw.rectangle(box, outline="#ff503e", width=3)
            draw.text((box[0] + 2, box[1] + 2), str(i + 1), fill="#ff503e", stroke_width=1, stroke_fill="black")
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        original.save(args.overlay)
    original.close()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
