"""Generate private local validation fixtures; never publish images or references."""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids


def clean(text):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    return text


def main(source, dataset, output, count, stress=False, split="val", arrays=True):
    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(source)
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    rows = [json.loads(line) for line in (dataset / split / "metadata.jsonl").read_text().splitlines()]
    rows = sorted(rows, key=lambda r: len(r["text"]))
    # Fixed length-stratified selection, no searching for examples that happen to pass.
    selected = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, count)]
    if stress:
        selected += [dict(selected[-1], id="stress-large", scale=6), dict(selected[0], id="stress-repeat")]
    out = output / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    examples = []
    for index, row in enumerate(selected):
        image_path = dataset / split / row["image_file"]
        image = Image.open(image_path).convert("RGB")
        if row.get("scale"):
            image = image.resize((image.width * row["scale"], image.height * row["scale"]), Image.Resampling.BICUBIC)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcris exactement le texte visible dans cette bulle. Ne rajoute rien."}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        native_ids = generated[0, inputs.input_ids.shape[1]:].tolist()
        raw = processor.tokenizer.decode(native_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        grid = inputs.image_grid_thw
        image.save(out / f"bubble-{index}.png")
        item = {"id": str(row["id"]), "image": f"fixtures/bubble-{index}.png", "reference": row["text"],
                "native": clean(raw), "raw": raw, "native_ids": native_ids,
                "width": image.width, "height": image.height, "grid": grid.tolist()[0]}
        examples.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        if not arrays:
            continue
        with torch.inference_mode():
            visual = model.model.visual(inputs.pixel_values, inputs.image_grid_thw).pooler_output
            positions, _ = model.model.get_rope_index(inputs.input_ids, inputs.mm_token_type_ids, inputs.image_grid_thw)
        indices, weights = get_vision_bilinear_indices_and_weights(grid, 48, 2)
        with torch.inference_mode():
            vpos = get_vision_position_ids(grid, 2)
            freq = model.model.visual.rotary_pos_emb(vpos).reshape(-1, 32)
            angles = torch.cat([freq, freq], -1)
        arrays = {"pixel_values": inputs.pixel_values.cpu().numpy().astype(np.float16),
                  "position_indices": indices.cpu().numpy().astype(np.int64),
                  "position_weights": weights[:, :, None].cpu().numpy(),
                  "vision_cos": angles.cos()[None].cpu().numpy(), "vision_sin": angles.sin()[None].cpu().numpy(),
                  "input_ids": inputs.input_ids.cpu().numpy(), "positions": positions.cpu().numpy(),
                  "image_grid_thw": grid.cpu().numpy(), "native_features": visual.float().cpu().numpy()}
        np.savez(out / f"inputs-{index}.npz", **arrays)
        # Small arrays also exercise JS preprocessing independently of raster interpolation.
        (out / f"inputs-{index}.json").write_text(json.dumps({
            k: v.tolist() for k, v in arrays.items() if k not in ["pixel_values", "native_features"]}))
    (out / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--stress", action="store_true", help="Append a near-max-pixel crop then repeat the first crop")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--no-arrays", action="store_true", help="Only save images and independent native generations")
    args = parser.parse_args()
    main(args.source, args.dataset, args.output, args.count, args.stress, args.split, not args.no_arrays)
