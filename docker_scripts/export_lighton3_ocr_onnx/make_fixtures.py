"""Private, fixed validation pages and native BF16 reference outputs for browser QA."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids


def main(source, dataset, output, count, task="page-ocr-bbox"):
    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(source, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).cuda().eval()
    rows = [json.loads(line) for line in (dataset / "val/metadata.jsonl").read_text().splitlines()]
    rows = sorted(rows, key=lambda row: len(row["text"]))
    selected = [{**rows[int(i)], "split": "val"} for i in np.linspace(0, len(rows) - 1, count)]
    if task == "page-ocr-bbox":
        if not any(row["id"] == "1491" for row in selected):
            selected.append({**next(row for row in rows if row["id"] == "1491"), "split": "val"})
        test = [json.loads(line) for line in (dataset / "test/metadata.jsonl").read_text().splitlines()]
        selected.append({**next(row for row in test if row["id"] == "1233"), "split": "test", "control": "previous-repetition"})
    else:
        for predicate, control in [(lambda r: '"' in r["text"], "visible-quotes"),
                                   (lambda r: "é" in r["text"], "accents"),
                                   (lambda r: r["text"].isupper() and len(r["text"]) > 3, "uppercase")]:
            row = next(r for r in rows if predicate(r))
            selected.append({**row, "split": "val", "control": control})
    selected.append({**selected[0], "control": "state-reset-repeat"})
    out = output / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    examples = []
    content = [{"type": "image"}]
    if task == "page-ocr-bbox":
        content.append({"type": "text", "text": "grounding"})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for index, row in enumerate(selected):
        image = Image.open(dataset / row["split"] / row["image_file"]).convert("RGB")
        image.save(out / f"page-{index}.png")
        page = image.copy()
        if task == "page-ocr-bbox":
            page.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        inputs = processor(text=[prompt], images=[page], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=256 if task == "bubble-ocr" else 1280, do_sample=False, use_cache=True)
            visual = model.model.visual(inputs.pixel_values, inputs.image_grid_thw).pooler_output
            positions, _ = model.model.get_rope_index(inputs.input_ids, inputs.mm_token_type_ids, inputs.image_grid_thw)
        tokens = generated[0, inputs.input_ids.shape[1]:].tolist()
        prediction = processor.decode(tokens, skip_special_tokens=True).strip()
        grid = inputs.image_grid_thw
        indices, weights = get_vision_bilinear_indices_and_weights(grid, 48, 2)
        with torch.inference_mode():
            vpos = get_vision_position_ids(grid, 2)
            freq = model.model.visual.rotary_pos_emb(vpos).reshape(-1, 32)
            angles = torch.cat([freq, freq], -1)
        patches = len(inputs.pixel_values)
        arrays = {"pixel_values": inputs.pixel_values.cpu().numpy().astype(np.float16),
                  "position_indices": indices.cpu().numpy().astype(np.int64),
                  "position_weights": weights[:, :, None].cpu().numpy(),
                  "vision_cos": angles.cos()[None].cpu().numpy(), "vision_sin": angles.sin()[None].cpu().numpy(),
                  "query_splits": np.array([patches // 16 + int(i < patches % 16) for i in range(16)], np.int64),
                  "input_ids": inputs.input_ids.cpu().numpy(), "positions": positions.cpu().numpy(),
                  "native_features": visual.float().cpu().numpy()}
        splits = arrays.pop("query_splits").reshape(4, 4)
        arrays["query_group_splits"] = splits.sum(axis=1)
        for group in range(4):
            arrays[f"query_splits_{group}"] = splits[group]
        np.savez(out / f"inputs-{index}.npz", **arrays)
        (out / f"inputs-{index}.json").write_text(json.dumps({
            key: value.tolist() for key, value in arrays.items() if key not in {"pixel_values", "native_features"}}))
        item = {"id": str(row["id"]), "split": row["split"], "control": row.get("control"),
                "image": f"fixtures/page-{index}.png", "reference": row["text"], "native": prediction,
                "task": task,
                "native_ids": tokens, "width": image.width, "height": image.height,
                "grid": grid.cpu().tolist()[0], "fixture": f"fixtures/inputs-{index}.json"}
        examples.append(item)
        (out / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"index": index, "id": item["id"], "tokens": len(tokens), "grid": item["grid"], "control": item["control"]}), flush=True)
    print(f"Saved {len(examples)} private fixtures; never upload fixtures/", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--task", choices=["page-ocr-bbox", "bubble-ocr"], default="page-ocr-bbox")
    args = parser.parse_args()
    main(args.source, args.dataset, args.output, args.count, args.task)
