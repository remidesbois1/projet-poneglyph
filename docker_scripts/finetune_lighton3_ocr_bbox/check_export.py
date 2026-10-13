"""Compare native FP32/AMP and BF16 exports on validation, never select on test."""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

# This checker can run from the bind mount while shared callbacks live in /app.
sys.path.insert(0, "/app")
from train import Collator, atomic_json, benchmark, load_datasets, subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/dataset"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    summary = json.loads((args.run / "training_summary.json").read_text(encoding="utf-8"))
    checkpoint = Path(summary["best_model_checkpoint"])
    processor = AutoProcessor.from_pretrained(args.run / "final_merged", local_files_only=True)
    processor.tokenizer.padding_side = "left"
    datasets, manifest = load_datasets(args.dataset, processor, 1280)
    validation = subset(datasets["val"], 48)
    collator = Collator(processor, args.dataset, augment=False, cache_items=16)
    scores = {}
    for name, path, dtype in (("bf16", args.run / "final_merged", torch.bfloat16), ("fp32_autocast", checkpoint, torch.float32)):
        print(f"Native export validation: {name}", flush=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(path, local_files_only=True, dtype=dtype, attn_implementation="sdpa").cuda().eval()
        model.generation_config.do_sample = False
        model.generation_config.temperature = model.generation_config.top_p = model.generation_config.top_k = None
        scores[name] = benchmark(model, processor, validation, collator, args.run / f"export_validation_{name}.json", 4, 1280)
        if name == "fp32_autocast":
            model.config.use_cache = model.config.text_config.use_cache = True
            export = args.run / "final_fp32"
            model.save_pretrained(export, safe_serialization=True, max_shard_size="4GB")
            processor.save_pretrained(export)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(args.run / "export_precision_comparison.json", {"checkpoint": str(checkpoint), "dataset": manifest,
                "selection_split": "validation", "generation_pages": 48, "scores": scores})
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
