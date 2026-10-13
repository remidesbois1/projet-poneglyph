"""Separate BF16 numerical drift from export errors using native FP32 vision."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import Qwen3_5ForConditionalGeneration


def main(source, directory):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = Qwen3_5ForConditionalGeneration.from_pretrained(source, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).cuda().eval()
    visual = model.model.visual.float()
    examples = json.loads((directory / "fixtures/examples.json").read_text(encoding="utf-8"))
    for index, example in enumerate(examples):
        path = directory / f"fixtures/inputs-{index}.npz"
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        grid = torch.tensor([example["grid"]], device="cuda")
        pixels = torch.from_numpy(arrays["pixel_values"].astype(np.float32)).cuda()
        with torch.inference_mode():
            reference = visual(pixels, grid).pooler_output.float().cpu().numpy()
        arrays["native_features_fp32"] = reference
        np.savez(path, **arrays)
        error = arrays["native_features"] - reference
        print(json.dumps({"id": example["id"], "native_bf16_vs_fp32_relative_rmse": float(np.linalg.norm(error) / np.linalg.norm(reference)),
                          "reference": "Native Transformers FP32 vision, TF32 disabled, same exported FP16 pixel input"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.model)
