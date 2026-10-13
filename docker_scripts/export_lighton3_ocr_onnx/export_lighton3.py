"""Export LightOnOCR-3 to portable ONNX plus the shared Qwen3.5 WebGPU decoder."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "export_surya_ocr_onnx"))
from export_onnx import vision, decoder
from export_decode_weights import export as export_decode_index
from export_prefill import export as export_state_prefill


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(source, output, task="page-ocr-bbox", source_revision=None):
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    text, visual = config["text_config"], config["vision_config"]
    expected = {"hidden_size": 1024, "num_attention_heads": 8, "num_key_value_heads": 2,
                "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": 16,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4, "intermediate_size": 3584, "vocab_size": 248320}
    if config["model_type"] != "qwen3_5" or any(text.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported LightOnOCR-3 architecture; do not silently export a different model")
    if text["layer_types"] != (["linear_attention"] * 3 + ["full_attention"]) * 6:
        raise ValueError("Unexpected hybrid attention layout")
    if visual["deepstack_visual_indexes"] or any(visual.get(k) != v for k, v in {
            "depth": 12, "hidden_size": 768, "num_heads": 12, "patch_size": 16,
            "spatial_merge_size": 2, "temporal_patch_size": 2, "num_position_embeddings": 2304}.items()):
        raise ValueError("Unsupported vision encoder")
    with safe_open(source / "model.safetensors", framework="pt") as weights:
        for name in weights.keys():
            value = weights.get_tensor(name).float().numpy()
            if not np.isfinite(value.astype(np.float16)).all():
                raise ValueError(f"FP16 overflow in {name}")
    # Exact global attention, query chunks only: retain every key/value and bound
    # the largest intermediate below WebGPU storage-buffer limits on 2048² pages.
    vision(source, output, attention_chunks=16)
    prefill = decoder(source, output, config)
    decoder(source, output, config, shared=prefill)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    content = [{"type": "image"}]
    if task == "page-ocr-bbox":
        content.append({"type": "text", "text": "grounding"})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    chunks = [processor.tokenizer.encode(part, add_special_tokens=False) for part in prompt.split("<|image_pad|>")]
    if len(chunks) != 2:
        raise ValueError("Expected exactly one image placeholder")
    generation = json.loads((source / "generation_config.json").read_text())
    stops = generation["eos_token_id"]
    if not isinstance(stops, list):
        stops = [stops]
    processor_config = json.loads((source / "processor_config.json").read_text())
    size = processor_config["image_processor"]["size"]
    files = ["vision.onnx", "vision.onnx.data", "decoder.onnx", "decode.onnx", "decoder.onnx.data",
             "tokenizer.json", "tokenizer_config.json"]
    for name in files[-2:]:
        shutil.copy2(source / name, output / name)
    bubble = task == "bubble-ocr"
    manifest = {"schema": 1, "model_id": "Remidesbois/LightOnOCR-3-Poneglyph" if bubble else "Remidesbois/LightOnOCR-3-Poneglyph-bbox",
                "task": task, "source_sha256": digest(source / "model.safetensors"),
                "config": config, "format": "mixed-fp16-fp32", "chunk_size": 32,
                "max_new_tokens": 256 if bubble else 1280, "vision_attention_chunks": 16,
                "prompt": "" if bubble else "grounding", "thinking": False, "prompt_chunks": chunks,
                "image_token_id": config["image_token_id"], "stop_ids": stops,
                "min_pixels": size["shortest_edge"], "max_pixels": size["longest_edge"],
                "files": {name: {"bytes": (output / name).stat().st_size, "sha256": digest(output / name)} for name in files}}
    if source_revision:
        manifest["source_revision"] = source_revision
    if bubble:
        if size != {"shortest_edge": 65536, "longest_edge": 1048576}:
            raise ValueError("Unexpected bubble processor pixel budget")
        manifest["postprocess"] = {"collapse_whitespace": True, "strip_outer_quote_pair": False}
        manifest["image_resize"] = {"backend": "torchvision-cpu-uint8", "filter": "bicubic", "antialias": True,
                                    "implementation": "adaptive-int16-coefficients-two-uint8-passes"}
    else:
        manifest.update(max_long_edge=2048, bbox={"format": "lighton-grounding", "coordinate_scale": 1000})
    (output / "browser_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    export_decode_index(source, output)
    export_state_prefill(output)
    export_state_prefill(output, fused_norms=True)
    print(json.dumps({"source_sha256": manifest["source_sha256"], "prompt_chunks": chunks,
                      "stop_ids": stops, "exported": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision-only", action="store_true")
    parser.add_argument("--task", choices=["page-ocr-bbox", "bubble-ocr"], default="page-ocr-bbox")
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    if args.vision_only:
        vision(args.source, args.output, attention_chunks=16)
        path = args.output / "browser_manifest.json"
        manifest = json.loads(path.read_text())
        for name in ["vision.onnx", "vision.onnx.data"]:
            manifest["files"][name] = {"bytes": (args.output / name).stat().st_size, "sha256": digest(args.output / name)}
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        export(args.source, args.output, args.task, args.source_revision)
