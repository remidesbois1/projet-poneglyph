"""Check portable ONNX integrity, finite recurrent states and complete page generation."""
import argparse
import json
import sys
import time
import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "export_surya_ocr_onnx"))
from validate_onnx import rotation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune_lighton3_ocr_bbox"))
from grounding import page_metrics, aggregate


def feeds_for(ids, positions, start, features, cache, image_offset, image_id):
    n = len(ids)
    cos, sin = rotation(positions)
    total = start + n
    return dict(input_ids=ids[None].astype(np.int64), image_features=features,
        image_indices=np.where(ids == image_id, np.arange(start, total) - image_offset, 0).astype(np.int64),
        image_mask=(ids == image_id)[None, :, None], rope_cos=cos, rope_sin=sin,
        attention_mask=np.where(np.arange(total)[None, :] <= np.arange(start, total)[:, None], 0, -10000).astype(np.float16)[None, None],
        identity=np.eye(n, dtype=np.float32)[None], lower=np.tri(n, dtype=np.float32)[None],
        strict_lower=np.tri(n, k=-1, dtype=np.float32)[None], **cache)


def main(directory, count, provider):
    manifest = json.loads((directory / "browser_manifest.json").read_text())
    for name in ["vision.onnx", "decoder.onnx", "decode.onnx", "prefill.onnx", "prefill-fast.onnx"]:
        print(f"Checking {name}", flush=True)
        onnx.checker.check_model(str(directory / name))
        graph = onnx.load(directory / name, load_external_data=False)
        assert all(node.domain in {"", "ai.onnx"} and node.op_type not in {"Loop", "Scan"} for node in graph.graph.node)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("session.disable_prepacking", "1")
    if provider == "cuda":
        import torch
        ort.preload_dlls()
        providers = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sessions = {}
    for name in ["vision", "decoder", "decode"]:
        print(f"Loading ORT {provider} {name}", flush=True)
        sessions[name] = ort.InferenceSession(str(directory / f"{name}.onnx"), opts, providers=providers)
        if provider == "cuda" and "CUDAExecutionProvider" not in sessions[name].get_providers():
            raise RuntimeError("CUDA EP failed to load; do not silently validate on CPU")
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    examples = json.loads((directory / "fixtures/examples.json").read_text(encoding="utf-8"))
    results = []
    for index, example in enumerate(examples[:count]):
        started = time.perf_counter()
        fixture = np.load(directory / f"fixtures/inputs-{index}.npz")
        print(f"Page {example['id']}: vision, {len(fixture['pixel_values'])} patches", flush=True)
        features = sessions["vision"].run(None, {item.name: fixture[item.name] for item in sessions["vision"].get_inputs()})[0]
        assert np.isfinite(features).all()
        native = fixture["native_features"]
        error = features.astype(np.float32) - native
        relative_rmse = float(np.linalg.norm(error) / max(np.linalg.norm(native), 1e-12))
        fp32_error = None
        native_precision_error = None
        if "native_features_fp32" in fixture.files:
            fp32 = fixture["native_features_fp32"]
            fp32_error = float(np.linalg.norm(features.astype(np.float32) - fp32) / max(np.linalg.norm(fp32), 1e-12))
            native_precision_error = float(np.linalg.norm(native - fp32) / max(np.linalg.norm(fp32), 1e-12))
        cache = {}
        for layer, kind in enumerate(manifest["config"]["text_config"]["layer_types"]):
            if kind == "linear_attention":
                cache[f"conv_{layer}"] = np.zeros((1, 3, 6144), np.float16)
                cache[f"state_{layer}"] = np.zeros((16, 128, 128), np.float32)
            else:
                for label in ["key", "value"]:
                    cache[f"{label}_{layer}"] = np.zeros((1, 2, 0, 256), np.float16)
        ids = fixture["input_ids"][0]
        image_id = manifest["image_token_id"]
        offset = int(np.flatnonzero(ids == image_id)[0])
        def run(tokens, positions, start):
            nonlocal cache
            session = sessions["decode"] if len(tokens) == 1 else sessions["decoder"]
            values = feeds_for(tokens, positions, start, features, cache, offset, image_id)
            output = dict(zip([item.name for item in session.get_outputs()],
                session.run(None, {item.name: values[item.name] for item in session.get_inputs()})))
            assert all(np.isfinite(value).all() for value in output.values()), "Non-finite ONNX activation/state"
            cache = {key.removeprefix("present_"): value for key, value in output.items() if key.startswith("present_")}
            return int(output["next_token"][0, 0])
        for start in range(0, len(ids), manifest["chunk_size"]):
            if start % 512 == 0:
                print(f"Page {example['id']}: prefill {start}/{len(ids)}", flush=True)
            end = min(len(ids), start + manifest["chunk_size"])
            token = run(ids[start:end], fixture["positions"][:, :, start:end], start)
        generated = []
        position = int(fixture["positions"].max()) + 1
        for step in range(manifest["max_new_tokens"]):
            if token in manifest["stop_ids"]:
                break
            generated.append(token)
            token = run(np.array([token]), np.full((3, 1, 1), position + step), len(ids) + step)
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        result = {"id": example["id"], "prediction": prediction, "matches_native": prediction == example["native"],
                  "runtime": ort.__version__, "provider": provider, "tf32": False if provider == "cuda" else None,
                  "tokens": len(generated), "stopped": token in manifest["stop_ids"],
                  "vision_mae": float(np.abs(error).mean()), "vision_relative_rmse": relative_rmse,
                  "vision_vs_native_fp32_relative_rmse": fp32_error,
                  "native_bf16_vs_fp32_relative_rmse": native_precision_error,
                  "seconds": time.perf_counter() - started,
                  "metrics": page_metrics(example["reference"], prediction, token not in manifest["stop_ids"]) if manifest["task"] == "page-ocr-bbox" else {
                      "exact_match": re.sub(r"\s+", " ", prediction.strip()) == re.sub(r"\s+", " ", example["reference"].strip())}}
        results.append(result)
        (directory / f"validation_onnx_{provider}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "prediction"}), flush=True)
        assert result["stopped"], "ONNX generation did not reach EOS"
        if manifest["task"] == "page-ocr-bbox":
            assert result["metrics"]["invalid_blocks"] == 0, "Invalid ONNX grounding"
        if fp32_error is not None:
            assert fp32_error < .03, "Unexpected visual feature drift from native FP32 reference"
        else:
            assert relative_rmse < 0.1, "Unexpected visual feature drift from BF16 reference"
    for index, example in enumerate(examples[:count]):
        if example.get("control") == "state-reset-repeat":
            assert results[index]["prediction"] == results[0]["prediction"], "Recurrent state leaked across requests"
    print(json.dumps({"runtime": ort.__version__, "provider": provider,
        "metrics": aggregate([r["metrics"] for r in results]) if manifest["task"] == "page-ocr-bbox" else {
            "samples": len(results), "matches_native": sum(r["matches_native"] for r in results),
            "max_vision_relative_rmse": max(r["vision_relative_rmse"] for r in results)}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    main(args.model, args.count, args.provider)
