"""Full ONNX greedy decoding on fixed, private validation fixtures."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from make_fixtures import clean


def rotation(positions):
    inv = (1 / (10000000 ** (np.arange(32, dtype=np.float32) / 32))).astype(np.float32)
    angles = positions[:, 0, :, None] * inv[None, None, :]
    freq = angles[0].copy()
    freq[:, 1:33:3] = angles[1, :, 1:33:3]
    freq[:, 2:30:3] = angles[2, :, 2:30:3]
    freq = np.concatenate([freq, freq], -1).astype(np.float32)
    return np.cos(freq)[None, None], np.sin(freq)[None, None]


def feeds_for(ids, positions, start, image_features, cache, image_offset):
    n = len(ids)
    cos, sin = rotation(positions)
    image_indices = np.where(ids == 11, np.arange(start, start + n) - image_offset, 0).astype(np.int64)
    total = start + n
    return dict(input_ids=ids[None].astype(np.int64), image_features=image_features,
                image_indices=image_indices, image_mask=(ids == 11)[None, :, None],
                rope_cos=cos, rope_sin=sin,
                attention_mask=np.where(np.arange(total)[None, :] <= np.arange(start, total)[:, None], 0, -10000).astype(np.float16)[None, None],
                identity=np.eye(n, dtype=np.float32)[None], lower=np.tri(n, dtype=np.float32)[None],
                strict_lower=np.tri(n, k=-1, dtype=np.float32)[None], **cache)


def main(directory, count):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    opts.log_severity_level = 3
    vision = ort.InferenceSession(str(directory / "vision.onnx"), opts)
    decoder = ort.InferenceSession(str(directory / "decoder.onnx"), opts)
    decode = ort.InferenceSession(str(directory / "decode.onnx"), opts)
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    manifest = json.loads((directory / "browser_manifest.json").read_text())
    examples = json.loads((directory / "fixtures/examples.json").read_text())
    results = []
    for index, example in enumerate(examples[:count]):
        begin = time.perf_counter()
        fixture = np.load(directory / f"fixtures/inputs-{index}.npz")
        vision_feeds = {x.name: fixture[x.name] for x in vision.get_inputs()}
        vision_feeds["position_indices"] = vision_feeds["position_indices"].astype(np.int64)
        features = vision.run(None, vision_feeds)[0]
        assert np.isfinite(features).all(), "Non-finite vision features"
        visual_error = float(np.mean(abs(features.astype(np.float32) - fixture["native_features"])))
        cache = {}
        for i, kind in enumerate(manifest["config"]["text_config"]["layer_types"]):
            if kind == "linear_attention":
                cache[f"conv_{i}"] = np.zeros((1, 3, 6144), np.float16)
                cache[f"state_{i}"] = np.zeros((16, 128, 128), np.float32)
            else:
                cache[f"key_{i}"] = cache[f"value_{i}"] = np.zeros((1, 2, 0, 256), np.float16)
        ids = fixture["input_ids"][0]
        offset = int(np.where(ids == 11)[0][0])
        output_names = [x.name for x in decoder.get_outputs()]
        def run(chunk_ids, positions, start):
            nonlocal cache
            session = decode if len(chunk_ids) == 1 else decoder
            feeds = feeds_for(chunk_ids, positions, start, features, cache, offset)
            feeds = {x.name: feeds[x.name] for x in session.get_inputs()}
            output = dict(zip(output_names, session.run(None, feeds)))
            for key, value in output.items():
                assert np.isfinite(value).all(), f"Non-finite {key}"
            cache = {key.removeprefix("present_"): value for key, value in output.items() if key.startswith("present_")}
            return int(output["next_token"][0, 0])
        for start in range(0, len(ids), 32):
            next_token = run(ids[start:start+32], fixture["positions"][:, :, start:start+32], start)
        generated = []
        position = int(fixture["positions"].max()) + 1
        for step in range(256):
            if next_token == 2:
                break
            generated.append(next_token)
            next_token = run(np.array([next_token]), np.full((3, 1, 1), position + step), len(ids) + step)
        text = clean(tokenizer.decode(generated, skip_special_tokens=True))
        result = {"id": example["id"], "text": text, "native": example["native"], "reference": example["reference"],
                  "tokens": len(generated), "stopped": next_token == 2, "seconds": time.perf_counter() - begin,
                  "vision_mae_vs_bf16": visual_error, "matches_native": text == example["native"]}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    (directory / "validation_python.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    assert all(r["stopped"] and r["text"] for r in results), "Incomplete generation"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()
    main(args.model, args.count)
