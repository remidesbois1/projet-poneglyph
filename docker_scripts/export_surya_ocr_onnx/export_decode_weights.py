"""Index FP16 ONNX weights for the fused WebGPU decoder; no duplicate weights file."""
import argparse
import hashlib
import json
from pathlib import Path

import onnx
from safetensors import safe_open


def export(source: Path, output: Path):
    graph = onnx.load(output / "decoder.onnx", load_external_data=False)
    index = {}
    # Initializer names retain the original parameter name and a numeric suffix.
    for tensor in graph.graph.initializer:
        if tensor.external_data:
            info = {x.key: x.value for x in tensor.external_data}
            index[tensor.name.rsplit("_", 1)[0]] = {
                "offset": int(info["offset"]), "bytes": int(info["length"]),
                "shape": list(tensor.dims), "dtype": "float16", "transpose": True,
            }
    result = {}
    with safe_open(source / "model.safetensors", framework="pt") as weights:
        for key in weights.keys():
            if not key.startswith("model.language_model."):
                continue
            value = weights.get_tensor(key).float().numpy()
            name = key.removeprefix("model.language_model.")
            if value.ndim == 2:
                item = dict(index[key])
                item["transpose"] = name != "embed_tokens.weight"
                expected = list(value.T.shape if item["transpose"] else value.shape)
                if item["shape"] != expected or item["bytes"] != value.size * 2:
                    raise ValueError(f"ONNX weight layout mismatch: {key}")
                result[name] = item
            else:
                result[name] = {"shape": list(value.shape), "dtype": "float32", "values": value.reshape(-1).tolist()}
    manifest = json.loads((output / "browser_manifest.json").read_text())
    result = {"schema": 1, "source_sha256": manifest["source_sha256"],
              "data_sha256": manifest["files"]["decoder.onnx.data"]["sha256"], "weights": result}
    path = output / "decode_weights.json"
    path.write_text(json.dumps(result, separators=(",", ":")))
    manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / "browser_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Indexed {len(result['weights'])} tensors, {path.stat().st_size} bytes; ONNX weights unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)
