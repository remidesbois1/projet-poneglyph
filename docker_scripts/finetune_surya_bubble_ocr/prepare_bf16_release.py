"""Create a smaller, verified BF16 inference copy of a Surya checkpoint."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoProcessor


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Release directory must be separate from the source model.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty release: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model, loading = AutoModelForImageTextToText.from_pretrained(
        source, dtype=torch.bfloat16, device_map="cpu", output_loading_info=True
    )
    if loading.get("missing_keys") or loading.get("unexpected_keys"):
        raise RuntimeError(f"Checkpoint did not reload cleanly: {loading}")
    model.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(source).save_pretrained(output)
    if (source / "benchmark_test.json").exists():
        shutil.copy2(source / "benchmark_test.json", output / "benchmark_test.json")
    if (source / "README.md").exists():
        readme = (source / "README.md").read_text(encoding="utf-8")
        (output / "README.md").write_text(readme.replace("Poids FP32", "Poids BF16"), encoding="utf-8")

    original_file, release_file = source / "model.safetensors", output / "model.safetensors"
    with safe_open(original_file, framework="pt", device="cpu") as original, \
         safe_open(release_file, framework="pt", device="cpu") as release:
        original_keys, release_keys = set(original.keys()), set(release.keys())
        if original_keys != release_keys:
            raise RuntimeError(f"Tensor keys differ: {original_keys ^ release_keys}")
        for key in sorted(original_keys):
            before, after = original.get_tensor(key), release.get_tensor(key)
            expected = before.to(torch.bfloat16) if before.is_floating_point() else before
            if after.dtype != expected.dtype or not torch.equal(after, expected):
                raise RuntimeError(f"BF16 conversion mismatch: {key}")
    receipt = {
        "source_sha256": sha256(original_file),
        "release_sha256": sha256(release_file),
        "source_bytes": original_file.stat().st_size,
        "release_bytes": release_file.stat().st_size,
        "verified_tensor_count": len(original_keys),
        "dtype": "bfloat16",
    }
    (output / "conversion.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
