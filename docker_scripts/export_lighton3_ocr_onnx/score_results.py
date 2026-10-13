"""Score real WebGPU outputs with the native full-page protocol; publish aggregates only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune_lighton3_ocr_bbox"))
from grounding import aggregate, page_metrics


def main(directory):
    manifest = json.loads((directory / "browser_manifest.json").read_text())
    records = json.loads((directory / "validation_webgpu_test.json").read_text(encoding="utf-8"))
    expected = json.loads((directory / "fixtures/benchmark.json").read_text(encoding="utf-8"))
    assert len(records) == len(expected) == 189
    assert [row["id"] for row in records] == [row["id"] for row in expected], "Missing, duplicated or reordered test pages"
    assert all(row["reference"] == ref["reference"] and row["native"] == ref["native"] for row, ref in zip(records, expected))
    for name, info in manifest["files"].items():
        with (directory / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert info["sha256"] == digest and info["bytes"] == (directory / name).stat().st_size
    measured = [page_metrics(row["reference"], row["text"], row["tokenLimit"]) for row in records]
    native = [page_metrics(row["reference"], row["native"]) for row in records]
    metrics, native_metrics = aggregate(measured), aggregate(native)
    assert native_metrics["cer"] == 0.010664923421740755, "Wrong frozen native reference"
    chars = np.array([row["reference_chars"] for row in measured])
    words = np.array([row["reference_words"] for row in measured])
    edits = np.array([row["character_edits"] for row in measured])
    native_edits = np.array([row["character_edits"] for row in native])
    word_edits = np.array([row["word_edits"] for row in measured])
    rng = np.random.default_rng(42)
    indexes = rng.integers(0, len(records), size=(5000, len(records)))
    boot_cer = edits[indexes].sum(axis=1) / chars[indexes].sum(axis=1)
    boot_wer = word_edits[indexes].sum(axis=1) / words[indexes].sum(axis=1)
    boot_delta = (edits - native_edits)[indexes].sum(axis=1) / chars[indexes].sum(axis=1)
    ci = lambda values: np.quantile(values, [.025, .975]).tolist()
    totals = np.array([row["timings"]["totalMs"] for row in records])
    prefill = np.array([row["timings"]["prefillMs"] for row in records])
    fixtures = json.loads((directory / "validation_webgpu_fixtures.json").read_text(encoding="utf-8"))
    control_pass = all(not row["tokenLimit"] and row["invalidBlocks"] == 0 for row in fixtures)
    repeated = fixtures[0]["text"] == fixtures[-1]["text"]
    cuda_path = directory / "validation_onnx_cuda.json"
    cuda = json.loads(cuda_path.read_text(encoding="utf-8")) if cuda_path.exists() else []
    cuda_pass = len(cuda) == len(fixtures) and [row["id"] for row in cuda] == [row["id"] for row in fixtures] and all(
        row["provider"] == "cuda" and row["tf32"] is False and row["stopped"] and row["metrics"]["invalid_blocks"] == 0
        and row["vision_relative_rmse"] < .1 for row in cuda)
    cuda_repeat = bool(cuda and cuda[0]["prediction"] == cuda[-1]["prediction"])
    source_files = ["src/lib/suryaWebgpu.js", "src/lib/suryaFusedDecoder.js", "src/lib/lighton3Webgpu.js",
                    "src/lib/lighton3Preprocessing.js", "src/lib/lighton3Grounding.js", "src/workers/lighton3.worker.js"]
    root = Path(__file__).resolve().parents[2] / "frontend"
    report = {"schema": 1, "verified": bool(cuda_pass and cuda_repeat and control_pass and repeated and metrics["truncated_pages"] == 0 and metrics["invalid_blocks"] == 0),
              "model_id": manifest["model_id"], "weights_sha256": manifest["source_sha256"], "date": "2026-10-12",
              "hardware": "NVIDIA GeForce RTX 5090 32 GB (Blackwell)", "browser_gpu": records[0]["gpu"],
              "runtime": "ONNX Runtime Web 1.27.0 + shared Qwen3.5 fused WGSL decoder; full WebGPU, CPU EP fallback disabled",
              "precision": manifest["format"], "quantized": False, "prompt": "grounding", "thinking": False,
              "max_new_tokens": 1280, "max_long_edge": 2048, "dataset": "Poneglyph frozen full-page test",
              "split": "test, 189 pages / 1646 reference text zones; already evaluated and inspected during training",
              "protocol": "Same frozen pages and reference strings as native BF16. RGB lossless PNGs avoid new JPEG compression. NFC and normalized whitespace, case/punctuation preserved; one-to-one IoU >= 0.1 assignment; missed/extra zones penalized; page exact preserves text order and ignores coordinates. Canvas resizing can differ from Python.",
              "metrics": metrics, "native_reference_metrics": native_metrics,
              "bootstrap": {"unit": "page", "draws": 5000, "seed": 42, "cer_ci95": ci(boot_cer), "wer_ci95": ci(boot_wer), "paired_cer_delta_ci95": ci(boot_delta)},
              "native_comparison": {"cer_delta": metrics["cer"] - native_metrics["cer"],
                  "character_edits_native": int(native_edits.sum()), "character_edits_webgpu": int(edits.sum()),
                  "literal_grounding_identical_pages": sum(row["text"] == row["native"] for row in records),
                  "pages_lower_cer": int((edits < native_edits).sum()), "pages_same_cer": int((edits == native_edits).sum()),
                  "pages_higher_cer": int((edits > native_edits).sum())},
              "latency": {"scope": "Loaded model, all frozen test pages, no model downloads; length and image dimensions vary",
                  "median_ms": float(np.median(totals)), "p95_ms": float(np.quantile(totals, .95)),
                  "median_prefill_ms": float(np.median(prefill)), "total_seconds": float(totals.sum() / 1000),
                  "generated_tokens": sum(row["tokens"] for row in records),
                  "aggregate_decode_tokens_per_second": sum(max(0, row["tokens"] - 1) for row in records) / ((totals - prefill).sum() / 1000)},
              "controls": {"fixed_pages": len(fixtures), "all_complete_valid": control_pass, "state_reset_repeat_identical": repeated,
                  "old_repetition_page_complete": all(not row["tokenLimit"] for row in fixtures if row["control"] == "previous-repetition")},
              "onnx_cuda_validation": {"pages": len(cuda), "passed": cuda_pass,
                  "runtime": f"ONNX Runtime {cuda[0]['runtime']} CUDA, TF32 disabled" if cuda else None,
                  "state_reset_repeat_identical": cuda_repeat,
                  "metrics": aggregate([row["metrics"] for row in cuda]) if cuda else None,
                  "native_control_metrics": aggregate([page_metrics(row["reference"], row["native"]) for row in fixtures]),
                  "webgpu_control_metrics": aggregate([page_metrics(row["reference"], row["text"], row["tokenLimit"]) for row in fixtures]),
                  "maximum_vision_relative_rmse": max((row["vision_relative_rmse"] for row in cuda), default=None)},
              "files": manifest["files"], "engine_source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files}}
    (directory / "webgpu_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["verified", "metrics", "native_comparison", "latency", "controls", "onnx_cuda_validation"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    main(args.directory)
