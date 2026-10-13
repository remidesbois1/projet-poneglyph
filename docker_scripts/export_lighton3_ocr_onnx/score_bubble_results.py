"""Paired native/WebGPU evaluation of the frozen 1612 bubble test; aggregates only."""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def normalize(text):
    return re.sub(r"\s+", " ", str(text or "").strip())


def distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def measure(reference, prediction):
    ref, pred = normalize(reference), normalize(prediction)
    return [distance(ref, pred), len(ref), distance(ref.split(), pred.split()), len(ref.split()), int(ref == pred)]


def aggregate(values):
    sums = np.asarray(values, dtype=np.int64).sum(axis=0)
    return {"cer": float(sums[0] / sums[1]), "wer": float(sums[2] / sums[3]),
            "exact_match": float(sums[4] / len(values)), "sample_count": len(values),
            "character_edits": int(sums[0]), "reference_characters": int(sums[1]),
            "word_edits": int(sums[2]), "reference_words": int(sums[3]), "exact_count": int(sums[4])}


def main(directory, date=None):
    manifest = json.loads((directory / "browser_manifest.json").read_text(encoding="utf-8"))
    records = json.loads((directory / "validation_webgpu_test.json").read_text(encoding="utf-8"))
    expected = json.loads((directory / "fixtures/benchmark.json").read_text(encoding="utf-8"))
    assert manifest["task"] == "bubble-ocr" and manifest["prompt"] == "" and manifest["max_new_tokens"] == 256
    assert len(records) == len(expected) == 1612
    assert [r["id"] for r in records] == [r["id"] for r in expected], "Missing/duplicated/reordered crops"
    assert all(r["reference"] == e["reference"] and r["native"] == e["native"] and r["page_id"] == e["page_id"]
               for r, e in zip(records, expected)), "Frozen references changed"
    assert all(r["provider"] == "webgpu" and r["implementation"] == "wgsl-fused" for r in records)
    for name, info in manifest["files"].items():
        path = directory / name
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == info["sha256"]
        assert path.stat().st_size == info["bytes"]
    measured = np.array([measure(r["reference"], r["text"]) for r in records])
    native = np.array([measure(r["reference"], r["native"]) for r in records])
    metrics, native_metrics = aggregate(measured), aggregate(native)
    assert native_metrics["character_edits"] == 202 and native_metrics["reference_characters"] == 52064, "Wrong native checkpoint/reference"
    assert abs(native_metrics["wer"] - .01448989650073928) < 1e-12
    metrics.update(blank_count=sum(not normalize(r["text"]) for r in records),
                   token_limit_count=sum(bool(r["tokenLimit"]) for r in records))
    pages = defaultdict(lambda: np.zeros(7, dtype=np.int64))
    for r, web, ref in zip(records, measured, native):
        pages[r["page_id"]] += np.array([web[0], ref[0], web[1], web[2], web[3], web[4], 1])
    page_values = np.stack(list(pages.values()))
    rng = np.random.default_rng(42)
    sums = page_values[rng.integers(0, len(pages), size=(5000, len(pages)))].sum(axis=1)
    ci = lambda v: np.quantile(v, [.025, .975]).tolist()
    fixtures = json.loads((directory / "validation_webgpu_fixtures.json").read_text(encoding="utf-8"))
    fixed = json.loads((directory / "fixtures/examples.json").read_text(encoding="utf-8"))
    cuda = json.loads((directory / "validation_onnx_cuda.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in fixtures] == [e["id"] for e in fixed] == [r["id"] for r in cuda]
    repeat = len(fixtures) == 16 and fixed[-1]["control"] == "state-reset-repeat" and fixtures[0]["text"] == fixtures[-1]["text"]
    control_pass = all(not r["tokenLimit"] and r["provider"] == "webgpu" for r in fixtures)
    cuda_pass = len(cuda) == 16 and all(r["provider"] == "cuda" and r["tf32"] is False and r["stopped"] and r["vision_vs_native_fp32_relative_rmse"] < .03 for r in cuda)
    cuda_repeat = cuda[0]["prediction"] == cuda[-1]["prediction"]
    resize = json.loads((directory / "validation_resize.json").read_text(encoding="utf-8"))
    assert resize["passed"] and resize["cases"] == 22 and all(r["different_channels"] == 0 for r in resize["results"])
    totals = np.array([r["timings"]["totalMs"] for r in records])
    prefill = np.array([r["timings"]["prefillMs"] for r in records])
    delta = metrics["cer"] - native_metrics["cer"]
    root = Path(__file__).resolve().parents[2]
    sources = ["lib/suryaWebgpu.js", "lib/suryaFusedDecoder.js", "lib/suryaPreprocessing.js", "lib/lighton3BubbleWebgpu.js",
               "lib/lighton3BubblePreprocessing.js", "lib/bicubicResize.js",
               "workers/lighton3Bubble.worker.js"]
    passed = repeat and control_pass and cuda_pass and cuda_repeat and not metrics["blank_count"] and not metrics["token_limit_count"] and delta <= .0001
    report = {"schema": 1, "verified": bool(passed), "task": "bubble-ocr", "model_id": manifest["model_id"],
        "weights_sha256": manifest["source_sha256"], "source_revision": manifest["source_revision"],
        "date": date or datetime.now(timezone.utc).date().isoformat(), "hardware": "NVIDIA GeForce RTX 5090 32 GB (Blackwell)",
        "browser_gpu": records[0]["gpu"], "runtime": "ONNX Runtime Web 1.27.0 + Qwen3.5 fused WGSL decoder; WebGPU, CPU EP fallback disabled",
        "precision": manifest["format"], "quantized": False, "prompt": "", "thinking": False,
        "max_new_tokens": 256, "min_pixels": 65536, "max_pixels": 1048576,
        "dataset": "Poneglyph frozen Surya single-bubble crops", "split": "test; 1612 crops, 185 pages; previously evaluated for Surya and native LightOnOCR-3",
        "protocol": "One lossless RGB crop per request; empty image-only user prompt, thinking disabled, greedy decoding. Collapse whitespace only; preserve visible quotes, case, accents and punctuation. Native Torchvision uint8 CPU antialiased bicubic resize ported to JavaScript, verified pixel-exact on 22 cases. Conversion fidelity on an already inspected test, not a new independent benchmark.",
        "scoring_version": "text-v2-preserve-visible-quotes", "metrics": metrics, "native_reference_metrics": native_metrics,
        "bootstrap": {"unit": "page", "draws": 5000, "seed": 42, "cer_ci95": ci(sums[:, 0] / sums[:, 2]),
                      "wer_ci95": ci(sums[:, 3] / sums[:, 4]), "exact_match_ci95": ci(sums[:, 5] / sums[:, 6]),
                      "paired_cer_delta_ci95": ci((sums[:, 0] - sums[:, 1]) / sums[:, 2])},
        "native_comparison": {"cer_delta": delta, "character_edits_native": native_metrics["character_edits"],
            "character_edits_webgpu": metrics["character_edits"], "literal_identical_crops": sum(r["text"] == r["native"] for r in records),
            "crops_lower_cer": int((measured[:, 0] < native[:, 0]).sum()), "crops_same_cer": int((measured[:, 0] == native[:, 0]).sum()),
            "crops_higher_cer": int((measured[:, 0] > native[:, 0]).sum())},
        "latency": {"scope": "Loaded model; sequential production worker; all 1612 frozen crops; downloads excluded",
            "median_ms": float(np.median(totals)), "p95_ms": float(np.quantile(totals, .95)),
            "median_prefill_ms": float(np.median(prefill)), "total_seconds": float(totals.sum() / 1000),
            "generated_tokens": sum(r["tokens"] for r in records),
            "aggregate_decode_tokens_per_second": sum(max(0, r["tokens"] - 1) for r in records) / ((totals - prefill).sum() / 1000)},
        "controls": {"fixed_crops": len(fixtures), "all_complete": control_pass, "state_reset_repeat_identical": repeat,
                     "matches_native": sum(r["text"] == e["native"] for r, e in zip(fixtures, fixed))},
        "pixel_resize_validation": {"native_processor": resize["native_processor"], "cases": resize["cases"], "passed": resize["passed"],
                                    "different_channels": sum(r["different_channels"] for r in resize["results"])},
        "onnx_cuda_validation": {"crops": len(cuda), "passed": cuda_pass, "state_reset_repeat_identical": cuda_repeat,
            "runtime": f"ONNX Runtime {cuda[0]['runtime']}, CUDA 12.8, TF32 disabled", "matches_native": sum(r["matches_native"] for r in cuda),
            "maximum_vision_relative_rmse": max(r["vision_relative_rmse"] for r in cuda),
            "maximum_vision_vs_native_fp32_relative_rmse": max(r["vision_vs_native_fp32_relative_rmse"] for r in cuda),
            "maximum_native_bf16_vs_fp32_relative_rmse": max(r["native_bf16_vs_fp32_relative_rmse"] for r in cuda),
            "vision_reference": "Native FP32 vision, TF32 disabled, same FP16 input pixels; separates export error from native BF16 rounding"},
        "release_criterion": "All crops scored, no empty/truncated outputs; 16 CUDA and browser controls pass; repeat identical; CER increase <= 0.01 percentage points",
        "files": manifest["files"], "engine_source_sha256": {s: hashlib.sha256((root / "frontend/src" / s).read_bytes()).hexdigest() for s in sources}}
    (directory / "webgpu_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["verified", "metrics", "native_comparison", "latency", "controls", "onnx_cuda_validation"]}), flush=True)
    if not passed:
        raise RuntimeError("Conversion did not pass its documented release criterion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--date", help="Evaluation date in the user's timezone")
    args = parser.parse_args()
    main(args.directory, args.date)
