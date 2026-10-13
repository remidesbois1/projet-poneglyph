"""Document measured bubble conversion quality and generate aggregate comparison figure."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(directory, model_card):
    report = json.loads((directory / "webgpu_metrics.json").read_text(encoding="utf-8"))
    if not report["verified"] or report["task"] != "bubble-ocr":
        raise RuntimeError("Validate all native/ONNX/browser controls before documenting a release")
    native, browser, latency = report["native_reference_metrics"], report["metrics"], report["latency"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    fig.patch.set_facecolor("#101827")
    for ax, key, title in zip(axes, ["cer", "wer", "exact_match"], ["CER ↓", "WER ↓", "Exact match ↑"]):
        ax.set_facecolor("#101827")
        values = [100 * native[key], 100 * browser[key]]
        bars = ax.bar(["Native BF16", "WebGPU"], values, color=["#94a3b8", "#38bdf8"], width=.6)
        ax.set_title(title, color="white", pad=14)
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.set_ylabel("%", color="#cbd5e1")
        ax.set_ylim(0, max(values) * 1.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * .025,
                    f"{value:.3f}%" if key != "exact_match" else f"{value:.2f}%", ha="center", color="white")
    fig.suptitle("LightOnOCR-3 · Poneglyph bubbles · 1,612 frozen test crops / 185 pages", color="white", fontsize=14)
    fig.savefig(directory / "quality_comparison.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    table = "| 1,612 frozen test crops | Native BF16 | Browser WebGPU |\n|---|---:|---:|\n"
    for name, key, digits in [("CER ↓", "cer", 4), ("WER ↓", "wer", 4), ("Exact match ↑", "exact_match", 2)]:
        table += f"| {name} | {native[key] * 100:.{digits}f}% | {browser[key] * 100:.{digits}f}% |\n"
    table += f"| Character errors / 52,064 chars | {native['character_edits']} | {browser['character_edits']} |\n"
    comparison = report["native_comparison"]
    ci = report["bootstrap"]["paired_cer_delta_ci95"]
    summary = table + f"""
**{comparison['literal_identical_crops']} / 1,612** literal transcripts identical to the native reference.
No empty outputs and no token-limit hits. CER change: **{100 * comparison['cer_delta']:+.4f} percentage points**;
paired page-cluster bootstrap 95% interval **[{100 * ci[0]:+.4f}, {100 * ci[1]:+.4f}] percentage points** (5,000 resamples, seed 42).
Loaded model on RTX 5090 / Blackwell: median **{latency['median_ms']:.0f} ms/bubble**, p95 **{latency['p95_ms']:.0f} ms**;
downloads excluded. Other devices and crop/text lengths will have different latencies.

The test is the same frozen set already evaluated for Surya and native LightOnOCR-3.
This measures conversion fidelity, not a new independent benchmark. Whitespace is collapsed;
visible quotes, capitalization, accents and punctuation are preserved. The native
Torchvision uint8 antialiased bicubic resize is reproduced in JavaScript; FP16/FP32
arithmetic can still differ from BF16 inference.
"""
    docs = f"""# LightOnOCR-3-Poneglyph · bubble ONNX / WebGPU

Native checkpoint: `1175`; safetensors SHA-256 `{report['weights_sha256']}`.
Native revision: `{report['source_revision']}`. These assets are for **one RGB bubble crop**.

## Measured quality

{summary}
![Aggregate conversion comparison](quality_comparison.png)

## Input and generation contract

- Empty image-only user prompt, saved chat template, `enable_thinking=False`.
- Qwen aspect-preserving resize on a 32-pixel grid, **65,536–1,048,576 pixels**;
  RGB normalization [-1, 1], duplicated temporal patches and native multimodal RoPE.
  Pixel-exact native uint8 antialiased bicubic resize verified on 22 cases; Canvas
  is used only to decode source pixels, without resampling.
- Greedy generation, EOS 248046, **256 new tokens maximum**, full 248,320-token vocabulary.
- Text output; the bubble rectangle is supplied by the app. This model does not generate bbox.
- FP16 weights/projections, FP32 residuals, normalizations and DeltaNet recurrent states;
  no INT8/INT4 quantization, no vocabulary pruning.

## Portable graphs and browser execution

`vision.onnx` + `vision.onnx.data`: full vision encoder, exact global attention split only
along queries. `decoder.onnx` / `decode.onnx` + shared `decoder.onnx.data`: chunked prompt
prefill and single-token decoding, explicit KV and DeltaNet states. `prefill.onnx` is the
state-only prefill graph. Standard ONNX operators, opset 17; `prefill-fast.onnx` uses
standard RMSNormalization, opset 23. No custom ONNX domain, Loop or Scan.

Poneglyph uses ONNX Runtime Web **1.27.0**, WebGPU FP16 and `prefill-fast.onnx`, followed
by the shared Qwen3.5 WGSL decoder indexed by `decode_weights.json`. The fused decoder
uses the exact same ONNX weights on the ORT GPUDevice, fixed-capacity caches, fused
norm/conv/recurrence, full-vocabulary argmax and batched token readback. Vision features,
prefill states and decoding stay on GPU; the CPU execution-provider fallback is disabled.
An all-ONNX decoding path remains available with `load(progress, {{ mode: 'onnx' }})`.

The loader checks WebGPU and shader-f16 support, resumes chunk downloads, verifies sizes
and SHA-256, and uses a pinned Hub revision. Requests run serially in a worker; cancellation
resets request state, and disposal releases the ONNX sessions and GPU buffers. Bubble
and full-page workers have separate model IDs, asset paths, prompts and preprocessing.

## Verification

All five graphs pass ONNX checker. **{report['onnx_cuda_validation']['matches_native']} / 16** CUDA and **{report['controls']['matches_native']} / 16** browser control
transcripts match native BF16, including visible quotes, accents, uppercase and a repeated
request. Every ONNX intermediate/state checked is finite; generation reaches EOS.
ORT GPU **1.26.0 / CUDA 12.8**, TF32 disabled, is used for the portable CUDA validation;
the browser runtime is independently **1.27.0 / WebGPU**.

For vision, native BF16 differs from native FP32 by up to
{100 * report['onnx_cuda_validation']['maximum_native_bf16_vs_fp32_relative_rmse']:.2f}% relative RMSE on these crops.
Comparing the exported encoder to native **FP32** with identical input pixels isolates
export error: maximum relative RMSE
**{100 * report['onnx_cuda_validation']['maximum_vision_vs_native_fp32_relative_rmse']:.3f}%**.
The final transcription comparison remains against the actual BF16 model.

The full browser test uses the production app worker on all **1,612** frozen RGB crops,
without a new JPEG compression. 22 pixel controls match Torchvision byte for byte,
including upscaling, downscaling and one-axis resizing. Metrics, intervals, control results, latency and asset hashes
are in [webgpu_metrics.json](webgpu_metrics.json). Training crops, annotations, fixtures
and individual predictions are never uploaded.

Implementation: Poneglyph `Lighton3BubbleWebGPU`, `lighton3Bubble.worker.js`, annotation
model selector and `/sandbox/lighton3-bubble`. Existing page OCR uses
[LightOnOCR-3-Poneglyph-bbox](https://huggingface.co/Remidesbois/LightOnOCR-3-Poneglyph-bbox).

References: [original model contract](https://huggingface.co/lightonai/LightOnOCR-3-0.8B),
[ORT GPU tensor lifecycle](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html).
"""
    (directory / "README.md").write_text(docs, encoding="utf-8")
    source = model_card.read_text(encoding="utf-8")
    source = source.replace("Only native weights, processor, model card and aggregate metrics are published.",
                            "Native weights, processor, validated ONNX assets and aggregate metrics are published.")
    start = source.find("<!-- onnx-bubble:start -->")
    if start >= 0:
        end = source.index("<!-- onnx-bubble:end -->", start) + len("<!-- onnx-bubble:end -->")
        source = source[:start] + source[end:]
    section = "\n\n<!-- onnx-bubble:start -->\n## ONNX and browser WebGPU\n\n" + summary
    section += "\n![Aggregate native/WebGPU comparison](onnx/quality_comparison.png)\n\n"
    section += "Portable ONNX graphs, manifest and fused WebGPU weight index are in [onnx/](onnx/README.md). "
    section += "Poneglyph offers **LightOnOCR 3 · bulles** beside Surya and Falcon for bubble crops. "
    section += "Empty image-only prompt, thinking disabled, 256-token budget; visible quotes are kept. "
    section += "[Complete browser measurements and asset hashes](onnx/webgpu_metrics.json).\n<!-- onnx-bubble:end -->\n"
    (directory / "model_card.md").write_text(source.rstrip() + section, encoding="utf-8")
    print("Wrote verified aggregate model card, ONNX documentation and comparison plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--model-card", type=Path, required=True)
    args = parser.parse_args()
    main(args.directory, args.model_card)
