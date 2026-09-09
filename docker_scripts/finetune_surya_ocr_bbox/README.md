---
license: openrail
language:
- fr
library_name: transformers
tags:
- surya
- qwen3_5
- vision-language-model
- ocr
- manga
- one-piece
- bbox-detection
- fine-tuned
base_model: datalab-to/surya-ocr-2
pipeline_tag: image-text-to-text
---

# Surya OCR 2 Poneglyph BBox Fine-Tune

This package mirrors the existing `finetune_lighton_ocr_bbox` flow, but swaps
the base model to `datalab-to/surya-ocr-2`.

Surya OCR 2 can return bboxes through its documented OCR, text-line detection,
and layout paths. This fine-tune uses the Hugging Face image-text-to-text model
and trains it to emit the Poneglyph page-level bbox contract:

```text
Bubble text [x1,y1,x2,y2]
```

Coordinates are normalized to `[0, 1000]`, matching the real LightOn bbox
dataset/export and benchmark scripts in this repo.

## What It Builds

- `surya_bbox_dataset/{train,val,test}/metadata.jsonl`
- resized full-page images under each split
- `outputs_surya_bbox/final_merged`
- `benchmark_surya_bbox.json`
- `benchmark_lighton_bbox.json`
- `comparison_lighton_bbox.json`
- HF-ready `README.md` generated into the final model folder

Default output repo:

```text
Remidesbois/surya-ocr-2-poneglyph-bbox
```

Default LightOn comparison repo:

```text
Remidesbois/LightonOCR-2-1b-poneglyph-bbox
```

## Run

```bash
python run_pipeline.py --dry-run --check-remote
python run_pipeline.py
```

Required env:

```text
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
```

If `pages.url_image` uses private `r2://...` references (the current production
page storage), the exporter also requires:

```text
R2_ENDPOINT
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
R2_PAGES_BUCKET_NAME
```

Private R2 pages are fetched through Cloudflare's S3-compatible API. The exporter
validates the bucket/key contract, retries transient S3 failures, compensates for
R2 request clock skew, and aborts rather than silently training on a partial
dataset when any source page cannot be read or decoded.

Optional env:

```text
HF_TOKEN
HF_REPO=Remidesbois/surya-ocr-2-poneglyph-bbox
SURYA_BBOX_PROFILE=auto
SURYA_BBOX_TRAIN_BATCH=4
SURYA_BBOX_EVAL_BATCH=2
SURYA_BBOX_GRAD_ACCUM=2
SURYA_BBOX_DATALOADER_WORKERS=2
SURYA_BBOX_REQUIRE_UPLOAD=1
SURYA_BBOX_COMPARE_LIGHTON=1
SURYA_BBOX_REQUIRE_LIGHTON_COMPARISON=1
RUNPOD_API_KEY
RUNPOD_POD_ID
```

## Dataset Contract

The exporter reads Supabase table `bulles`, filters `statut = "Valid\u00e9"` by
default, groups annotations by page, resizes each full page to a 1540px longest
side, normalizes bboxes to `[0, 1000]`, and splits by `id_page`.

Target output example:

```text
Salut Luffy ! [102,85,312,206]
On y va ! [680,410,904,538]
```

## Training Contract

The trainer uses the same Surya/Qwen3.5 image-text path as
`finetune_surya_bubble_ocr`:

- `AutoProcessor`
- `AutoModelForImageTextToText`
- LoRA by default (DoRA is explicitly opt-in)
- prompt-only generation checks during validation
- held-out generation benchmark after merge
- EOS/PAD token normalization before generation

### RTX 5090 profile

`--profile auto` detects the RTX 5090; `--profile rtx5090` selects it explicitly.
Environment variables override profile defaults, including old values in `.env`.
The profile is a starting point for 32 GiB, not a guarantee that every page size
or target length fits. Keep the GPU free of other workloads during measurement.

```text
SURYA_BBOX_TRAIN_BATCH=4
SURYA_BBOX_EVAL_BATCH=2
SURYA_BBOX_GRAD_ACCUM=2
SURYA_BBOX_DATALOADER_WORKERS=2
SURYA_BBOX_GEN_EVAL_MAX_SAMPLES=48
SURYA_BBOX_EVAL_STRATEGY=epoch
SURYA_BBOX_LOGGING_STEPS=5
```

The effective batch is 8 pages on one GPU. The default LoRA configuration is
rank 64 / alpha 128 / dropout 0.01; learning rate is 5e-5, cosine schedule,
5% warmup, six epochs. Both full-attention and DeltaNet projections are targeted.
The training dtype is BF16 when supported; full fine-tuning keeps FP32 master
weights under autocast. Fused AdamW, TF32, non-reentrant gradient checkpointing,
persistent spawn workers, pinned memory and length grouping are enabled on CUDA
as appropriate. Single-GPU only: select one device with `CUDA_VISIBLE_DEVICES`.

The collator preprocesses images once per batch and validates the tokenized
prompt/answer boundary. Padding is masked by position, preserving EOS supervision
even if PAD and EOS share an ID. The output head can omit prompt/image positions;
the loss is normalized by the supervised token count across accumulation steps.
No image resolution or answer truncation is introduced by this profile.

For a smaller memory budget, use `--profile safe` (batch 1 / accumulation 8,
evaluation batch 1, zero workers), or explicitly set a 2 / 4 pair:

```text
SURYA_BBOX_TRAIN_BATCH=2
SURYA_BBOX_GRAD_ACCUM=4
```

Changing only the microbatch also changes the effective batch, so adjust
accumulation deliberately. There is no silent OOM retry that changes training
semantics, and no silent fallback from a failed LoRA setup to full fine-tuning.

### Diagnose, smoke test, train, resume

Run these inside the installed training environment, after exporting the dataset:

```bash
# No weights or dataset download; prints CUDA, GPU, package and kernel checks.
python train_surya_bbox.py --diagnose --profile rtx5090

# Three optimizer steps in a separate smoke-* directory. No final model,
# validation, baseline comparison or upload; checks the actual training path.
python train_surya_bbox.py --profile rtx5090 --smoke-steps 3

python train_surya_bbox.py --profile rtx5090
python train_surya_bbox.py --profile rtx5090 --resume auto
# A complete explicit checkpoint path is also accepted after --resume.
```

An existing run is never overwritten by an accidental fresh start. Resume
requires optimizer, scheduler and Trainer state files; an incomplete latest
checkpoint produces a clear error instead of pretending to resume. Choose a
complete earlier checkpoint explicitly when necessary. Adapter rank, targets and
DoRA configuration are loaded from the checkpoint, not the new defaults. Keep the
same dataset, hyperparameters and software environment for consistent resumption.
To use resume through `run_pipeline.py`, set
`SURYA_BBOX_RESUME_FROM_CHECKPOINT=auto` in its environment.

Optional controls:

```text
SURYA_BBOX_OUTPUT_DIR=/workspace/outputs_surya_bbox_experiment2
SURYA_BBOX_RUN_NAME=surya-bbox-5090
SURYA_BBOX_EVAL_STRATEGY=steps
SURYA_BBOX_EVAL_STEPS=100
SURYA_BBOX_EARLY_STOPPING_PATIENCE=3
SURYA_BBOX_REQUIRE_FAST_LINEAR_ATTENTION=1
SURYA_BBOX_GROUP_BY_LENGTH=1
SURYA_BBOX_TRIM_LOGITS=1
SURYA_BBOX_USE_DORA=0
SURYA_BBOX_RICH=1
SURYA_BBOX_REPORT_TO=tensorboard
```

Early stopping is disabled unless a positive patience is provided. Generation
metrics are computed before Trainer's evaluation callbacks, so best-checkpoint
selection and early stopping see the same score. Setting
`SURYA_BBOX_GEN_EVAL_MAX_SAMPLES=0` uses validation loss instead. Generation errors
fail the run by default and are recorded in `validation_latest.json`.

`torch.compile` stays opt-in (`SURYA_BBOX_TORCH_COMPILE=1`): dynamic image/token
shapes, Triton compilation and recompilations must be measured on the actual
dataset. No FP8 or quantized training is enabled merely because the card supports
low-precision inference. `SURYA_BBOX_TRIM_LOGITS=0` is available for comparison.

### Monitoring

An interactive terminal displays a Rich progress panel; redirected logs stay
plain text. It shows train/validation loss, CER, F1, IoU, LR, gradient norm,
allocated/peak VRAM, optimizer steps per second and an indicative ETA.

The output directory contains `training_dashboard.html` (responsive, offline,
no server or CDN), `metrics.jsonl` (append-only sessions),
`training_summary.json`, `run_manifest.json` and `validation_latest.json`.
The HTML file refreshes every 15 seconds during training; its curves concern the
current session, while JSONL preserves previous sessions. HTML writes are
throttled independently of scalar logging. Resume throughput starts at the resumed
step, not step zero. ETA includes elapsed validation/checkpoint overhead; it is
not a prediction of final export/baseline duration. Non-finite logged losses or
gradient norms stop the run rather than disappearing from the log.

TensorBoard is optional; it is not needed for the local dashboard. No external
tracking/upload is enabled by the trainer. The existing pipeline's HF upload is
separate; set `SURYA_BBOX_SKIP_UPLOAD=1` to disable it.

## Benchmark Contract

The benchmark computes:

- CER / WER on bubbles matched at IoU 0.5
- mean and median IoU
- GIoU
- bbox area error
- precision / recall / F1 at IoU 0.3, 0.5, 0.75, and 0.9
- detection rate
- combined score
- average inference time

Reports now include `metrics_version: 2`: empty pages participate in the means,
the detection rate is recall at IoU 0.5 rather than a count-only ratio, and CER
is clamped only inside the combined score (raw CER may exceed 1). Consequently,
scores from older reports are not directly interchangeable with new scores.
Mean IoU still concerns matched boxes; CER/WER concern matched text, with a
penalty of 1 on nonempty pages with no matches. No-match IoU is 0. Benchmark
latency includes preprocessing and generation, synchronizes CUDA, and includes
first-use kernel overhead; it is not a warm-kernel-only throughput claim.

The same held-out pages are then evaluated with
`Remidesbois/LightonOCR-2-1b-poneglyph-bbox`, and the comparison is saved in
`comparison_lighton_bbox.json` for the generated Hugging Face README.

## Local Inference

```bash
python inference_bbox.py path/to/page.jpg --model-id outputs_surya_bbox/final_merged
```

It prints the raw generated contract, parses the bbox lines, and writes a JPEG
with drawn boxes.

## Docker

The build context is the **project root**, not this package directory, because
the image includes both `docker_scripts/common_training` and the canonical shared
prompt registry in `packages/shared/src/llm-prompts.json`. The Dockerfile-specific
ignore file still sends only required source/configuration files, excluding
datasets, caches and secrets.
The image pins PyTorch 2.8 / CUDA 12.8, Transformers 5.14.1 and PEFT 0.20.0.
It builds causal-conv1d 1.6.2.post1 for **SM120** and includes FLA 0.5.2; missing
DeltaNet fast kernels cause an explicit failure instead of a hidden slow path.
The Docker build itself only verifies that the CUDA dependencies import: Docker
build stages normally have no GPU, so FLA intentionally selects its CPU device
there. The strict Qwen3.5 fast-path and compute-capability checks run when the
container starts with `--gpus all`, where the RTX 5090 is actually visible.

From this directory (Windows Command Prompt / `.bat` syntax):

```bat
build.bat
REM or build + push:
build_and_push.bat

REM equivalent manual build:
docker build -f Dockerfile -t remidesbois/surya-ocr-bbox-finetune:latest ..\..
docker run --rm -it --gpus all --shm-size=8g ^
  --env-file ../../backend/.env ^
  --env-file ../../.env ^
  -v "%cd%\surya_bbox_dataset:/workspace/surya_bbox_dataset" ^
  -v "%cd%\outputs_surya_bbox:/workspace/outputs_surya_bbox" ^
  -v "%cd%\hf-cache:/workspace/hf-cache" ^
  remidesbois/surya-ocr-bbox-finetune:latest
```

`run_pipeline.bat` uses its own directory, shared memory for image workers and a
persistent HF/Triton cache. For local runs it loads `backend/.env` first (R2
credentials) and the project-root `.env` second (training/HF overrides). Neither
file is copied into the Docker image. For a noninteractive log, remove `-it`. To
diagnose without exporting or training:

```bash
docker run --rm --gpus all remidesbois/surya-ocr-bbox-finetune:latest python train_surya_bbox.py --diagnose
```

The native Windows Python path can use the plain PyTorch fallback, but the
documented fast-kernel environment is Linux through Docker/WSL. For another GPU
architecture, rebuild with matching `TORCH_CUDA_ARCH_LIST` and
`CAUSAL_CONV_CUDA_ARCH` build arguments; the SM120 binary is specific to 5090-class
Blackwell, not an all-GPU image.

## Regression tests

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```

The tests load selected production functions through AST to isolate numerical,
collation and monitoring behavior without model downloads or database access.
They need CPU PyTorch, NumPy and Pillow. They cover padding/EOS, suffix validation,
trimmed/full loss and gradients, unequal accumulation, evaluation-metric ordering,
empty-page/error metrics, resume guards, cache restoration and monitoring files.
They do not validate the installed Transformers/PEFT integration, CUDA kernels,
Docker build, real-model quality or achieved 5090 performance; use an actual
`--smoke-steps` run and held-out benchmark for those checks.

## Notes

- This is not a crop OCR model. It consumes a full page and returns all bubble
  text plus bboxes.
- The generated text format is strict, but still model-generated. Downstream
  parsers must reject malformed lines.
- The LightOn comparison is only meaningful when both models run on the exact
  same exported test split.
