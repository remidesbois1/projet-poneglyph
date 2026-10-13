"""Native LightOnOCR-3 bubble SFT, reusing the proven Surya loss/data/eval helpers."""
import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_ID = "lightonai/LightOnOCR-3-0.8B"
BASE_REVISION = "36d8636925327c671bd8e479c517a004f7566bd9"
DEFAULTS = {"MODEL_ID": BASE_ID, "DATASET_DIR": "/data/bubbles", "OUTPUT_DIR": "/workspace/outputs/rtx5090-20261012-bubble",
            "USER_PROMPT": "", "PROFILE": "rtx5090", "TRAIN_BATCH": "16", "GRAD_ACCUM": "2", "EVAL_BATCH": "16",
            "GEN_EVAL_BATCH": "32", "GEN_EVAL_MAX_SAMPLES": "1648", "MAX_NEW_TOKENS": "256",
            "FINAL_TEST_MAX_SAMPLES": "0", "EPOCHS": "6", "LR": "1.2e-5", "WARMUP_STEPS": "60",
            "EARLY_STOPPING_PATIENCE": "2", "TRAIN_MODE": "hybrid", "VISION_TRAIN_LAST_BLOCKS": "4",
            "REQUIRE_FAST_LINEAR_ATTENTION": "1", "EXPECTED_COMPUTE_CAPABILITY": "12.0",
            "DATALOADER_WORKERS": "0", "IMAGE_CACHE_ITEMS": "8192", "AUGMENT_PROBABILITY": "0.50",
            "GRADIENT_CHECKPOINTING": "0", "TORCH_COMPILE": "0", "RESUME_FROM_CHECKPOINT": "none",
            "RICH": "0", "REPORT_TO": "tensorboard", "SAVE_TOTAL_LIMIT": "2", "LOGGING_STEPS": "10"}
for key, value in DEFAULTS.items():
    os.environ["SURYA_" + key] = os.environ.get("LIGHTON3_BUBBLE_" + key, value)
for key, value in list(os.environ.items()):
    if key.startswith("LIGHTON3_BUBBLE_"):
        os.environ["SURYA_" + key[len("LIGHTON3_BUBBLE_"):]] = value
sys.path.insert(0, str(SCRIPT_DIR.parent / "finetune_surya_bubble_ocr"))
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.trainer_callback import PrinterCallback
from safetensors import safe_open
from transformers.models.qwen3_5 import modeling_qwen3_5
import train_surya_bubble_ocr as core
from training_monitor import atomic_json, write_final_report
from bubble_monitor import BubbleMonitor, write_dashboard
NATIVE_RMS_NORM = modeling_qwen3_5.Qwen3_5RMSNorm
NATIVE_MLP = modeling_qwen3_5.Qwen3_5MLP


def restore_native_classes():
    modeling_qwen3_5.Qwen3_5RMSNorm = NATIVE_RMS_NORM
    modeling_qwen3_5.Qwen3_5MLP = NATIVE_MLP

if os.environ["SURYA_USER_PROMPT"] != "":
    raise ValueError("This training contract requires the official empty OCR prompt")
core.USER_PROMPT = ""
legacy_postprocess_prediction = core.postprocess_prediction


def normalize_transcription(text):
    # Visible quotes are transcription characters, not an assistant envelope.
    # Reference extraction uses the same whitespace normalization.
    return re.sub(r"\s+", " ", str(text or "").strip())


core.postprocess_prediction = normalize_transcription


def messages_for_entry(image_path, reference_text=None):
    messages = [{"role": "user", "content": [{"type": "image", "image": str(image_path)}]}]
    if reference_text is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": reference_text}]})
    return messages


def apply_template(processor, messages, add_generation_prompt):
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)


core.messages_for_entry = messages_for_entry
core.apply_template = apply_template
base_packed_targets = core.add_packed_targets


def weighted_packed_targets(batch, sample_weights, eos_token_id=248046, eos_weight=1.0):
    batch = base_packed_targets(batch, sample_weights)
    if eos_weight == 1.0:
        return batch
    if eos_weight <= 0:
        raise ValueError("EOS weight must be positive")
    labels = batch["labels"]
    rows, columns = labels[:, 1:].ne(-100).nonzero(as_tuple=True)
    targets = labels[rows, columns + 1]
    factors = torch.where(targets.eq(eos_token_id), eos_weight, 1.0).float()
    denominator = torch.zeros(len(sample_weights), dtype=torch.float32)
    denominator.scatter_add_(0, rows, factors)
    if denominator.le(0).any():
        raise ValueError("Missing supervised assistant tokens")
    batch["ocr_token_weights"] = factors * (sample_weights / denominator)[rows]
    return batch


core.add_packed_targets = lambda batch, weights: weighted_packed_targets(
    batch, weights, eos_weight=float(os.getenv("LIGHTON3_BUBBLE_EOS_WEIGHT", "1.0")))
native_benchmark = core.run_generation_benchmark
native_generate = core.generate_predictions
ACTIVE_MONITOR = None
GENERATION_PROGRESS = None
STANDALONE_PHASE = "baseline"


def progress_generate(*args, **kwargs):
    global GENERATION_PROGRESS
    result = native_generate(*args, **kwargs)
    if GENERATION_PROGRESS is not None:
        progress = GENERATION_PROGRESS
        entries = args[2] if len(args) > 2 else kwargs["entries"]
        progress["processed"] += len(entries)
        now = time.perf_counter()
        if now - progress["updated"] >= 10 or progress["processed"] >= progress["total"]:
            progress["updated"] = now
            seconds = now - progress["started"]
            values = {"generation_processed": progress["processed"], "generation_total": progress["total"],
                      "generation_elapsed_seconds": seconds,
                      "generation_remaining_seconds": seconds * (progress["total"] - progress["processed"]) / max(progress["processed"], 1)}
            if ACTIVE_MONITOR:
                ACTIVE_MONITOR.phase(ACTIVE_MONITOR.summary["status"], **values)
            else:
                write_dashboard(core.OUTPUT_DIR / "training_dashboard.html",
                                {**progress.get("summary", {}), "status": STANDALONE_PHASE, **values}, progress.get("history", []))
    return result


core.generate_predictions = progress_generate


def autocast_benchmark(model, *args, **kwargs):
    global GENERATION_PROGRESS
    dataset = args[1] if len(args) > 1 else kwargs.get("dataset")
    previous = GENERATION_PROGRESS
    if dataset is not None:
        now = time.perf_counter()
        GENERATION_PROGRESS = {"started": now, "updated": now, "processed": 0,
                               "total": len(core.benchmark_indices(dataset, kwargs.get("max_samples")))}
        summary_path = core.OUTPUT_DIR / "training_summary.json"
        history_path = core.OUTPUT_DIR / "metrics.jsonl"
        if ACTIVE_MONITOR is None and summary_path.exists():
            GENERATION_PROGRESS["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
            GENERATION_PROGRESS["history"] = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
                                                if json.loads(line).get("event") == "metrics"] if history_path.exists() else []
        if ACTIVE_MONITOR:
            ACTIVE_MONITOR.phase(ACTIVE_MONITOR.summary["status"], generation_processed=0,
                                 generation_total=GENERATION_PROGRESS["total"], generation_remaining_seconds=0)
    # Master parameters stay FP32; generation validation uses the inference
    # precision rather than silently decoding the whole model in FP32.
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            return native_benchmark(model, *args, **kwargs)
    finally:
        GENERATION_PROGRESS = previous


core.run_generation_benchmark = autocast_benchmark


def load_processor(source=BASE_ID):
    kwargs = {"revision": BASE_REVISION} if source == BASE_ID else {}
    processor = AutoProcessor.from_pretrained(source, trust_remote_code=False, **kwargs)
    processor.tokenizer.padding_side = "left"
    image_processor = processor.image_processor
    if isinstance(image_processor.size, dict):
        image_processor.size["shortest_edge"] = core.MIN_IMAGE_PIXELS
        image_processor.size["longest_edge"] = core.MAX_IMAGE_PIXELS
    else:
        image_processor.size.shortest_edge = core.MIN_IMAGE_PIXELS
        image_processor.size.longest_edge = core.MAX_IMAGE_PIXELS
    return processor


def load_model(source=BASE_ID, training=False):
    restore_native_classes()
    if training and source == BASE_ID:
        source = os.getenv("LIGHTON3_BUBBLE_INIT_MODEL") or source
    kwargs = {"revision": BASE_REVISION} if source == BASE_ID else {}
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.float32 if training else torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", trust_remote_code=False, **kwargs)
    core.configure_generation(model)
    if training:
        model, _ = core.configure_trainable_model(model)
        try:
            core.install_training_acceleration(model)
        finally:
            # Liger 0.8 also changes the global constructors despite model=.
            # Keep the per-instance training patches, but restore the HF classes
            # so a subsequent model load really uses standard Transformers.
            restore_native_classes()
        model.config.use_cache = False
    return model


def training_args(smoke_steps=0):
    args = core.make_training_args(smoke_steps)
    # Keep FP32 master parameters when running loss validation outside train().
    args.bf16_full_eval = False
    return args


class BubbleTrainer(core.SuryaOCRTrainer):
    def evaluation_loop(self, *args, **kwargs):
        if getattr(self, "monitor", None):
            self.monitor.phase("validating")
        try:
            return super().evaluation_loop(*args, **kwargs)
        finally:
            if getattr(self, "monitor", None):
                self.monitor.phase("training")


def read_audit():
    path = core.OUTPUT_DIR / "dataset_audit.json"
    if not path.exists():
        raise RuntimeError("Run the pixel/page audit before training")
    report = json.loads(path.read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        digest = hashlib.sha256((core.DATASET_DIR / split / "metadata.jsonl").read_bytes()).hexdigest()
        if digest != report["splits"][split]["metadata_sha256"]:
            raise RuntimeError("Dataset metadata changed since its audit")
    return report


def benchmark(source, split, output, maximum=None):
    global STANDALONE_PHASE
    STANDALONE_PHASE = "baseline" if source == BASE_ID else ("validating" if split == "val" else "testing")
    processor = load_processor(source)
    model = load_model(source).eval()
    patched = [name for name, module in model.named_modules()
               if "liger" in type(module).__module__ or "liger" in getattr(module.forward, "__module__", "")]
    if patched or getattr(model, "_surya_training_acceleration", None):
        raise RuntimeError(f"Native inference still has training patches: {patched[:5]}")
    core.configure_generation(model, processor)
    dataset = core.prepare_dataset(core.DATASET_DIR / split / "metadata.jsonl", split, processor)
    metrics, samples = core.run_generation_benchmark(model, processor, dataset, split_name=split,
        max_samples=maximum, title="LIGHTONOCR-3 BUBBLE " + split.upper(), worst_count=0)
    metrics["confidence_intervals"] = core.bootstrap_test_metrics(samples, iterations=5000)
    metrics["native_forward_modules_verified"] = True
    metrics["scoring_version"] = "text-v2-preserve-visible-quotes"
    metrics["legacy_surya_postprocess"] = core.compute_text_metrics([
        {**s, "prediction": legacy_postprocess_prediction(s["raw_prediction"])} for s in samples])
    strict = [{**s, "prediction": s["raw_prediction"].strip()} for s in samples]
    metrics["strict_raw_text"] = core.compute_text_metrics(strict)
    payload = core.save_benchmark(output, metrics, samples)
    payload.update(model_id=BASE_ID if source == BASE_ID else "Remidesbois/LightOnOCR-3-Poneglyph", weights_source=source)
    atomic_json(output, payload)
    return metrics, samples


def calibrate(batch_size):
    core.configure_torch_runtime()
    core.set_seed(core.RANDOM_SEED)
    runtime = core.validate_training_runtime(True)
    processor = load_processor()
    dataset = core.prepare_dataset(core.TRAIN_FILE, "train", processor)
    ordered = sorted(range(len(dataset)), key=lambda i: dataset[i]["length"], reverse=True)[:batch_size]
    examples = [dataset[i] for i in ordered]
    collator = core.SuryaBubbleCollator(processor, training=False)
    batch = collator(examples)
    batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
    model = load_model(training=True).train()
    core.configure_generation(model, processor)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1.2e-5, weight_decay=0.05, fused=True)
    records = []
    for step in range(3):
        # Keep gradient buffers after the first update, as real accumulation does.
        optimizer.zero_grad(set_to_none=step == 0)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = core.SuryaOCRTrainer.compute_loss(None, model, batch)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        records.append({"seconds": time.perf_counter() - started, "loss": loss.item(), "grad_norm": grad.item(),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
    report = {"batch_size": batch_size, "sequence_shape": list(batch["input_ids"].shape), "runtime": runtime,
              "model_master_dtype": str(next(model.parameters()).dtype), "measurements": records,
              "trainable_parameters": sum(p.numel() for p in params), "total_parameters": sum(p.numel() for p in model.parameters())}
    atomic_json(core.OUTPUT_DIR / f"calibration_batch_{batch_size}.json", report)
    print(json.dumps(report), flush=True)


def export_selected(trainer, processor):
    checkpoint = Path(trainer.state.best_model_checkpoint)
    if not checkpoint.is_dir():
        raise RuntimeError("Missing CER-selected checkpoint")
    # Trainer has loaded the selected checkpoint. Export a native BF16 model;
    # then compare every saved tensor against the FP32 checkpoint's BF16 cast.
    model = trainer.model
    core.configure_generation(model, processor)
    model.to(dtype=torch.bfloat16)
    core.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(core.FINAL_DIR, safe_serialization=True)
    processor.save_pretrained(core.FINAL_DIR)
    shards = list(checkpoint.glob("model*.safetensors"))
    selected = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                selected[key] = shard
    exported = core.FINAL_DIR / "model.safetensors"
    count = 0
    with safe_open(exported, framework="pt", device="cpu") as target:
        if set(target.keys()) != set(selected):
            raise RuntimeError("Native export and selected checkpoint keys differ")
        for key in target.keys():
            with safe_open(selected[key], framework="pt", device="cpu") as source:
                if not torch.equal(target.get_tensor(key), source.get_tensor(key).to(torch.bfloat16)):
                    raise RuntimeError(f"Export differs from selected checkpoint: {key}")
            count += 1
    with exported.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {"selected_checkpoint": str(checkpoint), "selection_metric": "full validation CER",
              "selection_cer": trainer.state.best_metric, "best_step": int(checkpoint.name.split('-')[-1]),
              "all_native_tensors_verified": count, "selected_checkpoint_bf16_cast_exact": True,
              "weights_sha256": digest, "weights_bytes": exported.stat().st_size}
    atomic_json(core.OUTPUT_DIR / "model_verification.json", report)
    return report


def write_model_card(metrics, verification):
    ci = metrics["confidence_intervals"]["cer_95"]
    text = f'''---
license: apache-2.0
base_model: lightonai/LightOnOCR-3-0.8B
library_name: transformers
pipeline_tag: image-text-to-text
language:
- fr
tags:
- ocr
- manga
- qwen3_5
- bubble-transcription
---

# LightOnOCR-3-Poneglyph

Native 0.8B fine-tune for French manga speech bubbles. Input: one RGB bubble crop.
Output: its text, with case, accents and punctuation preserved. Use the official
**empty transcription prompt** (image only) and `enable_thinking=False`.

For full-page detection and bounding boxes, use
[LightOnOCR-3-Poneglyph-bbox](https://huggingface.co/Remidesbois/LightOnOCR-3-Poneglyph-bbox).

## Evaluation

| Metric | Value |
| --- | ---: |
| Test crops / pages | 1,612 / 185 |
| CER | {metrics['cer']*100:.4f}% |
| WER | {metrics['wer']*100:.4f}% |
| Exact match | {metrics['exact_match']*100:.2f}% |
| CER 95% interval, page-cluster bootstrap | {ci[0]*100:.4f}–{ci[1]*100:.4f}% |
| Empty / token-limited outputs | {round(metrics['blank_rate']*1612)} / {metrics['token_limit_count']} |

Frozen page-disjoint split: 7,496 training crops (860 pages), 1,648 validation
crops (184 pages), 1,612 test crops (185 pages). Pixel-level audit found no
identical crops across splits. Initialization is the original LightOn model,
revision `{BASE_REVISION}`. The bbox fine-tune was not used, because its
training pages overlap this benchmark's validation/test pages.

Checkpoint `{Path(verification['selected_checkpoint']).name}` selected only on
**full validation CER**. Native validation CER after preserving visible quotes:
{verification['fresh_native_bf16_validation_cer']*100:.4f}%.
The historical test was also used to evaluate Surya; it did not select this run's
checkpoint or hyperparameters. Metrics use the same frozen crops and scoring
as that Surya benchmark. Scoring preserves visible quotes, case, accents and
punctuation; only surrounding and repeated whitespace is normalized. Surya's
stored raw predictions are rescored with this identical corrected protocol.
Historical quote-stripping scores are retained separately for transparency.
Raw transcription scores are also provided in `benchmark_native.json`.
This is a local-domain evaluation, not a claim about other datasets.

## Usage

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "Remidesbois/LightOnOCR-3-Poneglyph"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
)
image = Image.open("bubble.png").convert("RGB")
conversation = [{{"role": "user", "content": [{{"type": "image", "image": image}}]}}]
inputs = processor.apply_chat_template(
    conversation, tokenize=True, add_generation_prompt=True,
    enable_thinking=False, return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
text = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(text.strip())
```

Verified with Transformers 5.14.1 / PyTorch 2.8.0 CUDA 12.8 on RTX 5090.
Processor uses aspect-preserving Qwen resizing, 65,536–1,048,576 pixels.
No thinking, sampling, coordinates or geometry are included in the targets.

## Training and integrity

Full language-model fine-tuning, merger and last four vision blocks:
793,333,056 / 852,985,920 parameters trained. FP32 master weights and AdamW
states, BF16 autocast, fused AdamW, SDPA, FLA Gated DeltaNet with SM120
causal-conv1d, Liger RMSNorm/SwiGLU and assistant-only weighted CE.
Every bubble contributes a normalized assistant/EOS loss, with mild extra
weight for short text and onomatopoeia. Conservative photometric augmentation;
length grouping and separate lower learning rates for vision/merger.

Export is standard BF16 safetensors. All {verification['all_native_tensors_verified']}
native tensors equal the BF16 cast of the CER-selected FP32 checkpoint.
Final validation/test use a fresh, unpatched Transformers reload.
Weights SHA-256: `{verification['weights_sha256']}`.
Only native weights, processor, model card and aggregate metrics are published.
Training crops, references and per-example predictions remain local.
'''
    (core.FINAL_DIR / "README.md").write_text(text, encoding="utf-8")


def train(smoke_steps=0):
    global ACTIVE_MONITOR
    audit = read_audit()
    init_model = os.getenv("LIGHTON3_BUBBLE_INIT_MODEL")
    if init_model:
        previous_path = Path(init_model).parent / "run_config.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous["model_id"] != BASE_ID or previous["revision"] != BASE_REVISION or previous["prompt"] != "":
            raise RuntimeError("Consolidation requires a native bubble run, not the bbox model")
        if previous["dataset"] != audit["splits"]:
            raise RuntimeError("Consolidation split differs from the initial bubble run")
    core.configure_torch_runtime()
    core.set_seed(core.RANDOM_SEED)
    runtime = core.validate_training_runtime(True)
    processor = load_processor()
    model = load_model(training=True)
    core.configure_generation(model, processor)
    train_dataset = core.prepare_dataset(core.TRAIN_FILE, "train", processor)
    excluded = set(audit["train_ids_excluded"])
    if excluded:
        train_dataset = train_dataset.filter(lambda r: str(r["id"]) not in excluded)
    val_dataset = core.prepare_dataset(core.VAL_FILE, "val", processor)
    args = training_args(smoke_steps)
    resume = core.resolve_resume_checkpoint() if not smoke_steps else None
    config = {"model_id": BASE_ID, "revision": BASE_REVISION, "prompt": "", "enable_thinking": False,
              "train_mode": "hybrid", "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "master_weights": "float32", "autocast": "bfloat16", "seed": core.RANDOM_SEED,
              "selection": "CER on all 1648 validation crops; test never used for tuning",
              "runtime": runtime, "dataset": audit["splits"], "dataset_excluded_train_ids": sorted(excluded),
              "resume": resume, "training_args": args.to_dict(), "acceleration": model._surya_training_acceleration,
              "init_model": init_model,
              "eos_loss_weight": float(os.getenv("LIGHTON3_BUBBLE_EOS_WEIGHT", "1.0")),
              "augmentation_enabled": core.env_bool("SURYA_AUGMENT", True),
              "augmentation_probability": float(os.getenv("SURYA_AUGMENT_PROBABILITY", "0.50")),
              "native_export_only": True, "initialization_bbox_rejected_due_to_split_leakage": True}
    config_path = core.OUTPUT_DIR / "run_config.json"
    if config_path.exists() and not smoke_steps:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if not resume:
            raise RuntimeError("This output already contains a run; use a new output or explicitly resume")
        for key in ("model_id", "revision", "prompt", "dataset", "dataset_excluded_train_ids"):
            if previous[key] != config[key]:
                raise RuntimeError(f"Resume contract changed: {key}")
    atomic_json(config_path, config)
    monitor = BubbleMonitor(core.OUTPUT_DIR, config)
    ACTIVE_MONITOR = monitor
    trainer = BubbleTrainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=core.SuryaBubbleCollator(processor, training=True), callbacks=[monitor], processor=processor,
        gen_eval_max_samples=16 if smoke_steps else len(val_dataset),
        generation_early_stopping_patience=0 if smoke_steps else int(os.environ["SURYA_EARLY_STOPPING_PATIENCE"]))
    trainer.monitor = monitor
    trainer.remove_callback(PrinterCallback)
    try:
        result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        if smoke_steps:
            trainer.eval_dataset = val_dataset.select(range(16))
            trainer.evaluate()
            monitor.close("smoke_complete", trainer.state)
            return
        monitor.phase("exporting")
        verification = export_selected(trainer, processor)
        monitor.summary.update(verification=verification)
        # Fresh standard Transformers reload: inference must work without Liger
        # and without the training forward monkeypatch.
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        monitor.phase("validating")
        val_metrics, _ = benchmark(str(core.FINAL_DIR), "val", core.OUTPUT_DIR / "exported_validation.json")
        verification["fresh_native_bf16_validation_cer"] = val_metrics["cer"]
        verification["fresh_native_bf16_validation_wer"] = val_metrics["wer"]
        verification["fresh_native_bf16_reload"] = True
        verification["native_forward_modules_verified"] = val_metrics["native_forward_modules_verified"]
        atomic_json(core.OUTPUT_DIR / "model_verification.json", verification)
        monitor.summary.update({"eval_" + k: v for k, v in val_metrics.items() if isinstance(v, (int, float))})
        if os.getenv("LIGHTON3_BUBBLE_DEFER_TEST", "0") == "1":
            monitor.close("validation_complete")
            return
        monitor.phase("testing")
        metrics, samples = benchmark(str(core.FINAL_DIR), "test", core.FINAL_DIR / "benchmark_test.json")
        payload = json.loads((core.FINAL_DIR / "benchmark_test.json").read_text(encoding="utf-8"))
        monitor.summary.update({"test_" + k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        monitor.summary.update(test_metrics=metrics, dataset=audit["splits"], final_model=str(core.FINAL_DIR))
        artifacts = write_final_report(core.OUTPUT_DIR, payload, monitor.summary)
        # The figures/calculations are shared with Surya; give the new artifacts
        # their actual model identity without duplicating report calculations.
        for name in ("report.html",):
            path = core.OUTPUT_DIR / name
            if path.exists():
                path.write_text(path.read_text(encoding="utf-8").replace("Surya OCR", "LightOnOCR-3 0.8B").replace("SURYA OCR", "LIGHTONOCR-3 / BULLES"), encoding="utf-8")
        write_model_card(metrics, verification)
        monitor.summary["artifacts"] = artifacts
        monitor.close("complete")
    except BaseException as exc:
        monitor.summary["error"] = str(exc)
        monitor.close("failed")
        raise


def finalize():
    global ACTIVE_MONITOR
    # Run only after all validation-based decisions are finished.
    core.configure_torch_runtime()
    verification_path = core.OUTPUT_DIR / "model_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    config = json.loads((core.OUTPUT_DIR / "run_config.json").read_text(encoding="utf-8"))
    previous_summary = json.loads((core.OUTPUT_DIR / "training_summary.json").read_text(encoding="utf-8"))
    if not verification["selected_checkpoint_bf16_cast_exact"] or not verification["fresh_native_bf16_reload"]:
        raise RuntimeError("Native checkpoint verification is incomplete")
    monitor = BubbleMonitor(core.OUTPUT_DIR, {**config, "resume": True})
    ACTIVE_MONITOR = monitor
    monitor.summary.update(previous_summary)
    monitor.phase("testing")
    try:
        metrics, _ = benchmark(str(core.FINAL_DIR), "test", core.FINAL_DIR / "benchmark_test.json")
        payload = json.loads((core.FINAL_DIR / "benchmark_test.json").read_text(encoding="utf-8"))
        monitor.summary.update({"test_" + k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        monitor.summary.update(test_metrics=metrics, dataset=config["dataset"], final_model=str(core.FINAL_DIR))
        monitor.summary["artifacts"] = write_final_report(core.OUTPUT_DIR, payload, monitor.summary)
        path = core.OUTPUT_DIR / "report.html"
        path.write_text(path.read_text(encoding="utf-8").replace("Surya OCR", "LightOnOCR-3 0.8B").replace("SURYA OCR", "LIGHTONOCR-3 / BULLES"), encoding="utf-8")
        write_model_card(metrics, verification)
        monitor.close("complete")
    except BaseException as exc:
        monitor.summary["error"] = str(exc)
        monitor.close("failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "baseline", "train", "test", "finalize", "verify-native"), default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "calibrate":
        calibrate(args.batch_size)
    elif args.mode == "baseline":
        core.configure_torch_runtime()
        core.set_seed(core.RANDOM_SEED)
        core.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_dashboard(core.OUTPUT_DIR / "training_dashboard.html", {"status": "baseline"}, [])
        metrics, _ = benchmark(BASE_ID, "val", core.OUTPUT_DIR / "baseline_validation.json")
        summary = {"status": "preparing", **{"baseline_" + k: v for k, v in metrics.items() if isinstance(v, (int, float))}}
        write_dashboard(core.OUTPUT_DIR / "training_dashboard.html", summary, [])
    elif args.mode == "test":
        core.configure_torch_runtime()
        benchmark(str(core.FINAL_DIR), "test", core.FINAL_DIR / "benchmark_test.json")
    elif args.mode == "finalize":
        finalize()
    elif args.mode == "verify-native":
        core.configure_torch_runtime()
        metrics, _ = benchmark(str(core.FINAL_DIR), "val", core.OUTPUT_DIR / "exported_validation_native.json")
        path = core.OUTPUT_DIR / "model_verification.json"
        verification = json.loads(path.read_text(encoding="utf-8"))
        verification.update(fresh_native_bf16_validation_cer=metrics["cer"], fresh_native_bf16_validation_wer=metrics["wer"],
                            fresh_native_bf16_reload=True, native_forward_modules_verified=metrics["native_forward_modules_verified"],
                            separate_process_native_validation=True)
        atomic_json(path, verification)
        summary_path = core.OUTPUT_DIR / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(status="validation_complete", verification=verification,
                       **{"eval_" + k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        atomic_json(summary_path, summary)
        history_path = core.OUTPUT_DIR / "metrics.jsonl"
        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if json.loads(line).get("event") == "metrics"]
        write_dashboard(core.OUTPUT_DIR / "training_dashboard.html", summary, history)
    else:
        train(args.smoke_steps)
