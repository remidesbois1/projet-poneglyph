"""RTX 5090 SFT of LightOnOCR-3-0.8B on complete Poneglyph manga pages."""

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from datasets import Dataset
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from transformers import AutoProcessor, EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, Qwen3_5ForConditionalGeneration
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_utils import get_last_checkpoint

from grounding import PROMPT, aggregate, page_metrics, parse_grounding
from monitor import Monitor, atomic_json
from training_acceleration import install_acceleration, packed_targets
from loss_weighting import token_weights

ROOT = Path(__file__).resolve().parent
MODEL_ID = "lightonai/LightOnOCR-3-0.8B"
REVISION = "36d8636925327c671bd8e479c517a004f7566bd9"
SEED = 42


def messages(reference=None):
    result = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    if reference is not None:
        result.append({"role": "assistant", "content": [{"type": "text", "text": reference}]})
    return result


def render(processor, reference=None):
    return processor.apply_chat_template(messages(reference), tokenize=False, add_generation_prompt=reference is None, enable_thinking=False)


def image_path(dataset_dir, row):
    path = dataset_dir / row["split"] / row["image_file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class Collator:
    def __init__(self, processor, dataset_dir, augment=True, cache_items=64, bbox_loss_weight=1.0, eos_loss_weight=1.0):
        self.processor, self.dataset_dir, self.augment = processor, dataset_dir, augment
        self.cache_items, self.cache = cache_items, OrderedDict()
        self.bbox_loss_weight, self.eos_loss_weight = bbox_loss_weight, eos_loss_weight

    def image(self, row, augment=False):
        path = image_path(self.dataset_dir, row)
        key = str(path)
        if key not in self.cache:
            with Image.open(path) as image:
                value = image.convert("RGB")
                value.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            self.cache[key] = value
            while len(self.cache) > self.cache_items:
                _, old = self.cache.popitem(last=False)
                old.close()
        self.cache.move_to_end(key)
        image = self.cache[key].copy()
        # Only photometric augmentation: geometry and bbox labels stay exact.
        if augment and random.random() < .30:
            choice = random.randrange(3)
            if choice == 0:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(.85, 1.15))
            elif choice == 1:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(.9, 1.1))
            else:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(.1, .35)))
        return image

    def __call__(self, rows):
        texts = [render(self.processor, row["text"]) for row in rows]
        prompt = render(self.processor)
        images = [self.image(row, self.augment and row["split"] == "train") for row in rows]
        try:
            batch = self.processor(text=texts, images=images, padding=True, pad_to_multiple_of=16, return_tensors="pt")
        finally:
            for image in images:
                image.close()
        weighted = self.bbox_loss_weight != 1 or self.eos_loss_weight != 1
        tokenization = self.processor.tokenizer(texts + [prompt], add_special_tokens=False, return_offsets_mapping=weighted)
        tokenized = tokenization["input_ids"]
        prefix = tokenized[-1]
        labels = batch["input_ids"].clone()
        weights = torch.ones_like(labels, dtype=torch.float32) if weighted else None
        for i, ids in enumerate(tokenized[:-1]):
            if ids[:len(prefix)] != prefix:
                raise ValueError(f"Assistant template prefix mismatch on page {rows[i]['page_id']}")
            target = ids[len(prefix):]
            end = labels.shape[1]  # Qwen generation uses left padding.
            start = end - len(target)
            if not target or labels[i, start:end].tolist() != target:
                raise ValueError(f"Assistant suffix mismatch on page {rows[i]['page_id']}")
            labels[i, :start] = -100
            if weighted:
                if not texts[i].startswith(prompt):
                    raise ValueError("Character prompt prefix mismatch for OCR weighting")
                values = token_weights(rows[i]["text"], tokenization["offset_mapping"][i][len(prefix):], len(prompt),
                                       target, self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                                       self.bbox_loss_weight, self.eos_loss_weight)
                weights[i, start:end] = torch.tensor(values)
        labels.masked_fill_(batch["attention_mask"].eq(0), -100)
        batch["labels"] = labels
        if weights is not None:
            batch["ocr_token_weights"] = weights
        return packed_targets(batch)


def to_device(batch):
    return {key: value.to("cuda", non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def load_datasets(dataset_dir, processor, max_new_tokens):
    result, manifest = {}, {}
    seen, seen_images = {}, {}
    for split in ("train", "val", "test"):
        path = dataset_dir / split / "metadata.jsonl"
        content = path.read_bytes()
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"Empty split {split}")
        for row in rows:
            page_id = str(row["page_id"])
            if page_id in seen:
                raise ValueError(f"Duplicate or leaking page {page_id}")
            seen[page_id] = split
            digest = row["image_sha256"]
            if digest in seen_images and seen_images[digest] != split:
                raise ValueError("Image-content split leakage")
            seen_images[digest] = split
            row["split"] = split
            with Image.open(image_path(dataset_dir, row)) as image:
                actual_digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
                if actual_digest != digest or list(image.size) != row["image_size"]:
                    raise ValueError(f"Image content changed since export on page {page_id}")
            full_ids = processor.tokenizer(render(processor, row["text"]), add_special_tokens=False)["input_ids"]
            prefix_ids = processor.tokenizer(render(processor), add_special_tokens=False)["input_ids"]
            target_length = len(full_ids) - len(prefix_ids)
            if target_length > max_new_tokens or target_length <= 0:
                raise ValueError(f"Page {page_id} target requires {target_length} tokens; budget={max_new_tokens}")
            width, height = row["image_size"]
            ratio = min(1, 2048 / max(width, height))
            row["length"] = len(full_ids) + math.ceil(width * ratio / 32) * math.ceil(height * ratio / 32)
            row["target_length"] = target_length
        result[split] = Dataset.from_list(rows)
        manifest[split] = {"pages": len(rows), "bubbles": sum(r["num_bubbles"] for r in rows),
                           "metadata_sha256": hashlib.sha256(content).hexdigest(),
                           "target_tokens_p95": float(np.percentile([r["target_length"] for r in rows], 95)),
                           "target_tokens_max": max(r["target_length"] for r in rows)}
    return result, manifest


def subset(dataset, maximum):
    if maximum <= 0 or len(dataset) <= maximum:
        return dataset
    # Cover sparse/dense pages on an identical subset at every evaluation.
    ordered = sorted(range(len(dataset)), key=lambda i: (dataset[i]["num_bubbles"], dataset[i]["target_length"]))
    indices = [ordered[int(i)] for i in np.linspace(0, len(ordered) - 1, maximum)]
    return dataset.select(sorted(indices))


def visualize_predictions(rows, dataset_dir, output_dir, limit=8):
    output_dir.mkdir(parents=True, exist_ok=True)
    worst = sorted(rows, key=lambda row: row["metrics"]["f1@0_5"])[:limit]
    for row in worst:
        with Image.open(image_path(dataset_dir, row)) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for text, color in ((row["reference"], "#00b982"), (row["prediction"], "#ff503e")):
            items, _ = parse_grounding(text)
            for i, item in enumerate(items):
                box = [v * image.width / 1000 if j % 2 == 0 else v * image.height / 1000 for j, v in enumerate(item["bbox"])]
                draw.rectangle(box, outline=color, width=3)
                draw.text((box[0] + 2, box[1] + 2), str(i + 1), fill=color, stroke_width=1, stroke_fill="black")
        image.save(output_dir / f"page_{row['page_id']}.jpg", quality=90)
        image.close()


def benchmark(model, processor, dataset, collator, output_path, batch_size=4, max_new_tokens=2048, bootstrap=False):
    previous_training = model.training
    configs = [model.config, model.config.text_config]
    caches = [getattr(config, "use_cache", None) for config in configs]
    for config in configs:
        config.use_cache = True
    model.eval()
    rows, offset, started = [], 0, time.perf_counter()
    dataset = dataset.sort("length")
    try:
        while offset < len(dataset):
            entries = [dataset[i] for i in range(offset, min(offset + batch_size, len(dataset)))]
            images = [collator.image(row) for row in entries]
            try:
                inputs = processor(text=[render(processor)] * len(entries), images=images, padding=True, pad_to_multiple_of=16, return_tensors="pt")
            finally:
                for image in images:
                    image.close()
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model.generate(**to_device(inputs), use_cache=True, do_sample=False, max_new_tokens=max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                print(f"Generation memory fallback: batch={batch_size}", flush=True)
                continue
            generated = outputs[:, inputs["input_ids"].shape[1]:].detach().cpu()
            texts = processor.batch_decode(generated, skip_special_tokens=True)
            eos_ids = model.generation_config.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
            for row, prediction, tokens in zip(entries, texts, generated.tolist()):
                stopped = any(token in eos_ids for token in tokens)
                metrics = page_metrics(row["text"], prediction.strip(), truncated=not stopped and len(tokens) >= max_new_tokens)
                rows.append({"page_id": row["page_id"], "image_file": row["image_file"], "split": row["split"],
                             "reference": row["text"], "prediction": prediction.strip(), "metrics": metrics})
            offset += len(entries)
            print(f"Grounding evaluation {offset}/{len(dataset)} pages ({time.perf_counter()-started:.1f}s)", flush=True)
        metrics = aggregate([row["metrics"] for row in rows])
        metrics.update(pages_per_second=len(rows) / (time.perf_counter() - started), generation_batch_size=batch_size)
        if bootstrap:
            rng = np.random.default_rng(SEED)
            estimates = [aggregate([rows[int(i)]["metrics"] for i in rng.integers(len(rows), size=len(rows))]) for _ in range(1000)]
            intervals = {key: [float(np.percentile([r[key] for r in estimates], p)) for p in (2.5, 97.5)]
                         for key in ("cer", "wer", "exact_bubble_text_rate", "page_text_exact", "f1@0_5", "f1@0_75", "mean_iou")}
        else:
            intervals = None
        atomic_json(output_path, {"model_id": MODEL_ID, "revision": REVISION, "prompt": PROMPT, "metrics": metrics, "bootstrap_ci95": intervals, "pages": rows})
        visualize_predictions(rows, collator.dataset_dir, output_path.parent / f"{output_path.stem}_overlays")
        print(json.dumps(metrics, indent=2), flush=True)
        return metrics
    finally:
        for config, cache in zip(configs, caches):
            config.use_cache = cache
        model.train(previous_training)


class GroundingTrainer(Seq2SeqTrainer):
    def __init__(self, *args, processor, generation_dataset, generation_collator, monitor, generation_batch, max_new_tokens, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor, self.generation_dataset, self.generation_collator, self.monitor = processor, generation_dataset, generation_collator, monitor
        self.generation_batch, self.max_new_tokens = generation_batch, max_new_tokens
        self.model_accepts_loss_kwargs = True
        self._loss_shifts_labels = True

    def _get_num_items_in_batch(self, batch_samples, device):
        # Exact assistant-token normalization across the whole accumulation window.
        return torch.tensor(sum(float(batch["ocr_weights"].sum()) if "ocr_weights" in batch else batch["ocr_targets"].numel()
                                for batch in batch_samples), device=device) if batch_samples else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        inputs.pop("labels", None)
        count = (inputs["ocr_weights"].sum() if "ocr_weights" in inputs else inputs["ocr_targets"].numel()) if num_items_in_batch is None else num_items_in_batch
        outputs = model(**inputs, use_cache=False)
        loss = outputs.loss / count
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self, model=None):
        if self.optimizer is not None:
            return self.optimizer
        target = self.model if model is None else model
        decay = set(self.get_decay_parameter_names(target))
        groups = {}
        for name, parameter in target.named_parameters():
            if not parameter.requires_grad:
                continue
            factor = .5 if ".visual.merger." in name else .25 if ".visual." in name else 1.0
            weight_decay = self.args.weight_decay if name in decay else 0.0
            groups.setdefault((factor, weight_decay), []).append(parameter)
        self.optimizer = torch.optim.AdamW([{"params": params, "lr": self.args.learning_rate * factor, "weight_decay": wd} for (factor, wd), params in groups.items()], lr=self.args.learning_rate, fused=True)
        return self.optimizer

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        self.monitor.phase("validating")
        output = super().evaluation_loop(dataloader, description, prediction_loss_only=True, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        metrics = benchmark(self.model, self.processor, self.generation_dataset, self.generation_collator,
                            Path(self.args.output_dir) / f"validation_step_{self.state.global_step}.json", self.generation_batch, self.max_new_tokens)
        output.metrics.update({f"{metric_key_prefix}_{key}": value for key, value in metrics.items()})
        return output


def calibrate(model, processor, dataset, collator, output_dir, effective_batch, maximum_batch=8):
    """Measure real forward/backward/optimizer peaks without changing weights."""
    hard = sorted(range(len(dataset)), key=lambda i: dataset[i]["length"], reverse=True)[:effective_batch]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.0, weight_decay=0.0, fused=True)
    initial_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())
    observations, candidates, best = [], [b for b in (1, 2, 4, 8) if b <= min(effective_batch, maximum_batch)], None
    total_memory = torch.cuda.get_device_properties(0).total_memory
    try:
        for checkpointing in (False, True):
            model.gradient_checkpointing_disable()
            if checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            for batch_size in candidates:
                if observations and observations[-1].get("checkpointing") == checkpointing and not observations[-1].get("oom"):
                    previous = observations[-1]
                    # Account for optimizer residency plus measured activation growth.
                    resident = torch.cuda.memory_allocated()
                    # Gradients, like weights/Adam states, are batch-independent.
                    fixed = resident + sum(p.numel() * p.element_size() for p in parameters)
                    predicted = fixed + max(0, previous["peak_bytes"] - fixed) * batch_size / previous["batch"]
                    if predicted > .91 * total_memory:
                        break
                features = [dataset[i] for i in hard[:batch_size]]
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                batch = None
                try:
                    batch = to_device(collator(features))
                    batch.pop("labels")
                    # The second forward must retain the full FP32 gradients:
                    # accumulated microbatches carry them through their forward.
                    # set_to_none=True here underestimates the real peak by ~3 GiB.
                    for repetition in range(2):
                        model.zero_grad(set_to_none=False)
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = model(**batch).loss / (batch["ocr_weights"].sum() if "ocr_weights" in batch else batch["ocr_targets"].numel())
                        loss.backward()
                        optimizer.step()  # lr=0: weights remain exactly unchanged.
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - started
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite calibration loss")
                    peak = torch.cuda.max_memory_allocated()
                    free, _ = torch.cuda.mem_get_info()
                    observation = {"batch": batch_size, "checkpointing": checkpointing, "peak_bytes": peak,
                                   "peak_gib": peak / 2**30, "free_gib": free / 2**30,
                                   "seconds_per_microbatch": seconds, "pages_per_second": batch_size / seconds,
                                   "optimizer_states_fp32": all(state["exp_avg"].dtype == torch.float32 for state in optimizer.state.values())}
                    observations.append(observation)
                    print("Calibration: " + json.dumps(observation), flush=True)
                    if peak < .91 * total_memory and free > 1.25 * 2**30:
                        if best is None or observation["pages_per_second"] > best["pages_per_second"]:
                            best = observation
                    else:
                        break
                except torch.cuda.OutOfMemoryError:
                    observations.append({"batch": batch_size, "checkpointing": checkpointing, "oom": True})
                    print(f"Calibration OOM at batch={batch_size}, checkpointing={checkpointing}", flush=True)
                    break
                finally:
                    model.zero_grad(set_to_none=True)
                    del batch
                    gc.collect()
                    torch.cuda.empty_cache()
            if best is not None and not checkpointing:
                # Native backprop avoids recomputing thousands of visual tokens.
                break
        if best is None:
            raise RuntimeError("No safe full-page batch fits on this GPU")
    finally:
        del optimizer
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        random.setstate(initial_rng[0])
        np.random.set_state(initial_rng[1])
        torch.set_rng_state(initial_rng[2])
        torch.cuda.set_rng_state(initial_rng[3])
    model.gradient_checkpointing_disable()
    if best["checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    atomic_json(output_dir / "hardware_profile.json", {"selected": best, "effective_batch": effective_batch, "observations": observations})
    return best


def configure_model(mode, init_model=None):
    source = str(init_model) if init_model else MODEL_ID
    source_options = {"local_files_only": True} if init_model else {"revision": REVISION}
    model = Qwen3_5ForConditionalGeneration.from_pretrained(source, **source_options, dtype=torch.float32, attn_implementation="sdpa").cuda()
    if mode == "hybrid":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module in (model.model.language_model, model.lm_head, model.model.visual.merger, *model.model.visual.blocks[-4:]):
            for parameter in module.parameters():
                parameter.requires_grad = True
    model.config.use_cache = model.config.text_config.use_cache = False
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.train()
    return model


def healthy_checkpoint(output_dir):
    candidates = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1, reverse=True)
    for path in candidates:
        if not path.is_dir():
            continue
        required = [path / "trainer_state.json", path / "optimizer.pt", path / "scheduler.pt", path / "rng_state.pth"]
        weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
        if all(p.exists() and p.stat().st_size > 0 for p in required) and weights:
            try:
                json.loads(required[0].read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            return str(path)
    if candidates:
        raise RuntimeError("Checkpoint directories exist but none has complete weights/optimizer/scheduler/RNG state")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "rtx5090-20261011")
    parser.add_argument("--train-mode", choices=("hybrid", "full"), default="hybrid")
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--lr", type=float, default=1.2e-5)
    parser.add_argument("--effective-batch", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--batch", type=int, default=0, help="0 calibrates the physical batch")
    parser.add_argument("--generation-batch", type=int, default=4)
    parser.add_argument("--generation-pages", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--recalibrate", action="store_true", help="Re-measure hardware on resume; preserve effective batch, optimizer and schedule")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--init-model", type=Path, help="New training stage from native weights; fresh optimizer/schedule")
    parser.add_argument("--selection-metric", choices=("combined_score", "cer"), default="combined_score")
    parser.add_argument("--bbox-loss-weight", type=float, default=1.0)
    parser.add_argument("--eos-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-physical-batch", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--no-augmentation", action="store_true")
    args = parser.parse_args()
    if not 0 < args.bbox_loss_weight <= 1 or args.eos_loss_weight <= 0:
        raise ValueError("Invalid loss weights")
    if args.init_model and not args.init_model.is_dir():
        raise FileNotFoundError(args.init_model)
    if args.batch and (args.batch > args.effective_batch or args.effective_batch % args.batch):
        raise ValueError("Physical batch must divide effective batch")
    if args.smoke_steps:
        args.output = args.output / "smoke"
    args.output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA/BF16 required; refusing CPU training")
    from transformers.models.qwen3_5 import modeling_qwen3_5
    if not modeling_qwen3_5.is_fast_path_available:
        raise RuntimeError("Install SM120 causal-conv1d and flash-linear-attention before training")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    processor.tokenizer.padding_side = "left"
    size = processor.image_processor.size
    if isinstance(size, dict):
        size.update(shortest_edge=65536, longest_edge=2048 * 2048)
    else:
        size.shortest_edge, size.longest_edge = 65536, 2048 * 2048
    datasets, manifest = load_datasets(args.dataset, processor, args.max_new_tokens)
    config = {"model_id": MODEL_ID, "revision": REVISION, "prompt": PROMPT, "thinking": False, "train_mode": args.train_mode,
              "epochs": args.epochs, "lr": args.lr, "effective_batch": args.effective_batch, "dataset": manifest,
              "gpu": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability()),
              "torch": torch.__version__, "cuda": torch.version.cuda, "master_weights_dtype": "float32", "autocast_dtype": "bfloat16"}
    config.update(init_model=str(args.init_model) if args.init_model else None, selection_metric=args.selection_metric,
                  bbox_loss_weight=args.bbox_loss_weight, eos_loss_weight=args.eos_loss_weight, augmentation=not args.no_augmentation,
                  generation_pages=args.generation_pages, max_new_tokens=args.max_new_tokens)
    if args.init_model:
        config["init_weight_files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.init_model.glob("*.safetensors"))}
        config["test_previously_evaluated"] = True
    old_config_path = args.output / "run_config.json"
    checkpoint = healthy_checkpoint(args.output) if args.resume == "auto" else None if args.resume == "none" else args.resume
    previous = None
    if checkpoint and old_config_path.exists():
        previous = json.loads(old_config_path.read_text(encoding="utf-8"))
        for key in ("model_id", "revision", "train_mode", "dataset", "effective_batch", "epochs", "lr"):
            if previous.get(key) != config.get(key):
                raise ValueError(f"Resume configuration changed: {key}")
        for key, default in (("init_model", None), ("selection_metric", "combined_score"), ("bbox_loss_weight", 1.0),
                             ("eos_loss_weight", 1.0), ("augmentation", True), ("generation_pages", 48), ("max_new_tokens", 1280)):
            if previous.get(key, default) != config[key]:
                raise ValueError(f"Resume stage configuration changed: {key}")
    if not checkpoint and (args.output / "trainer_state.json").exists() and args.resume != "none":
        raise RuntimeError("Completed/existing output has no resumable checkpoint; choose a new output")
    if checkpoint:
        config["resume_checkpoint"] = str(checkpoint)
        config["resume_step"] = json.loads((Path(checkpoint) / "trainer_state.json").read_text(encoding="utf-8"))["global_step"]
        if previous:
            config["hardware_before_resume"] = previous.get("hardware")
    atomic_json(old_config_path, config)
    monitor = Monitor(args.output, config)
    try:
        monitor.phase("loading_model")
        model = configure_model(args.train_mode, args.init_model)
        config["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        config["total_parameters"] = sum(p.numel() for p in model.parameters())
        config["acceleration"] = install_acceleration(model)
        collator = Collator(processor, args.dataset, augment=not args.no_augmentation, bbox_loss_weight=args.bbox_loss_weight, eos_loss_weight=args.eos_loss_weight)
        eval_collator = Collator(processor, args.dataset, augment=False, cache_items=16, bbox_loss_weight=args.bbox_loss_weight, eos_loss_weight=args.eos_loss_weight)
        if checkpoint and previous and previous.get("hardware") and not args.recalibrate:
            selected = previous["hardware"]
            if args.batch and args.batch != selected["batch"]:
                raise ValueError("Preserve the physical batch when resuming an exact checkpoint")
            model.gradient_checkpointing_disable()
            if selected["checkpointing"]:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        elif args.batch:
            selected = {"batch": args.batch, "checkpointing": False}
        else:
            monitor.phase("calibrating")
            selected = calibrate(model, processor, datasets["train"], eval_collator, args.output, args.effective_batch, args.max_physical_batch)
        config["hardware"] = selected
        config["gradient_accumulation"] = args.effective_batch // selected["batch"]
        atomic_json(old_config_path, config)
        if args.calibrate_only:
            monitor.close("calibrated")
            return
        generation_dataset = subset(datasets["val"], 2 if args.smoke_steps else args.generation_pages)
        baseline_path = args.output / "baseline_validation.json"
        if args.init_model and args.selection_metric == "cer" and args.skip_baseline and not args.smoke_steps:
            raise ValueError("A CER refinement must evaluate its starting model")
        if not args.skip_baseline and not checkpoint and not baseline_path.exists():
            monitor.phase("baseline")
            baseline = benchmark(model, processor, generation_dataset, eval_collator, baseline_path, args.generation_batch, args.max_new_tokens)
            monitor.summary.update({f"baseline_{key}": value for key, value in baseline.items()})
        steps = math.ceil(math.ceil(len(datasets["train"]) / selected["batch"]) / config["gradient_accumulation"]) * args.epochs
        fields = Seq2SeqTrainingArguments.__dataclass_fields__
        kwargs = dict(output_dir=str(args.output), per_device_train_batch_size=selected["batch"], per_device_eval_batch_size=selected["batch"],
                      gradient_accumulation_steps=config["gradient_accumulation"], learning_rate=args.lr, num_train_epochs=args.epochs,
                      max_steps=args.smoke_steps or -1, bf16=True, bf16_full_eval=False, tf32=True,
                      gradient_checkpointing=selected["checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
                      optim="adamw_torch_fused", weight_decay=.05, max_grad_norm=1.0,
                      warmup_steps=max(1, round(steps * .05)) if not args.smoke_steps else 0,
                      lr_scheduler_type="cosine_with_min_lr", lr_scheduler_kwargs={"min_lr_rate": .1},
                      logging_steps=5, logging_first_step=True, logging_nan_inf_filter=False, disable_tqdm=True,
                      eval_strategy="no" if args.smoke_steps else "epoch", save_strategy="no" if args.smoke_steps else "epoch",
                      save_total_limit=2, save_only_model=False, load_best_model_at_end=not bool(args.smoke_steps),
                      metric_for_best_model=f"eval_{args.selection_metric}", greater_is_better=args.selection_metric != "cer",
                      dataloader_num_workers=0, dataloader_pin_memory=True, remove_unused_columns=False,
                      label_names=["labels"], prediction_loss_only=True, report_to=["tensorboard"],
                      seed=SEED, data_seed=SEED, include_num_input_tokens_seen="all", torch_compile=False,
                      accelerator_config={"non_blocking": True}, length_column_name="length")
        kwargs["train_sampling_strategy" if "train_sampling_strategy" in fields else "group_by_length"] = "group_by_length" if "train_sampling_strategy" in fields else True
        training_args = Seq2SeqTrainingArguments(**kwargs)
        config["training_args"] = training_args.to_dict()
        config["training_args"].pop("hub_token", None)
        atomic_json(old_config_path, config)
        trainer = GroundingTrainer(model=model, args=training_args, train_dataset=datasets["train"], eval_dataset=datasets["val"],
                                  data_collator=collator, processing_class=processor, processor=processor,
                                  generation_dataset=generation_dataset, generation_collator=eval_collator, monitor=monitor,
                                  generation_batch=args.generation_batch, max_new_tokens=args.max_new_tokens,
                                  callbacks=[monitor] + ([] if args.smoke_steps else [EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=.0002 if args.selection_metric == "cer" else .001)]))
        trainer.remove_callback(PrinterCallback)
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        if args.smoke_steps:
            metrics = trainer.evaluate(eval_dataset=subset(datasets["val"], 2))
            monitor.summary.update(metrics)
            monitor.close("smoke_complete", trainer.state)
            return
        monitor.phase("exporting", best_model_checkpoint=trainer.state.best_model_checkpoint, best_metric=trainer.state.best_metric)
        final_dir = args.output / "final_merged"
        # Release FP32 optimizer states before exporting the best weights as BF16.
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        if args.init_model and args.selection_metric == "cer":
            baseline_cer = json.loads(baseline_path.read_text(encoding="utf-8"))["metrics"]["cer"]
            if monitor.summary["best_metric"] >= baseline_cer - .0002:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model = configure_model(args.train_mode, args.init_model)
                monitor.summary.update(best_model_checkpoint=str(args.init_model), best_metric=baseline_cer,
                                       retained_initial_model=True)
            else:
                monitor.summary["retained_initial_model"] = False
        model.to(torch.bfloat16)
        model.config.use_cache = model.config.text_config.use_cache = True
        model.save_pretrained(final_dir, safe_serialization=True, max_shard_size="4GB")
        processor.save_pretrained(final_dir)
        (final_dir / "README.md").write_text(f"---\nbase_model: {MODEL_ID}\nlicense: apache-2.0\npipeline_tag: image-text-to-text\n---\n\n# Poneglyph LightOnOCR-3 0.8B grounding\n\nPrompt: `grounding`; `enable_thinking=False`. Full-page manga text zones, native `![text](x1,y1,x2,y2)` output, normalized 0–1000.\n\nBase revision: `{REVISION}`. See parent run_config.json and test_benchmark.json for validation.\n", encoding="utf-8")
        monitor.phase("testing")
        metrics = benchmark(model, processor, datasets["test"], eval_collator, args.output / "test_benchmark.json", args.generation_batch, args.max_new_tokens, bootstrap=True)
        monitor.summary.update({f"test_{key}": value for key, value in metrics.items()})
        atomic_json(args.output / "completion.json", {"status": "complete", "final_model": str(final_dir), "test": metrics})
        monitor.close("complete")
        from report import write_report
        write_report(args.output)
    except BaseException as error:
        atomic_json(args.output / "failure.json", {"type": type(error).__name__, "error": str(error)})
        monitor.summary["error"] = str(error)
        monitor.close("failed")
        raise


if __name__ == "__main__":
    main()
