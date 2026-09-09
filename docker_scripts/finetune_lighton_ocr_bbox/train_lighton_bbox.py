import argparse
import gc
import glob
import importlib.metadata
import inspect
import json
import math
import multiprocessing
import os
import random
import re
import sys
import time
from pathlib import Path

# Avoid tokenizer thread pools being inherited by DataLoader workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jiwer
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from Levenshtein import distance as levenshtein_distance
from PIL import Image
from transformers import (
    EarlyStoppingCallback,
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_utils import get_last_checkpoint

from training_monitor import TrainingMonitorCallback, atomic_json


SCRIPT_DIR = Path(__file__).resolve().parent
DOCKER_SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DOCKER_SCRIPTS_DIR.parent
sys.path.insert(0, str(DOCKER_SCRIPTS_DIR))
from common_training.prompts import get_prompt

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(DOCKER_SCRIPTS_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

MODEL_ID = os.getenv("LIGHTON_BBOX_MODEL_ID", "lightonai/LightOnOCR-2-1B-bbox-base")
DATASET_DIR = Path(
    os.getenv("LIGHTON_BBOX_DATASET_DIR", str(SCRIPT_DIR / "lighton_bbox_dataset"))
)
OUTPUT_DIR = Path(
    os.getenv("LIGHTON_BBOX_OUTPUT_DIR", str(SCRIPT_DIR / "outputs_lighton_bbox"))
)
FINAL_DIR = OUTPUT_DIR / "final_merged"
TRAIN_FILE = DATASET_DIR / "train" / "metadata.jsonl"
VAL_FILE = DATASET_DIR / "val" / "metadata.jsonl"
TEST_FILE = DATASET_DIR / "test" / "metadata.jsonl"
SPLITS = ("train", "val", "test")

# Keep the canonical bbox contract as the source of truth for validation and
# model-card generation. LightOnOCR's bbox base model (and Poneglyph's LightOn
# bbox runtime) is intentionally conditioned with the image only, so unlike
# Surya we must not inject this text into the conversation at train time.
OUTPUT_CONTRACT = get_prompt("ocr_page_bbox_training_lines", "LIGHTON_BBOX_USER_PROMPT")
BBOX_NORM_SCALE = int(os.getenv("LIGHTON_BBOX_NORM_SCALE", "1000"))
MAX_NEW_TOKENS = int(os.getenv("LIGHTON_BBOX_MAX_NEW_TOKENS", "1280"))
IMAGE_LONGEST_EDGE = int(os.getenv("LIGHTON_BBOX_IMAGE_LONGEST_EDGE", "1500"))
GEN_EVAL_MAX_SAMPLES = int(os.getenv("LIGHTON_BBOX_GEN_EVAL_MAX_SAMPLES", "16"))
LOSS_EVAL_MAX_SAMPLES = int(os.getenv("LIGHTON_BBOX_LOSS_EVAL_MAX_SAMPLES", "64"))
FINAL_TEST_MAX_SAMPLES = int(os.getenv("LIGHTON_BBOX_FINAL_TEST_MAX_SAMPLES", "0"))
RANDOM_SEED = int(os.getenv("LIGHTON_BBOX_RANDOM_SEED", "42"))
IOU_THRESHOLDS = (0.3, 0.5, 0.75, 0.9)
BBOX_PATTERN = re.compile(r"(.+?)\s*\[(\d+),(\d+),(\d+),(\d+)\]")
INITIAL_ENV_KEYS = frozenset(os.environ)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LightOnOCR 2 for Poneglyph full-page text+bbox extraction."
    )
    parser.add_argument("--merge-only", action="store_true", help="Merge the best LoRA checkpoint and exit.")
    parser.add_argument("--benchmark-only", action="store_true", help="Benchmark an existing merged model.")
    parser.add_argument("--model-path", default=None, help="Model path for merge-only or benchmark-only.")
    parser.add_argument("--profile", choices=("auto", "rtx5090", "safe"),
                        default=os.getenv("LIGHTON_BBOX_PROFILE", "auto"))
    parser.add_argument("--diagnose", action="store_true", help="Check the runtime without loading model weights or data.")
    parser.add_argument("--resume", nargs="?", const="auto",
                        default=os.getenv("LIGHTON_BBOX_RESUME_FROM_CHECKPOINT", "auto"),
                        help="Resume the latest checkpoint (auto), a path, or start fresh (none).")
    parser.add_argument("--smoke-steps", type=int, default=0,
                        help="Run N optimizer steps in a separate smoke directory; no merge, benchmark or upload.")
    return parser.parse_args()


def configure_torch_runtime(profile="auto"):
    if profile == "auto":
        profile = "rtx5090" if torch.cuda.is_available() and "5090" in torch.cuda.get_device_name(0) else "safe"
    defaults = {
        # Start aggressively on a 5090, then the calibration pass can move the
        # physical batch up or down based on real page memory/throughput.
        "TRAIN_BATCH": "8" if profile == "rtx5090" else "1",
        "EVAL_BATCH": "4" if profile == "rtx5090" else "1",
        "GRAD_ACCUM": "1" if profile == "rtx5090" else "8",
        "GEN_BATCH": "4" if profile == "rtx5090" else "1",
        "DATALOADER_WORKERS": "2" if profile == "rtx5090" else "0",
        "PREFETCH_FACTOR": "1" if profile == "rtx5090" else "2",
        "PERSISTENT_WORKERS": "0",
        "AUTO_BATCH": "1" if profile == "rtx5090" else "0",
        "BATCH_CANDIDATES": "1,2,4,8" if profile == "rtx5090" else "1",
        "CALIBRATION_MAX_VRAM_RATIO": "0.90" if profile == "rtx5090" else "0.85",
        "EFFECTIVE_BATCH": "8" if profile == "rtx5090" else "8",
    }
    for key, value in defaults.items():
        os.environ.setdefault(f"LIGHTON_BBOX_{key}", value)
    torch.set_num_threads(int(os.getenv("LIGHTON_BBOX_TORCH_THREADS", "8")))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    # Compilation is opt-in; never silently hide compiler failures.
    if int(os.getenv("LIGHTON_BBOX_DATALOADER_WORKERS", "0")) > 0:
        multiprocessing.set_start_method("spawn", force=True)
    return profile


def runtime_diagnostics(profile, require_cuda=False):
    packages = ("torch", "transformers", "peft", "accelerate")
    report = {"profile": profile, "cuda_runtime": torch.version.cuda, "packages": {}}
    for name in packages:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        capability_string = f"{capability[0]}.{capability[1]}"
        report.update(gpu=torch.cuda.get_device_name(0), compute_capability=capability_string,
                      vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                      bf16_supported=torch.cuda.is_bf16_supported())
        if capability[0] >= 12 and tuple(map(int, (torch.version.cuda or "0.0").split(".")[:2])) < (12, 8):
            raise RuntimeError("Blackwell requires a compatible PyTorch CUDA build (CUDA 12.8 or newer).")
    elif require_cuda and not env_bool("LIGHTON_BBOX_ALLOW_CPU_TRAINING", False):
        raise RuntimeError("CUDA is unavailable. Refusing accidental CPU training; use --diagnose to inspect the runtime.")
    print(json.dumps(report, indent=2), flush=True)
    expected_capability = os.getenv("LIGHTON_BBOX_EXPECTED_COMPUTE_CAPABILITY", "").strip()
    if expected_capability:
        report["expected_compute_capability"] = expected_capability
    if (
        require_cuda
        and expected_capability
        and report.get("compute_capability") != expected_capability
    ):
        raise RuntimeError(
            f"This training image targets compute capability {expected_capability}, "
            f"but the selected GPU reports {report.get('compute_capability')}. "
            "Rebuild causal-conv1d/TORCH_CUDA_ARCH_LIST for that GPU or run this image on the RTX 5090."
        )
    return report


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset(file_path: Path, split_name: str, processor=None):
    if not file_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {file_path}")
    if file_path.stat().st_size <= 0:
        raise ValueError(
            f"The {split_name} dataset metadata file is empty: {file_path}. "
            "Re-run the dataset export before training."
        )
    print(f"Loading {split_name} dataset from {file_path}", flush=True)
    try:
        dataset = load_dataset("json", data_files=str(file_path), split="train")
    except StopIteration as exc:
        raise ValueError(
            f"The {split_name} dataset contains no readable JSON samples: {file_path}. "
            "Re-run the dataset export before training."
        ) from exc
    if "split" not in dataset.column_names:
        dataset = dataset.map(lambda _: {"split": split_name})
    if len(dataset) == 0:
        raise ValueError(f"The {split_name} dataset is empty: {file_path}")
    if processor is not None and env_bool("LIGHTON_BBOX_GROUP_BY_LENGTH", True):
        def add_length(entry):
            path = resolve_image_path(entry, split_name)
            if path is None:
                raise FileNotFoundError(f"Missing image for page {entry.get('page_id')}")
            with Image.open(path) as image:
                width, height = image.size
            text_length = len(processor.tokenizer(extract_reference_text(entry), add_special_tokens=False)["input_ids"])
            # LightOn's page cost is dominated by visual patches. This is only a
            # bucketing proxy, so use the model's effective ~16 px patch grid
            # rather than running the image processor once per sample.
            return {"length": text_length + ((width + 15) // 16) * ((height + 15) // 16)}
        dataset = dataset.map(add_length, desc="Indexing page lengths")
    print(f"  {split_name}: {len(dataset)} samples", flush=True)
    return dataset


def resolve_image_path(entry, preferred_split=None):
    image_file = entry.get("image_file")
    if not image_file:
        messages = entry.get("messages") or []
        if messages and messages[0].get("content"):
            image_file = messages[0]["content"][0].get("image")
    if not image_file:
        return None

    candidate_splits = []
    if preferred_split:
        candidate_splits.append(preferred_split)
    candidate_splits.extend(split for split in SPLITS if split not in candidate_splits)

    for split_name in candidate_splits:
        candidate = DATASET_DIR / split_name / image_file
        if candidate.exists():
            return candidate
    return None


def extract_reference_text(entry) -> str:
    if entry.get("assistant_text") is not None:
        return str(entry.get("assistant_text", "")).strip()
    for message in entry.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for content in message.get("content") or []:
            if "text" in content:
                return str(content["text"]).strip()
    return ""


def messages_for_entry(image_path: Path, reference_text: str = None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
            ],
        }
    ]
    if reference_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": reference_text}],
            }
        )
    return messages


def apply_template(processor, messages, add_generation_prompt):
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )


def process_batch(examples, processor):
    batch_texts = []
    batch_images = []
    prompt_texts = []

    image_files = examples["image_file"]
    split_names = examples.get("split", [None] * len(image_files))
    references = examples.get("assistant_text", [""] * len(image_files))
    page_ids = examples.get("page_id", [None] * len(image_files))

    for image_file, split_name, reference, page_id in zip(
        image_files, split_names, references, page_ids
    ):
        entry = {"image_file": image_file, "split": split_name, "page_id": page_id}
        image_path = resolve_image_path(entry, split_name)
        if image_path is None:
            raise FileNotFoundError(f"Missing image for page {page_id}: {image_file}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        full_messages = messages_for_entry(image_path, str(reference))
        prompt_messages = messages_for_entry(image_path, None)
        batch_texts.append(apply_template(processor, full_messages, add_generation_prompt=False))
        prompt_texts.append(apply_template(processor, prompt_messages, add_generation_prompt=True))
        batch_images.append(image)

    model_inputs = processor(
        text=batch_texts,
        images=batch_images,
        padding=True,
        truncation=False,
        size={"longest_edge": IMAGE_LONGEST_EDGE},
        pad_to_multiple_of=int(os.getenv("LIGHTON_BBOX_PAD_TO_MULTIPLE_OF", "16")),
        return_tensors="pt",
    )

    labels = model_inputs["input_ids"].clone()
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("The processor must expose a tokenizer for assistant-only labels.")
    padding_side = getattr(tokenizer, "padding_side", "right")
    # Image expansion only affects the prompt. Tokenize the rendered text once,
    # without repeating the expensive image resize/normalization 2*batch_size times.
    rendered = tokenizer(batch_texts + prompt_texts, add_special_tokens=False,
                         padding=False, truncation=False)["input_ids"]
    assistant_lengths = []
    for idx, (full_ids, prompt_ids) in enumerate(zip(rendered[:len(batch_texts)], rendered[len(batch_texts):])):
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Chat template prompt is not a token prefix for page {page_ids[idx]}; refusing incorrect labels.")
        answer_ids = full_ids[len(prompt_ids):]
        assistant_len = len(answer_ids)
        full_len = int(model_inputs["attention_mask"][idx].sum())
        if not 0 < assistant_len < full_len:
            raise ValueError(f"Invalid assistant span for page {page_ids[idx]}.")
        end = labels.shape[1] if padding_side == "left" else full_len
        start = end - assistant_len
        if model_inputs["input_ids"][idx, start:end].tolist() != answer_ids:
            raise ValueError(f"Multimodal/text suffix mismatch for page {page_ids[idx]}; check the processor template.")
        labels[idx, :start] = -100
        labels[idx, end:] = -100
        assistant_lengths.append(assistant_len)
    # Mask positions, NOT token values: PAD can share the EOS id, which must
    # remain supervised or the model never learns to stop generating.
    labels.masked_fill_(model_inputs["attention_mask"].eq(0), -100)

    model_inputs["labels"] = labels
    model_inputs["logits_to_keep"] = max(assistant_lengths) + 1 if padding_side == "left" else 0
    return model_inputs


class LightOnBBoxCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        if not features:
            raise ValueError("Cannot collate an empty batch.")
        keys = ("page_id", "image_file", "split", "assistant_text")
        examples = {key: [feature.get(key) for feature in features] for key in keys}
        examples["assistant_text"] = [extract_reference_text(feature) for feature in features]
        return process_batch(examples, self.processor)


def parse_bbox_output(text: str):
    results = []
    for line in str(text or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = BBOX_PATTERN.match(line)
        if not match:
            continue
        bbox = [int(match.group(i)) for i in range(2, 6)]
        if any(coord < 0 or coord > BBOX_NORM_SCALE for coord in bbox):
            continue
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        results.append({"text": match.group(1).strip(), "bbox": bbox})
    return results


def normalize_prediction_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = max(0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_giou(box_a, box_b):
    iou = compute_iou(box_a, box_b)
    enc_x1 = min(box_a[0], box_b[0])
    enc_y1 = min(box_a[1], box_b[1])
    enc_x2 = max(box_a[2], box_b[2])
    enc_y2 = max(box_a[3], box_b[3])
    enc_area = max(0, (enc_x2 - enc_x1) * (enc_y2 - enc_y1))
    if enc_area <= 0:
        return iou
    area_a = max(0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    inter_w = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    inter_h = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    inter_area = inter_w * inter_h
    return iou - (enc_area - area_a - area_b + inter_area) / enc_area


def match_predictions_to_gt(pred_items, gt_items, iou_threshold=0.5):
    if not pred_items or not gt_items:
        return [], list(range(len(gt_items))), list(range(len(pred_items)))

    iou_matrix = np.zeros((len(pred_items), len(gt_items)))
    for i, pred in enumerate(pred_items):
        for j, gt in enumerate(gt_items):
            iou_matrix[i, j] = compute_iou(pred["bbox"], gt["bbox"])

    matched_pairs = []
    used_preds = set()
    used_gts = set()
    while True:
        if iou_matrix.size == 0:
            break
        max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        max_iou = iou_matrix[max_idx]
        if max_iou < iou_threshold:
            break
        pred_idx, gt_idx = max_idx
        if pred_idx in used_preds or gt_idx in used_gts:
            iou_matrix[pred_idx, gt_idx] = 0
            continue
        matched_pairs.append((pred_idx, gt_idx, float(max_iou)))
        used_preds.add(pred_idx)
        used_gts.add(gt_idx)
        iou_matrix[pred_idx, :] = 0
        iou_matrix[:, gt_idx] = 0

    unmatched_gts = [idx for idx in range(len(gt_items)) if idx not in used_gts]
    unmatched_preds = [idx for idx in range(len(pred_items)) if idx not in used_preds]
    return matched_pairs, unmatched_gts, unmatched_preds


def safe_wer(reference: str, prediction: str):
    if not reference:
        return 0.0 if not prediction else 1.0
    try:
        return jiwer.wer(reference, prediction)
    except Exception:
        return 1.0


def compute_metrics_from_results(results):
    metrics = {
        "num_samples": len(results),
        "total_gt_bubbles": 0,
        "total_pred_bubbles": 0,
    }
    all_sample_cer = []
    all_sample_wer = []
    all_matched_iou = []
    all_matched_giou = []
    all_bbox_area_error = []
    detection_rates = []
    inference_times = []
    exact_matches = 0
    precision_by_threshold = {threshold: [] for threshold in IOU_THRESHOLDS}
    recall_by_threshold = {threshold: [] for threshold in IOU_THRESHOLDS}
    f1_by_threshold = {threshold: [] for threshold in IOU_THRESHOLDS}
    per_sample = []

    for result in results:
        gt_items = result["gt_items"]
        pred_items = result["pred_items"]
        n_gt = len(gt_items)
        n_pred = len(pred_items)
        metrics["total_gt_bubbles"] += n_gt
        metrics["total_pred_bubbles"] += n_pred
        inference_times.append(result.get("inference_time", 0.0))

        if n_gt == 0 and n_pred == 0:
            success = not result.get("error")
            exact_matches += int(success)
            detection_rates.append(float(success))
            all_sample_cer.append(0.0 if success else 1.0)
            all_sample_wer.append(0.0 if success else 1.0)
            for threshold in IOU_THRESHOLDS:
                precision_by_threshold[threshold].append(float(success))
                recall_by_threshold[threshold].append(float(success))
                f1_by_threshold[threshold].append(float(success))
            per_sample.append(
                {
                    "page_id": result.get("page_id"),
                    "cer": 0.0 if success else 1.0,
                    "wer": 0.0 if success else 1.0,
                    "mean_iou": float(success),
                    "num_gt": 0,
                    "num_pred": 0,
                    "detection_rate": float(success),
                    "precision@0_5": float(success),
                    "recall@0_5": float(success),
                    "f1@0_5": float(success),
                }
            )
            continue

        detection_rate = 0.0
        if n_gt == n_pred and normalize_prediction_text(result["pred_text"]) == normalize_prediction_text(result["gt_text"]):
            exact_matches += 1

        sample_cers = []
        sample_wers = []
        sample_ious = []
        sample_pr = {}

        for threshold in IOU_THRESHOLDS:
            matched, unmatched_gt, unmatched_pred = match_predictions_to_gt(
                pred_items, gt_items, iou_threshold=threshold
            )
            tp = len(matched)
            fp = len(unmatched_pred)
            fn = len(unmatched_gt)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            precision_by_threshold[threshold].append(precision)
            recall_by_threshold[threshold].append(recall)
            f1_by_threshold[threshold].append(f1)
            sample_pr[threshold] = (precision, recall, f1)

            if threshold == 0.5:
                detection_rate = recall
                for pred_idx, gt_idx, iou_value in matched:
                    pred_text = normalize_prediction_text(pred_items[pred_idx]["text"])
                    gt_text = normalize_prediction_text(gt_items[gt_idx]["text"])
                    sample_cers.append(
                        levenshtein_distance(pred_text, gt_text) / max(len(gt_text), 1)
                    )
                    sample_wers.append(safe_wer(gt_text, pred_text))
                    sample_ious.append(iou_value)
                    all_matched_iou.append(iou_value)
                    all_matched_giou.append(
                        compute_giou(pred_items[pred_idx]["bbox"], gt_items[gt_idx]["bbox"])
                    )
                    gt_area = max(
                        0,
                        (gt_items[gt_idx]["bbox"][2] - gt_items[gt_idx]["bbox"][0])
                        * (gt_items[gt_idx]["bbox"][3] - gt_items[gt_idx]["bbox"][1]),
                    )
                    pred_area = max(
                        0,
                        (pred_items[pred_idx]["bbox"][2] - pred_items[pred_idx]["bbox"][0])
                        * (pred_items[pred_idx]["bbox"][3] - pred_items[pred_idx]["bbox"][1]),
                    )
                    if gt_area > 0:
                        all_bbox_area_error.append(abs(pred_area - gt_area) / gt_area)

        sample_cer = float(np.mean(sample_cers)) if sample_cers else 1.0
        sample_wer = float(np.mean(sample_wers)) if sample_wers else 1.0
        sample_iou = float(np.mean(sample_ious)) if sample_ious else 0.0
        all_sample_cer.append(sample_cer)
        all_sample_wer.append(sample_wer)
        detection_rates.append(detection_rate)

        p50, r50, f50 = sample_pr.get(0.5, (0.0, 0.0, 0.0))
        per_sample.append(
            {
                "page_id": result.get("page_id"),
                "cer": sample_cer,
                "wer": sample_wer,
                "mean_iou": sample_iou,
                "num_gt": n_gt,
                "num_pred": n_pred,
                "detection_rate": detection_rate,
                "precision@0_5": p50,
                "recall@0_5": r50,
                "f1@0_5": f50,
            }
        )

    metrics["cer"] = float(np.mean(all_sample_cer)) if all_sample_cer else 1.0
    metrics["wer"] = float(np.mean(all_sample_wer)) if all_sample_wer else 1.0
    metrics["exact_match_rate"] = exact_matches / max(len(results), 1)
    metrics["mean_iou"] = float(np.mean(all_matched_iou)) if all_matched_iou else 0.0
    metrics["median_iou"] = float(np.median(all_matched_iou)) if all_matched_iou else 0.0
    metrics["mean_giou"] = float(np.mean(all_matched_giou)) if all_matched_giou else 0.0
    metrics["mean_bbox_area_error"] = (
        float(np.mean(all_bbox_area_error)) if all_bbox_area_error else 0.0
    )
    metrics["avg_detection_rate"] = float(np.mean(detection_rates)) if detection_rates else 0.0
    metrics["avg_inference_time"] = float(np.mean(inference_times)) if inference_times else 0.0
    metrics["generation_errors"] = sum(bool(result.get("error")) for result in results)

    for threshold in IOU_THRESHOLDS:
        suffix = str(threshold).replace(".", "_")
        metrics[f"precision@{suffix}"] = (
            float(np.mean(precision_by_threshold[threshold]))
            if precision_by_threshold[threshold]
            else 0.0
        )
        metrics[f"recall@{suffix}"] = (
            float(np.mean(recall_by_threshold[threshold]))
            if recall_by_threshold[threshold]
            else 0.0
        )
        metrics[f"f1@{suffix}"] = (
            float(np.mean(f1_by_threshold[threshold]))
            if f1_by_threshold[threshold]
            else 0.0
        )

    metrics["iou_distribution"] = [float(value) for value in all_matched_iou]
    metrics["cer_distribution"] = [float(value) for value in all_sample_cer]
    metrics["bbox_area_error_distribution"] = [float(value) for value in all_bbox_area_error]
    metrics["per_sample"] = per_sample

    ious = metrics["iou_distribution"]
    metrics["iou_p25"] = float(np.percentile(ious, 25)) if ious else 0.0
    metrics["iou_p75"] = float(np.percentile(ious, 75)) if ious else 0.0
    metrics["iou_p90"] = float(np.percentile(ious, 90)) if ious else 0.0
    metrics["iou_p95"] = float(np.percentile(ious, 95)) if ious else 0.0
    metrics["cer_median"] = float(np.median(metrics["cer_distribution"])) if metrics["cer_distribution"] else 1.0
    metrics["combined_score"] = (
        (1 - min(max(metrics["cer"], 0.0), 1.0)) * 0.4
        + metrics["f1@0_5"] * 0.3
        + metrics["mean_iou"] * 0.2
        + metrics["avg_detection_rate"] * 0.1
    )
    return metrics


def model_dtype(model):
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def move_inputs_to_device(inputs, device, dtype):
    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def decode_tokens(processor, token_ids):
    if hasattr(processor, "decode"):
        return processor.decode(token_ids, skip_special_tokens=True)
    return processor.tokenizer.decode(token_ids, skip_special_tokens=True)


def batch_decode_tokens(processor, token_ids):
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(token_ids, skip_special_tokens=True)
    return processor.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def generate_lighton_prediction(model, processor, entry):
    split_name = entry.get("split") or "test"
    image_path = resolve_image_path(entry, split_name)
    if image_path is None:
        raise FileNotFoundError(f"Missing image for page {entry.get('page_id')}")
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    messages = messages_for_entry(image_path, None)
    prompt = apply_template(processor, messages, add_generation_prompt=True)
    inputs = processor(
        text=[prompt],
        images=[image],
        size={"longest_edge": IMAGE_LONGEST_EDGE},
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    dtype = model_dtype(model)
    inputs = move_inputs_to_device(inputs, device, dtype)

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                               enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return decode_tokens(processor, gen_ids).strip()


def generate_lighton_predictions(model, processor, entries):
    """Generate several pages in one call so large GPUs are not starved by batch=1 decoding."""
    if not entries:
        return []
    prompts = []
    images = []
    try:
        for entry in entries:
            split_name = entry.get("split") or "test"
            image_path = resolve_image_path(entry, split_name)
            if image_path is None:
                raise FileNotFoundError(f"Missing image for page {entry.get('page_id')}")
            with Image.open(image_path) as img:
                images.append(img.convert("RGB"))
            messages = messages_for_entry(image_path, None)
            prompts.append(apply_template(processor, messages, add_generation_prompt=True))

        inputs = processor(
            text=prompts,
            images=images,
            padding=True,
            size={"longest_edge": IMAGE_LONGEST_EDGE},
            pad_to_multiple_of=int(os.getenv("LIGHTON_BBOX_PAD_TO_MULTIPLE_OF", "16")),
            return_tensors="pt",
        )
    finally:
        for image in images:
            try:
                image.close()
            except Exception:
                pass

    device = next(model.parameters()).device
    dtype = model_dtype(model)
    inputs = move_inputs_to_device(inputs, device, dtype)
    prompt_width = inputs["input_ids"].shape[1]
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    ):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    generated = output_ids[:, prompt_width:].detach().cpu()
    return [text.strip() for text in batch_decode_tokens(processor, generated)]


def benchmark_indices(dataset, max_samples):
    total = len(dataset)
    if max_samples is None or max_samples <= 0 or max_samples >= total:
        return list(range(total))
    rng = random.Random(RANDOM_SEED)
    return sorted(rng.sample(range(total), max_samples))


def evaluation_subset(dataset, max_samples):
    """Return a deterministic eval subset while keeping the held-out test set intact."""
    if dataset is None or not max_samples or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    indices = benchmark_indices(dataset, max_samples)
    return dataset.select(indices) if hasattr(dataset, "select") else [dataset[index] for index in indices]


def run_generation_benchmark(model, *args, **kwargs):
    """Restore training/cache state even when generation or metric computation fails."""
    configs = [model.config]
    if hasattr(model.config, "get_text_config"):
        text_config = model.config.get_text_config()
        if text_config is not model.config:
            configs.append(text_config)
    old_caches = [(config, config.use_cache) for config in configs if hasattr(config, "use_cache")]
    was_training = model.training
    try:
        for config, _ in old_caches:
            config.use_cache = True
        return _run_generation_benchmark(model, *args, **kwargs)
    finally:
        for config, value in old_caches:
            config.use_cache = value
        model.train(was_training)


def _run_generation_benchmark(
    model,
    processor,
    dataset,
    split_name,
    generator,
    model_label,
    max_samples=None,
    worst_count=20,
    batch_generator=None,
    batch_size=None,
):
    was_training = model.training
    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = True
    model.eval()

    indices = benchmark_indices(dataset, max_samples)
    results = []
    benchmark_start = time.perf_counter()
    print("", flush=True)
    print("=" * 72, flush=True)
    print(f"{model_label} BBOX BENCHMARK [{split_name}] - {len(indices)}/{len(dataset)} pages", flush=True)
    print("=" * 72, flush=True)

    requested_batch_size = max(1, int(batch_size or os.getenv("LIGHTON_BBOX_GEN_BATCH", "1")))
    active_batch_size = requested_batch_size if batch_generator is not None else 1
    offset = 0
    while offset < len(indices):
        current_indices = indices[offset : offset + active_batch_size]
        entries = [dataset[index] for index in current_indices]
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        try:
            if batch_generator is not None:
                predictions = batch_generator(model, processor, entries)
                if len(predictions) != len(entries):
                    raise RuntimeError(
                        f"Batch generator returned {len(predictions)} predictions for {len(entries)} inputs."
                    )
            else:
                predictions = [generator(model, processor, entries[0])]
            errors = [None] * len(entries)
        except torch.cuda.OutOfMemoryError:
            if batch_generator is None or active_batch_size <= 1:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            reduced = max(1, active_batch_size // 2)
            print(
                f"  generation OOM at batch={active_batch_size}; retrying with batch={reduced}",
                flush=True,
            )
            active_batch_size = reduced
            continue
        except Exception as exc:
            predictions = [""] * len(entries)
            errors = [str(exc)] * len(entries)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        per_page_elapsed = elapsed / max(len(entries), 1)

        for dataset_index, entry, pred_text, error in zip(
            current_indices, entries, predictions, errors
        ):
            gt_text = extract_reference_text(entry)
            gt_items = parse_bbox_output(gt_text)
            pred_items = parse_bbox_output(pred_text)
            results.append(
                {
                    "dataset_idx": dataset_index,
                    "page_id": entry.get("page_id"),
                    "image_file": entry.get("image_file"),
                    "gt_text": gt_text,
                    "pred_text": pred_text,
                    "gt_items": gt_items,
                    "pred_items": pred_items,
                    "num_gt_bubbles": len(gt_items),
                    "num_pred_bubbles": len(pred_items),
                    "inference_time": per_page_elapsed,
                    "error": error,
                }
            )
        offset += len(current_indices)
        print(
            f"  generated {offset}/{len(indices)} (batch={len(current_indices)}, {elapsed:.1f}s)",
            flush=True,
        )

    metrics = compute_metrics_from_results(results)
    benchmark_elapsed = time.perf_counter() - benchmark_start
    metrics["benchmark_wall_time"] = benchmark_elapsed
    metrics["pages_per_second"] = len(indices) / benchmark_elapsed if benchmark_elapsed > 0 else 0.0
    metrics["generation_batch_size"] = active_batch_size
    ranked = sorted(
        metrics["per_sample"],
        key=lambda sample: (sample["f1@0_5"], sample["mean_iou"], -sample["cer"]),
    )
    print("-" * 72, flush=True)
    print(f"CER:             {metrics['cer']:.6f} ({metrics['cer'] * 100:.3f}%)", flush=True)
    print(f"WER:             {metrics['wer']:.6f} ({metrics['wer'] * 100:.3f}%)", flush=True)
    print(f"Mean IoU:        {metrics['mean_iou']:.6f}", flush=True)
    print(f"F1@0.5:          {metrics['f1@0_5']:.6f}", flush=True)
    print(f"Detection rate:  {metrics['avg_detection_rate']:.6f}", flush=True)
    print(f"Combined score:  {metrics['combined_score']:.6f}", flush=True)
    print(f"Avg inference:   {metrics['avg_inference_time']:.3f}s/page", flush=True)
    print(f"Throughput:      {metrics['pages_per_second']:.3f} pages/s (batch={active_batch_size})", flush=True)
    print("-" * 72, flush=True)
    for rank, sample in enumerate(ranked[: min(worst_count, len(ranked))], 1):
        print(
            f"#{rank} page={sample['page_id']} CER={sample['cer']:.4f} "
            f"IoU={sample['mean_iou']:.4f} F1@0.5={sample['f1@0_5']:.4f} "
            f"GT={sample['num_gt']} PRED={sample['num_pred']}",
            flush=True,
        )
    print("=" * 72, flush=True)

    if old_use_cache is not None:
        model.config.use_cache = old_use_cache
    if was_training:
        model.train()
    return metrics, results


class PromptOnlyEvalTrainer(Seq2SeqTrainer):
    def __init__(self, *args, processor=None, gen_eval_max_samples=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.gen_eval_max_samples = gen_eval_max_samples
        # This implementation really uses the token count over the whole gradient
        # accumulation window. Do not divide again by gradient_accumulation_steps.
        self.model_accepts_loss_kwargs = True
        self._loss_shifts_labels = True
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        self.trim_logits = env_bool("LIGHTON_BBOX_TRIM_LOGITS", True) and "logits_to_keep" in inspect.signature(base.forward).parameters

    def _get_eval_sampler(self, eval_dataset):
        # Transformers 5.14.1 reuses `train_sampling_strategy=group_by_length`
        # for evaluation too. Our eval dataset intentionally does not materialize
        # image tensors or a synthetic length column, so the stock implementation
        # tries to infer lengths from a `pixel_values` field and crashes at the
        # first validation pass. Keep length grouping where it matters (training)
        # and make eval deterministic/sequential on this single-GPU trainer.
        if getattr(self.args, "train_sampling_strategy", None) == "group_by_length":
            if eval_dataset is None:
                return None
            return torch.utils.data.SequentialSampler(eval_dataset)
        return super()._get_eval_sampler(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        keep = inputs.pop("logits_to_keep", 0)
        if self.trim_logits:
            inputs["logits_to_keep"] = keep
        outputs = model(**inputs, use_cache=False)
        # Keep one preceding token: logits at position t predict label t+1.
        logits = outputs.logits[:, :-1, :].contiguous()
        targets = labels[:, -outputs.logits.shape[1]:][:, 1:].contiguous()
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               targets.reshape(-1), ignore_index=-100, reduction="sum")
        denominator = num_items_in_batch if num_items_in_batch is not None else targets.ne(-100).sum()
        if torch.is_tensor(denominator):
            denominator = denominator.to(device=loss.device).clamp_min(1)
        else:
            denominator = max(denominator, 1)
        loss = loss / denominator
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                        ignore_keys=None, metric_key_prefix="eval"):
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only=True,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        dataset = dataloader.dataset
        if dataset is None or self.gen_eval_max_samples == 0:
            return output
        try:
            split_name = dataset[0].get("split") or "val"
        except Exception:
            split_name = "val"
        monitor = next((cb for cb in self.callback_handler.callbacks if isinstance(cb, TrainingMonitorCallback)), None)
        if monitor is not None:
            monitor.summary["status"] = "validating"
            monitor._persist("validation_start")
        try:
            with self.compute_loss_context_manager():
                gen_metrics, results = run_generation_benchmark(
                    self.model, self.processor, dataset, split_name=split_name,
                    generator=generate_lighton_prediction, model_label="LIGHTON VALIDATION",
                    max_samples=self.gen_eval_max_samples, worst_count=5,
                    batch_generator=generate_lighton_predictions,
                    batch_size=int(os.getenv("LIGHTON_BBOX_GEN_BATCH", "1")),
                )
            save_benchmark(Path(self.args.output_dir) / "validation_latest.json", MODEL_ID, gen_metrics, results)
            if gen_metrics["generation_errors"] and env_bool("LIGHTON_BBOX_FAIL_ON_GENERATION_ERROR", True):
                raise RuntimeError("Validation generation failed on one or more pages; see validation_latest.json.")
        finally:
            if monitor is not None:
                monitor.summary["status"] = "training"
        prefixed = {
            f"{metric_key_prefix}_{key}": value
            for key, value in gen_metrics.items()
            if isinstance(value, (int, float))
        }
        # Trainer.evaluate logs and dispatches on_evaluate AFTER this method.
        # EarlyStoppingCallback now sees the generation score on its first call.
        output.metrics.update(prefixed)
        return output


def find_best_checkpoint(output_dir):
    output_dir = Path(output_dir)
    state_files = [output_dir / "trainer_state.json"]
    state_files.extend(
        Path(path) / "trainer_state.json" for path in glob.glob(str(output_dir / "checkpoint-*"))
    )
    for state_file in sorted(
        state_files,
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    ):
        if not state_file.exists():
            continue
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            best = state.get("best_model_checkpoint")
            if best and Path(best).exists():
                return best
        except Exception:
            continue

    checkpoints = sorted(glob.glob(str(output_dir / "checkpoint-*")), key=os.path.getmtime)
    return checkpoints[-1] if checkpoints else None


def configure_processor(model_id_or_path=MODEL_ID):
    print(f"Loading processor: {model_id_or_path}", flush=True)
    processor = LightOnOcrProcessor.from_pretrained(model_id_or_path)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and hasattr(image_processor, "default_to_square"):
        image_processor.default_to_square = False
    if image_processor is not None:
        image_processor.size = {"longest_edge": IMAGE_LONGEST_EDGE}
    return processor


def valid_token_id(token_id, tokenizer) -> bool:
    if token_id is None or tokenizer is None:
        return False
    try:
        token_count = len(tokenizer)
    except Exception:
        return isinstance(token_id, int)
    if isinstance(token_id, int):
        return 0 <= token_id < token_count
    if isinstance(token_id, (list, tuple)):
        return all(isinstance(item, int) and 0 <= item < token_count for item in token_id)
    return False


def configure_generation(model, processor=None):
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.generation_config.do_sample = False
    model.generation_config.max_new_tokens = MAX_NEW_TOKENS
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.max_length = None

    tokenizer = getattr(processor, "tokenizer", None) if processor is not None else None
    if tokenizer is None:
        return
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    configs = [model.config]
    if hasattr(model.config, "get_text_config"):
        text_config = model.config.get_text_config()
        if text_config is not model.config:
            configs.append(text_config)
    if valid_token_id(eos_token_id, tokenizer):
        model.generation_config.eos_token_id = eos_token_id
        for config in configs:
            if hasattr(config, "eos_token_id"):
                config.eos_token_id = eos_token_id
    if valid_token_id(pad_token_id, tokenizer):
        model.generation_config.pad_token_id = pad_token_id
        for config in configs:
            if hasattr(config, "pad_token_id"):
                config.pad_token_id = pad_token_id


def load_lighton_model(model_id_or_path=MODEL_ID, for_training=False):
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float32
    if for_training and os.getenv("LIGHTON_BBOX_TRAIN_MODE", "lora").lower() == "full":
        # Full fine-tuning keeps master parameters/Adam moments in FP32; the
        # Trainer still uses BF16 autocast. Avoid low-precision Adam updates.
        dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LightOnOCR model: {model_id_or_path} ({dtype}, device={device})", flush=True)
    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    attn_impl = os.getenv("LIGHTON_BBOX_ATTN_IMPLEMENTATION", "sdpa").strip()
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    if device == "cuda":
        kwargs["device_map"] = {"": "cuda:0"}
    model = LightOnOcrForConditionalGeneration.from_pretrained(model_id_or_path, **kwargs)
    if device != "cuda":
        model.to(device)
    return model


def build_lora_config():
    from peft import LoraConfig

    # The 5090 has enough parameter-state headroom to use a materially larger
    # adapter than the original r=64 setup. Activations, not LoRA weights, are the
    # limiting factor at 1500px (batch=1 already peaks around 26 GiB), so r=128 is
    # a better use of the remaining VRAM without the optimizer cost of full SFT.
    lora_r = int(os.getenv("LIGHTON_BBOX_LORA_R", "128"))
    lora_alpha = int(os.getenv("LIGHTON_BBOX_LORA_ALPHA", str(lora_r * 2)))
    target_modules = [
        item.strip()
        for item in os.getenv(
            "LIGHTON_BBOX_LORA_TARGET_MODULES",
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        ).split(",")
        if item.strip()
    ]
    modules_to_save = [
        item.strip()
        for item in os.getenv(
            "LIGHTON_BBOX_MODULES_TO_SAVE",
            "vision_projection",
        ).split(",")
        if item.strip()
    ]
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=float(os.getenv("LIGHTON_BBOX_LORA_DROPOUT", "0")),
        use_rslora=env_bool("LIGHTON_BBOX_USE_RSLORA", True),
        use_dora=env_bool("LIGHTON_BBOX_USE_DORA", False),
        target_modules=target_modules,
        modules_to_save=modules_to_save or None,
        bias="none",
        task_type="CAUSAL_LM",
    )


def configure_trainable_model(model, resume_checkpoint=None):
    train_mode = os.getenv("LIGHTON_BBOX_TRAIN_MODE", "lora").strip().lower()
    if train_mode == "full":
        if resume_checkpoint and (Path(resume_checkpoint) / "adapter_config.json").exists():
            raise ValueError("Cannot resume an adapter checkpoint in full training mode.")
        print("Training mode: full fine-tuning", flush=True)
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model, "full"

    if train_mode != "lora":
        raise ValueError("LIGHTON_BBOX_TRAIN_MODE must be lora or full.")
    from peft import PeftModel, get_peft_model

    print("Training mode: rsLoRA (attention + MLP projections)", flush=True)
    if resume_checkpoint:
        if not (Path(resume_checkpoint) / "adapter_config.json").exists():
            raise ValueError("This is a full-model checkpoint; set LIGHTON_BBOX_TRAIN_MODE=full.")
        # Preserve the original adapter rank, targets and DoRA settings on resume.
        model = PeftModel.from_pretrained(model, resume_checkpoint, is_trainable=True)
    else:
        model = get_peft_model(model, build_lora_config())
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, "lora"


def configure_gradient_checkpointing(model, enabled: bool):
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if enabled:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        try:
            model.gradient_checkpointing_disable()
        except AttributeError:
            pass


def _calibration_loss(model, batch):
    """Same assistant-only causal loss as the trainer, without optimizer state."""
    inputs = dict(batch)
    labels = inputs.pop("labels")
    keep = inputs.pop("logits_to_keep", 0)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    supports_trim = "logits_to_keep" in inspect.signature(base.forward).parameters
    if supports_trim and keep:
        inputs["logits_to_keep"] = keep
    outputs = model(**inputs, use_cache=False)
    logits = outputs.logits[:, :-1, :].contiguous()
    targets = labels[:, -outputs.logits.shape[1] :][:, 1:].contiguous()
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )


def _hardest_training_features(dataset, count):
    ranked = sorted(
        (dataset[index] for index in range(len(dataset))),
        key=lambda entry: int(entry.get("length") or 0),
        reverse=True,
    )
    return ranked[: max(1, min(count, len(ranked)))]


def _project_next_calibration_ratio(
    *, baseline_bytes, observed_peak_bytes, observed_batch, next_batch, total_vram
):
    """Conservatively extrapolate peak VRAM before probing a larger batch."""
    if observed_batch <= 0 or next_batch <= observed_batch or total_vram <= 0:
        return 0.0
    dynamic = max(0, int(observed_peak_bytes) - int(baseline_bytes))
    per_sample = dynamic / max(int(observed_batch), 1)
    projected = int(baseline_bytes) + per_sample * int(next_batch)
    return projected / int(total_vram)


def calibrate_rtx5090_batch(model, processor, train_dataset, profile, resume_checkpoint=None):
    """Select the fastest safe physical batch from real worst-case pages.

    This intentionally calibrates only fresh runs. A resumed run reuses the
    persisted profile so optimizer/scheduler semantics do not change mid-run.
    """
    profile_path = OUTPUT_DIR / "hardware_profile.json"
    if resume_checkpoint and profile_path.exists():
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        for key in ("train_batch", "grad_accum", "gradient_checkpointing"):
            if key not in payload:
                break
        else:
            os.environ["LIGHTON_BBOX_TRAIN_BATCH"] = str(payload["train_batch"])
            os.environ["LIGHTON_BBOX_GRAD_ACCUM"] = str(payload["grad_accum"])
            os.environ["LIGHTON_BBOX_GRADIENT_CHECKPOINTING"] = "1" if payload["gradient_checkpointing"] else "0"
            configure_gradient_checkpointing(model, bool(payload["gradient_checkpointing"]))
            print(
                f"Reusing hardware profile: batch={payload['train_batch']} x accum={payload['grad_accum']}",
                flush=True,
            )
            return payload

    if (
        profile != "rtx5090"
        or not torch.cuda.is_available()
        or not env_bool("LIGHTON_BBOX_AUTO_BATCH", True)
    ):
        payload = {
            "profile": profile,
            "auto_batch": False,
            "train_batch": int(os.getenv("LIGHTON_BBOX_TRAIN_BATCH", "1")),
            "grad_accum": int(os.getenv("LIGHTON_BBOX_GRAD_ACCUM", "1")),
            "gradient_checkpointing": env_bool("LIGHTON_BBOX_GRADIENT_CHECKPOINTING", False),
        }
        configure_gradient_checkpointing(model, payload["gradient_checkpointing"])
        atomic_json(profile_path, payload)
        return payload

    candidates = sorted(
        {
            int(value.strip())
            for value in os.getenv("LIGHTON_BBOX_BATCH_CANDIDATES", "1,2,4,8").split(",")
            if value.strip() and int(value.strip()) > 0
        }
    )
    if not candidates:
        raise ValueError("LIGHTON_BBOX_BATCH_CANDIDATES contains no valid batch size.")
    sample_pool = _hardest_training_features(train_dataset, max(candidates))
    collator = LightOnBBoxCollator(processor)
    device = next(model.parameters()).device
    dtype = model_dtype(model)
    total_vram = torch.cuda.get_device_properties(device).total_memory
    max_ratio = float(os.getenv("LIGHTON_BBOX_CALIBRATION_MAX_VRAM_RATIO", "0.90"))
    probe_ratio = float(os.getenv("LIGHTON_BBOX_CALIBRATION_PROBE_MAX_VRAM_RATIO", str(max_ratio)))
    attempts = []
    best = None

    # Native backward is materially faster. Checkpointing is a fallback only if
    # no useful physical batch fits without it.
    for checkpointing in (False, True):
        configure_gradient_checkpointing(model, checkpointing)
        previous_success = None
        for batch_size in candidates:
            if previous_success is not None:
                projected_ratio = _project_next_calibration_ratio(
                    baseline_bytes=previous_success["baseline_vram_bytes"],
                    observed_peak_bytes=previous_success["peak_vram_bytes"],
                    observed_batch=previous_success["batch_size"],
                    next_batch=batch_size,
                    total_vram=total_vram,
                )
                if projected_ratio > probe_ratio:
                    attempts.append(
                        {
                            "batch_size": batch_size,
                            "gradient_checkpointing": checkpointing,
                            "status": "skipped_projected_oom",
                            "projected_peak_vram_ratio": projected_ratio,
                        }
                    )
                    print(
                        f"Calibration batch={batch_size} checkpointing={checkpointing}: "
                        f"skipped (projected peak {projected_ratio:.1%} > {probe_ratio:.1%}).",
                        flush=True,
                    )
                    break
            features = [sample_pool[index % len(sample_pool)] for index in range(batch_size)]
            batch = loss = None
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                baseline_alloc = torch.cuda.memory_allocated(device)
                batch = move_inputs_to_device(collator(features), device, dtype)
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                loss = _calibration_loss(model, batch)
                loss.backward()
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated(device)
                ratio = peak / max(total_vram, 1)
                attempt = {
                    "batch_size": batch_size,
                    "gradient_checkpointing": checkpointing,
                    "status": "ok",
                    "seconds": elapsed,
                    "samples_per_second": batch_size / max(elapsed, 1e-9),
                    "peak_vram_gib": peak / 2**30,
                    "peak_vram_ratio": ratio,
                    "baseline_vram_bytes": baseline_alloc,
                    "peak_vram_bytes": peak,
                }
                attempts.append(attempt)
                previous_success = attempt
                print(
                    f"Calibration batch={batch_size} checkpointing={checkpointing}: "
                    f"{attempt['samples_per_second']:.2f} pages/s, peak={attempt['peak_vram_gib']:.2f} GiB ({ratio:.1%})",
                    flush=True,
                )
                if ratio <= max_ratio and (
                    best is None
                    or attempt["samples_per_second"] > best["samples_per_second"] * 1.01
                ):
                    best = attempt
                if ratio > max_ratio:
                    break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                message = str(exc).lower()
                if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in message:
                    raise
                attempts.append(
                    {
                        "batch_size": batch_size,
                        "gradient_checkpointing": checkpointing,
                        "status": "oom",
                        "error": str(exc)[:500],
                    }
                )
                print(f"Calibration batch={batch_size}: CUDA OOM, stopping this mode.", flush=True)
                break
            finally:
                model.zero_grad(set_to_none=True)
                del loss, batch
                gc.collect()
                torch.cuda.empty_cache()
        if best is not None and not best["gradient_checkpointing"]:
            break

    if best is None:
        raise RuntimeError("RTX 5090 auto-batch calibration could not find a safe physical batch.")
    train_batch = int(best["batch_size"])
    # Preserve the established optimization dynamics (effective batch 8) while
    # maximizing the physical batch. If 8 pages fit, accumulation becomes 1;
    # otherwise we compensate rather than silently changing the run semantics.
    effective_batch = max(1, int(os.getenv("LIGHTON_BBOX_EFFECTIVE_BATCH", "8")))
    grad_accum = max(1, math.ceil(effective_batch / train_batch))
    checkpointing = bool(best["gradient_checkpointing"])
    os.environ["LIGHTON_BBOX_TRAIN_BATCH"] = str(train_batch)
    os.environ["LIGHTON_BBOX_GRAD_ACCUM"] = str(grad_accum)
    os.environ["LIGHTON_BBOX_GRADIENT_CHECKPOINTING"] = "1" if checkpointing else "0"
    # Keep validation/generation conservative unless the user explicitly overrides
    # them. On this LightOn vision stack, a single worst-case training page already
    # consumes most of the 32 GiB card, and unsafe eval/gen batches can poison the
    # CUDA context just like an over-aggressive calibration probe.
    # Runtime profile defaults must not override the calibration result. Preserve
    # only values that were genuinely supplied by the user/container environment.
    if "LIGHTON_BBOX_EVAL_BATCH" not in INITIAL_ENV_KEYS:
        os.environ["LIGHTON_BBOX_EVAL_BATCH"] = str(train_batch)
    if "LIGHTON_BBOX_GEN_BATCH" not in INITIAL_ENV_KEYS:
        os.environ["LIGHTON_BBOX_GEN_BATCH"] = str(max(1, min(train_batch, 2)))
    configure_gradient_checkpointing(model, checkpointing)
    payload = {
        "profile": profile,
        "auto_batch": True,
        "gpu": torch.cuda.get_device_name(device),
        "vram_gib": total_vram / 2**30,
        "image_longest_edge": IMAGE_LONGEST_EDGE,
        "max_vram_ratio": max_ratio,
        "train_batch": train_batch,
        "grad_accum": grad_accum,
        "effective_batch": train_batch * grad_accum,
        "gradient_checkpointing": checkpointing,
        "selected_samples_per_second": best["samples_per_second"],
        "selected_peak_vram_gib": best["peak_vram_gib"],
        "attempts": attempts,
    }
    atomic_json(profile_path, payload)
    print(
        f"Selected RTX 5090 profile: batch={train_batch}, accum={grad_accum}, "
        f"checkpointing={checkpointing}, peak={best['peak_vram_gib']:.2f} GiB",
        flush=True,
    )
    return payload


def estimate_total_update_steps(train_size, train_batch, grad_accum, epochs, smoke_steps=0):
    if smoke_steps:
        return int(smoke_steps)
    if not train_size or train_size < 1:
        return 0
    microbatches = math.ceil(train_size / max(int(train_batch), 1))
    updates_per_epoch = math.ceil(microbatches / max(int(grad_accum), 1))
    return max(1, math.ceil(updates_per_epoch * float(epochs)))


def make_training_args(smoke_steps=0, train_size=None):
    eval_steps = int(os.getenv("LIGHTON_BBOX_EVAL_STEPS", "100"))
    workers = int(os.getenv("LIGHTON_BBOX_DATALOADER_WORKERS", "0"))
    strategy = os.getenv("LIGHTON_BBOX_EVAL_STRATEGY", "epoch").strip().lower()
    if strategy not in {"steps", "epoch"}:
        raise ValueError("LIGHTON_BBOX_EVAL_STRATEGY must be steps or epoch.")
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    train_batch = int(os.getenv("LIGHTON_BBOX_TRAIN_BATCH", "2"))
    grad_accum = int(os.getenv("LIGHTON_BBOX_GRAD_ACCUM", "4"))
    epochs = float(os.getenv("LIGHTON_BBOX_EPOCHS", "3"))
    total_update_steps = estimate_total_update_steps(
        train_size,
        train_batch,
        grad_accum,
        epochs,
        smoke_steps=smoke_steps,
    )
    explicit_warmup_steps = os.getenv("LIGHTON_BBOX_WARMUP_STEPS")
    if explicit_warmup_steps is not None:
        warmup_steps = max(0, int(explicit_warmup_steps))
    else:
        warmup_ratio = float(os.getenv("LIGHTON_BBOX_WARMUP_RATIO", "0.05"))
        warmup_steps = math.ceil(total_update_steps * max(0.0, warmup_ratio)) if total_update_steps else 0
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(OUTPUT_DIR / "tensorboard"))
    kwargs = {
        "output_dir": str(OUTPUT_DIR),
        "learning_rate": float(os.getenv("LIGHTON_BBOX_LR", "1e-5")),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": train_batch,
        "per_device_eval_batch_size": int(os.getenv("LIGHTON_BBOX_EVAL_BATCH", "2")),
        "gradient_accumulation_steps": grad_accum,
        "gradient_checkpointing": env_bool("LIGHTON_BBOX_GRADIENT_CHECKPOINTING", False),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": os.getenv(
            "LIGHTON_BBOX_OPTIM",
            "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        ),
        "bf16": bf16,
        "fp16": False,
        "tf32": torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        "logging_steps": int(os.getenv("LIGHTON_BBOX_LOGGING_STEPS", "5")),
        "logging_first_step": True,
        "logging_nan_inf_filter": False,
        "disable_tqdm": True,
        "run_name": os.getenv("LIGHTON_BBOX_RUN_NAME", "lighton-bbox"),
        "eval_strategy": strategy,
        "eval_steps": eval_steps,
        "save_strategy": strategy,
        "save_steps": eval_steps,
        "save_total_limit": int(os.getenv("LIGHTON_BBOX_SAVE_TOTAL_LIMIT", "3")),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_combined_score" if GEN_EVAL_MAX_SAMPLES else "eval_loss",
        "greater_is_better": bool(GEN_EVAL_MAX_SAMPLES),
        "remove_unused_columns": False,
        "report_to": os.getenv("LIGHTON_BBOX_REPORT_TO", "none"),
        "predict_with_generate": False,
        "prediction_loss_only": True,
        "label_names": ["labels"],
        "dataloader_num_workers": workers,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "dataloader_persistent_workers": workers > 0 and env_bool("LIGHTON_BBOX_PERSISTENT_WORKERS", False),
        "dataloader_prefetch_factor": int(os.getenv("LIGHTON_BBOX_PREFETCH_FACTOR", "1")) if workers else None,
        "torch_compile": env_bool("LIGHTON_BBOX_TORCH_COMPILE", False),
        "lr_scheduler_type": os.getenv("LIGHTON_BBOX_LR_SCHEDULER", "cosine"),
        "warmup_steps": warmup_steps,
        "weight_decay": float(os.getenv("LIGHTON_BBOX_WEIGHT_DECAY", "0.01")),
        "max_grad_norm": float(os.getenv("LIGHTON_BBOX_MAX_GRAD_NORM", "1.0")),
        "seed": RANDOM_SEED,
        "data_seed": RANDOM_SEED,
    }
    if kwargs["torch_compile"]:
        kwargs["torch_compile_backend"] = os.getenv("LIGHTON_BBOX_TORCH_COMPILE_BACKEND", "inductor")
        kwargs["torch_compile_mode"] = os.getenv("LIGHTON_BBOX_TORCH_COMPILE_MODE", "default")
    fields = Seq2SeqTrainingArguments.__dataclass_fields__
    if env_bool("LIGHTON_BBOX_GROUP_BY_LENGTH", True):
        kwargs["length_column_name"] = "length"
        kwargs["train_sampling_strategy" if "train_sampling_strategy" in fields else "group_by_length"] = (
            "group_by_length" if "train_sampling_strategy" in fields else True
        )
    if smoke_steps:
        kwargs.update(max_steps=smoke_steps, eval_strategy="no", save_strategy="no", load_best_model_at_end=False)
    try:
        return Seq2SeqTrainingArguments(**kwargs)
    except TypeError as exc:
        if "eval_strategy" not in str(exc):
            raise
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return Seq2SeqTrainingArguments(**kwargs)


def preflight_trainer(trainer):
    """Fail before step 1 if the eval sampler/collator path is incompatible."""
    print("Preflight: validating training/evaluation dataloaders...", flush=True)
    train_loader = trainer.get_train_dataloader()
    if len(train_loader) <= 0:
        raise RuntimeError("Training dataloader is empty.")
    if trainer.eval_dataset is not None:
        if len(trainer.eval_dataset) <= 0:
            raise RuntimeError("Evaluation dataloader is empty.")
        # Exercise the exact collator path without spinning up disposable
        # multiprocessing DataLoader workers. PyTorch can emit noisy worker-abort
        # messages when a one-batch preflight iterator is immediately destroyed.
        eval_batch_size = max(1, int(trainer.args.per_device_eval_batch_size))
        features = [
            trainer.eval_dataset[index]
            for index in range(min(eval_batch_size, len(trainer.eval_dataset)))
        ]
        batch = trainer.data_collator(features)
        required = ("input_ids", "attention_mask", "labels")
        if not hasattr(batch, "keys") or any(key not in batch for key in required):
            raise RuntimeError(
                "Evaluation collator returned an invalid batch; expected mapping-like "
                "input_ids/attention_mask/labels outputs."
            )
        for key in required:
            value = batch[key]
            if not torch.is_tensor(value) or value.ndim < 2 or value.shape[0] <= 0:
                raise RuntimeError(f"Evaluation collator returned invalid tensor for {key}: {type(value).__name__}.")
        if batch["input_ids"].shape != batch["attention_mask"].shape or batch["labels"].shape != batch["input_ids"].shape:
            raise RuntimeError("Evaluation collator returned inconsistent input/attention/label shapes.")
        supervised = int(batch["labels"].ne(-100).sum())
        if supervised <= 0:
            raise RuntimeError("Evaluation collator produced no supervised assistant tokens.")
        del batch, features
    del train_loader
    gc.collect()
    print("Preflight passed.", flush=True)


def merge_and_save(model, processor, train_mode: str):
    print("Saving final model...", flush=True)
    if train_mode.startswith("lora"):
        print("Merging LoRA weights into the base model...", flush=True)
        model = model.merge_and_unload(safe_merge=True)
    configure_generation(model, processor)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINAL_DIR, safe_serialization=True)
    processor.save_pretrained(FINAL_DIR)
    print(f"Final model saved to {FINAL_DIR}", flush=True)
    return model


def save_benchmark(path, model_id, metrics, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "metrics_version": 2,
        "base_model": MODEL_ID,
        # LightOn-BBox is conditioned on the image only. Keep the textual shared
        # prompt as output-contract metadata, never as an injected user prompt.
        "conditioning": "image_only",
        "prompt": "",
        "output_contract": OUTPUT_CONTRACT,
        "bbox_norm_scale": BBOX_NORM_SCALE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "metrics": metrics,
        "results": results,
    }
    atomic_json(path, payload)
    print(f"Benchmark saved to {path}", flush=True)


def release_model(model):
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_final_lighton_benchmark(model, processor, test_dataset):
    metrics, results = run_generation_benchmark(
        model,
        processor,
        test_dataset,
        split_name="test",
        generator=generate_lighton_prediction,
        model_label="LIGHTON FINAL",
        max_samples=FINAL_TEST_MAX_SAMPLES or None,
        worst_count=20,
        batch_generator=generate_lighton_predictions,
        batch_size=int(os.getenv("LIGHTON_BBOX_GEN_BATCH", "1")),
    )
    save_benchmark(
        FINAL_DIR / "benchmark_lighton_bbox.json",
        os.getenv("HF_REPO", "Remidesbois/LightonOCR-2-1b-poneglyph-bbox"),
        metrics,
        results,
    )
    if metrics["generation_errors"] and env_bool("LIGHTON_BBOX_FAIL_ON_GENERATION_ERROR", True):
        raise RuntimeError("Held-out generation failed on one or more pages; see benchmark_lighton_bbox.json.")
    return metrics, results


def benchmark_existing_model(model_path: str):
    processor = configure_processor(model_path)
    model = load_lighton_model(model_path).eval()
    configure_generation(model, processor)
    test_dataset = prepare_dataset(TEST_FILE, "test")
    run_final_lighton_benchmark(model, processor, test_dataset)
    release_model(model)


def resolve_resume_checkpoint(requested):
    if str(requested).strip().lower() in {"none", "", "0", "false", "off"}:
        if list(OUTPUT_DIR.glob("checkpoint-*")) or (FINAL_DIR / "config.json").exists():
            raise FileExistsError("Output already contains a training run. Use --resume auto or a new LIGHTON_BBOX_OUTPUT_DIR.")
        return None
    checkpoint = get_last_checkpoint(str(OUTPUT_DIR)) if requested == "auto" and OUTPUT_DIR.exists() else requested
    if requested == "auto" and (not OUTPUT_DIR.exists() or not checkpoint):
        if (FINAL_DIR / "config.json").exists():
            raise FileExistsError("A final model exists but no resumable checkpoint was found. Use a new output directory.")
        return None
    path = Path(checkpoint)
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete resume checkpoint {path}: missing {', '.join(missing)}. Choose an earlier complete checkpoint.")
    return str(path)


def main():
    global OUTPUT_DIR, FINAL_DIR
    args = parse_args()
    if args.smoke_steps < 0:
        raise ValueError("--smoke-steps must be non-negative.")
    if sum((args.merge_only, args.benchmark_only, args.diagnose, bool(args.smoke_steps))) > 1:
        raise ValueError("Choose only one of --merge-only, --benchmark-only, --diagnose and --smoke-steps.")
    profile = configure_torch_runtime(args.profile)
    set_seed(RANDOM_SEED)
    training = not (args.merge_only or args.benchmark_only or args.diagnose)
    if training and (int(os.getenv("WORLD_SIZE", "1")) > 1 or torch.cuda.device_count() > 1):
        raise ValueError("This trainer targets one GPU. Select one with CUDA_VISIBLE_DEVICES before launching.")
    runtime = runtime_diagnostics(profile, require_cuda=training)
    if args.diagnose:
        return
    if args.smoke_steps:
        OUTPUT_DIR = OUTPUT_DIR / f"smoke-{time.time_ns()}"
        FINAL_DIR = OUTPUT_DIR / "final_merged"
        args.resume = "none"

    if args.benchmark_only:
        benchmark_existing_model(args.model_path or str(FINAL_DIR))
        return

    resume_checkpoint = resolve_resume_checkpoint(args.resume) if training else None
    has_processor = resume_checkpoint and any((Path(resume_checkpoint) / name).exists()
                                             for name in ("preprocessor_config.json", "processor_config.json"))
    processor = configure_processor(resume_checkpoint if has_processor else MODEL_ID)
    model = load_lighton_model(for_training=training)
    configure_generation(model, processor)

    if args.merge_only:
        from peft import PeftModel

        checkpoint = args.model_path or find_best_checkpoint(OUTPUT_DIR)
        if not checkpoint:
            raise RuntimeError("No LoRA checkpoint found to merge.")
        print(f"Loading LoRA checkpoint: {checkpoint}", flush=True)
        model = PeftModel.from_pretrained(model, checkpoint)
        merge_and_save(model, processor, "lora")
        return

    train_dataset = prepare_dataset(TRAIN_FILE, "train", processor=processor)
    val_dataset = prepare_dataset(VAL_FILE, "val")
    test_dataset = prepare_dataset(TEST_FILE, "test") if not args.smoke_steps else None
    trainer_val_dataset = evaluation_subset(val_dataset, LOSS_EVAL_MAX_SAMPLES)
    if len(trainer_val_dataset) != len(val_dataset):
        print(
            f"Training-time eval subset: {len(trainer_val_dataset)}/{len(val_dataset)} pages "
            f"(full held-out test remains unchanged).",
            flush=True,
        )

    model, train_mode = configure_trainable_model(model, resume_checkpoint=resume_checkpoint)
    hardware_profile = calibrate_rtx5090_batch(
        model,
        processor,
        train_dataset,
        profile,
        resume_checkpoint=resume_checkpoint,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    monitor = TrainingMonitorCallback(OUTPUT_DIR, os.getenv("LIGHTON_BBOX_RUN_NAME", f"LightOn BBox / {profile}"))
    callbacks = [monitor]
    patience = int(os.getenv("LIGHTON_BBOX_EARLY_STOPPING_PATIENCE", "2"))
    if patience > 0 and not args.smoke_steps:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = PromptOnlyEvalTrainer(
        model=model,
        args=make_training_args(args.smoke_steps, train_size=len(train_dataset)),
        train_dataset=train_dataset,
        eval_dataset=trainer_val_dataset,
        data_collator=LightOnBBoxCollator(processor),
        callbacks=callbacks,
        processor=processor,
        processing_class=processor,
        gen_eval_max_samples=GEN_EVAL_MAX_SAMPLES,
    )
    trainer.remove_callback(PrinterCallback)
    preflight_trainer(trainer)
    manifest = {"runtime": runtime, "model_id": MODEL_ID, "train_mode": train_mode,
                "resume_checkpoint": resume_checkpoint, "training_arguments": trainer.args.to_dict(),
                "train_samples": len(train_dataset), "val_samples": len(val_dataset),
                "trainer_eval_samples": len(trainer_val_dataset),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "model_revision": getattr(model.config, "_commit_hash", None),
                "trim_logits": trainer.trim_logits, "hardware_profile": hardware_profile,
                "metrics_version": 2}
    atomic_json(OUTPUT_DIR / "run_manifest.json", manifest)
    state = trainer.state
    try:
        print(f"Starting LightOn bbox fine-tuning; profile={profile}, resume={resume_checkpoint or 'none'}", flush=True)
        print(f"Local dashboard: {OUTPUT_DIR / 'training_dashboard.html'}", flush=True)
        result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        state = trainer.state
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        if args.smoke_steps:
            monitor.close("smoke_complete", state)
            print("Smoke run complete. No merge, final benchmark or upload was performed.", flush=True)
            return
        print(f"Best checkpoint: {state.best_model_checkpoint or find_best_checkpoint(OUTPUT_DIR)}", flush=True)
        trained_model = trainer.model
        # Free optimizer/moment tensors before merging and running autoregressive tests.
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        monitor.summary["status"] = "exporting"
        monitor._persist("export_start")
        final_model = merge_and_save(trained_model, processor, train_mode)
        print("Running held-out LightOn bbox benchmark...", flush=True)
        run_final_lighton_benchmark(final_model, processor, test_dataset)
        release_model(final_model)
        final_model = trained_model = model = None
        gc.collect()
        monitor.close("complete", state)
    except KeyboardInterrupt:
        monitor.close("interrupted", trainer.state if "trainer" in locals() else state)
        raise
    except BaseException:
        monitor.close("failed", trainer.state if "trainer" in locals() else state)
        raise


if __name__ == "__main__":
    main()
