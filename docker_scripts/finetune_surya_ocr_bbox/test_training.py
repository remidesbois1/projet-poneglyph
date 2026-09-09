"""CPU regression tests; no model download, CUDA, credentials or training job.

AST loading isolates the numerical/data-path functions from optional Hugging Face
and database imports. These tests do NOT replace the real-model --smoke-steps run.
Run: python -m unittest discover -s . -p 'test_*.py' -v
"""

import ast
import contextlib
import inspect
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from training_monitor import TrainingMonitorCallback, atomic_json, duration, json_safe, write_dashboard


SOURCE = Path(__file__).with_name("train_surya_bbox.py")


def symbols(*names, **extra):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(selected) != len(names):
        raise AssertionError("A production symbol under test is missing.")
    scope = {"torch": torch, "F": F, "np": np, "os": os, "re": re, "Path": Path,
             "Image": Image, "inspect": inspect, "TrainingMonitorCallback": TrainingMonitorCallback,
             "BBOX_NORM_SCALE": 1000, "IOU_THRESHOLDS": (0.3, 0.5, 0.75, 0.9),
             "BBOX_PATTERN": re.compile(r"(.+?)\s*\[(\d+),(\d+),(\d+),(\d+)\]"), **extra}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names}), scope


class FakeProcessor:
    """Minimal image-expanding tokenizer, with PAD deliberately equal to EOS."""
    pad_token_id = eos_token_id = 9

    def __init__(self, side="left", invalid_prefix=False, invalid_suffix=False):
        self.padding_side = side
        self.tokenizer = self
        self.image_calls = 0
        self.invalid_prefix = invalid_prefix
        self.invalid_suffix = invalid_suffix

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "P:" if len(messages) == 1 else "P:" + messages[1]["content"][0]["text"]

    def __call__(self, text, images=None, **kwargs):
        items = [text] if isinstance(text, str) else text
        rows = [[1, 2] + ([3 + ord(char) % 5 for char in item[2:]] + [9] if item[2:] else []) for item in items]
        if images is None:
            if self.invalid_prefix:
                rows = [[99, *row[1:]] if len(row) == 2 else row for row in rows]
            return {"input_ids": rows}
        self.image_calls += 1
        rows = [[1, 2, 20, 20, 20, *row[2:]] for row in rows]
        if self.invalid_suffix:
            rows[0][-1] = 88
        size = max(map(len, rows))
        multiple = kwargs.get("pad_to_multiple_of", 1)
        size = (size + multiple - 1) // multiple * multiple
        padded, masks = [], []
        for row in rows:
            pads, zeros = [9] * (size - len(row)), [0] * (size - len(row))
            padded.append(pads + row if self.padding_side == "left" else row + pads)
            masks.append(zeros + [1]*len(row) if self.padding_side == "left" else [1]*len(row) + zeros)
        return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(masks)}


class CollatorTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.image = Path(self.folder.name) / "page.png"
        Image.new("RGB", (24, 40)).save(self.image)
        self.api, _ = symbols("process_batch", "SuryaBBoxCollator", "extract_reference_text", "messages_for_entry", "apply_template",
                              USER_PROMPT="Read the page", resolve_image_path=lambda *args: self.image)

    def test_single_image_processing_pass_and_supervised_eos(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                processor = FakeProcessor(side)
                batch = self.api.SuryaBBoxCollator(processor)([
                    {"image_file": "page.png", "assistant_text": "AB"},
                    {"image_file": "page.png", "assistant_text": "CDEF"}])
                self.assertEqual(processor.image_calls, 1)
                self.assertEqual(batch["logits_to_keep"], 6 if side == "left" else 0)
                for index, answer in enumerate(("AB", "CDEF")):
                    actual = batch["labels"][index]
                    self.assertEqual(actual[actual.ne(-100)].tolist(), [3 + ord(c) % 5 for c in answer] + [9])
                    self.assertTrue(batch["labels"][index][batch["attention_mask"][index].eq(0)].eq(-100).all())

    def test_rejects_template_prefix_or_suffix_mismatch(self):
        for kwargs in ({"invalid_prefix": True}, {"invalid_suffix": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.api.SuryaBBoxCollator(FakeProcessor(**kwargs))([{"image_file": "page.png", "assistant_text": "ABC"}])

    def test_rejects_empty_batch(self):
        with self.assertRaises(ValueError):
            self.api.SuryaBBoxCollator(FakeProcessor())([])

    def test_reference_can_come_from_messages(self):
        batch = self.api.SuryaBBoxCollator(FakeProcessor())([{
            "image_file": "page.png", "messages": [{"role": "assistant", "content": [{"text": "AB"}]}]}])
        self.assertEqual(int(batch["labels"].ne(-100).sum()), 3)


class TrainerBase:
    def __init__(self, model=None, **kwargs):
        self.model = model

    def evaluation_loop(self, *args, **kwargs):
        return SimpleNamespace(metrics={"eval_loss": 0.25})

    def _get_eval_sampler(self, eval_dataset):
        return "base-sampler"


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(5, 11)

    def forward(self, features, logits_to_keep=0, use_cache=False):
        return SimpleNamespace(logits=self.projection(features[:, -logits_to_keep:, :]))


class LossAndEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.api, self.scope = symbols("env_bool", "PromptOnlyEvalTrainer", Seq2SeqTrainer=TrainerBase)
        torch.manual_seed(42)
        self.model = TinyModel()
        self.trainer = self.api.PromptOnlyEvalTrainer(model=self.model)
        self.features = torch.randn(2, 9, 5)
        self.labels = torch.tensor([[-100]*6 + [2, 3, 4], [-100]*4 + [1, 2, 3, 4, 5]])

    def test_trimmed_loss_and_gradients_match_full_head(self):
        losses, grads = [], []
        for trim in (False, True):
            self.model.zero_grad(set_to_none=True)
            self.trainer.trim_logits = trim
            loss = self.trainer.compute_loss(self.model, {"features": self.features, "labels": self.labels, "logits_to_keep": 6})
            loss.backward()
            losses.append(loss.detach())
            grads.append(self.model.projection.weight.grad.clone())
        torch.testing.assert_close(losses[0], losses[1])
        torch.testing.assert_close(grads[0], grads[1])

    def test_unequal_microbatches_use_global_supervised_token_count(self):
        denominator = self.labels[:, 1:].ne(-100).sum()
        full_loss = self.trainer.compute_loss(self.model, {"features": self.features, "labels": self.labels, "logits_to_keep": 6})
        full_loss.backward()
        expected = self.model.projection.weight.grad.clone()
        self.model.zero_grad(set_to_none=True)
        for i, keep in enumerate((4, 6)):
            loss = self.trainer.compute_loss(self.model, {"features": self.features[i:i+1], "labels": self.labels[i:i+1], "logits_to_keep": keep}, num_items_in_batch=denominator)
            loss.backward()
        torch.testing.assert_close(expected, self.model.projection.weight.grad)
        self.assertTrue(self.trainer.model_accepts_loss_kwargs)
        self.assertTrue(self.trainer._loss_shifts_labels)

    def test_generation_metrics_exist_when_evaluation_loop_returns(self):
        self.scope.update(run_generation_benchmark=Mock(return_value=({"combined_score": 0.8, "generation_errors": 0}, [])),
                          generate_surya_prediction=Mock(), generate_surya_predictions=Mock(),
                          save_benchmark=Mock(), MODEL_ID="test")
        self.trainer.callback_handler = SimpleNamespace(callbacks=[])
        self.trainer.args = SimpleNamespace(output_dir="unused")
        self.trainer.compute_loss_context_manager = contextlib.nullcontext
        output = self.trainer.evaluation_loop(SimpleNamespace(dataset=[{"split": "val"}]), "test")
        self.assertEqual(output.metrics, {"eval_loss": 0.25, "eval_combined_score": 0.8, "eval_generation_errors": 0})

    def test_loss_only_evaluation_skips_generation(self):
        self.trainer.gen_eval_max_samples = 0
        self.assertEqual(self.trainer.evaluation_loop(SimpleNamespace(dataset=[{}]), "test").metrics, {"eval_loss": 0.25})

    def test_grouped_training_forces_sequential_eval_sampler(self):
        self.trainer.args = SimpleNamespace(train_sampling_strategy="group_by_length")
        dataset = [1, 2, 3]
        sampler = self.trainer._get_eval_sampler(dataset)
        self.assertIsInstance(sampler, torch.utils.data.SequentialSampler)
        self.assertEqual(list(sampler), [0, 1, 2])

    def test_non_grouped_eval_sampler_delegates_to_transformers(self):
        self.trainer.args = SimpleNamespace(train_sampling_strategy="random")
        self.assertEqual(self.trainer._get_eval_sampler([1]), "base-sampler")

    def test_preflight_accepts_transformers_mapping_like_batch(self):
        api, _ = symbols("preflight_trainer", gc=SimpleNamespace(collect=lambda: None))

        class BatchLike:
            def __init__(self, values):
                self.values = values

            def keys(self):
                return self.values.keys()

            def __contains__(self, key):
                return key in self.values

            def __getitem__(self, key):
                return self.values[key]

        batch = BatchLike({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[-100, 2, 3]]),
        })
        trainer = SimpleNamespace(
            eval_dataset=[{}],
            get_train_dataloader=lambda: [object()],
            get_eval_dataloader=lambda: [batch],
        )
        api.preflight_trainer(trainer)


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, char in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (char != other)))
        previous = current
    return previous[-1]


class MetricsAndRuntimeTests(unittest.TestCase):
    def test_generation_special_tokens_align_nested_text_config(self):
        api, _ = symbols("valid_token_id", "configure_generation", MAX_NEW_TOKENS=2048)

        class Tokenizer:
            eos_token_id = 2
            pad_token_id = 0

            def __len__(self):
                return 65425

        text_config = SimpleNamespace(eos_token_id=248044, pad_token_id=None)
        config = SimpleNamespace(
            use_cache=False,
            get_text_config=lambda: text_config,
        )
        generation_config = SimpleNamespace(
            do_sample=True,
            max_new_tokens=None,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_length=128,
            eos_token_id=None,
            pad_token_id=None,
        )
        model = SimpleNamespace(config=config, generation_config=generation_config)
        processor = SimpleNamespace(tokenizer=Tokenizer())

        api.configure_generation(model, processor)

        self.assertEqual(model.generation_config.eos_token_id, 2)
        self.assertEqual(model.generation_config.pad_token_id, 0)
        self.assertEqual(text_config.eos_token_id, 2)
        self.assertEqual(text_config.pad_token_id, 0)

    def setUp(self):
        self.api, self.scope = symbols("parse_bbox_output", "normalize_prediction_text", "compute_iou", "compute_giou", "match_predictions_to_gt", "compute_metrics_from_results",
                                      levenshtein_distance=edit_distance, safe_wer=lambda a, b: float(a != b))

    def metrics(self, gt, pred, error=None):
        return self.api.compute_metrics_from_results([{
            "gt_items": self.api.parse_bbox_output(gt), "pred_items": self.api.parse_bbox_output(pred),
            "gt_text": gt, "pred_text": pred, "error": error}])

    def test_empty_page_is_not_an_ocr_failure(self):
        result = self.metrics("", "")
        self.assertEqual(result["cer"], 0)
        self.assertEqual(result["f1@0_5"], 1)

    def test_error_on_empty_page_is_not_a_success(self):
        result = self.metrics("", "", error="generation failed")
        self.assertEqual(result["generation_errors"], 1)
        self.assertEqual(result["f1@0_5"], 0)

    def test_wrongly_placed_boxes_do_not_get_detection_credit(self):
        result = self.metrics("A [0,0,20,20]", "A [500,500,600,600]")
        self.assertEqual(result["avg_detection_rate"], 0)

    def test_cer_above_one_is_visible_but_score_is_bounded(self):
        result = self.metrics("A [0,0,20,20]", "ABCDEFGHIJ [0,0,20,20]")
        self.assertGreater(result["cer"], 1)
        self.assertGreaterEqual(result["combined_score"], 0)
        self.assertLessEqual(result["combined_score"], 1)

    def test_perfect_prediction(self):
        result = self.metrics("ABC [0,0,20,20]", "ABC [0,0,20,20]")
        self.assertAlmostEqual(result["combined_score"], 1.0)
        self.assertEqual(result["mean_iou"], 1.0)

    def test_missing_prediction(self):
        result = self.metrics("ABC [0,0,20,20]", "")
        self.assertEqual(result["combined_score"], 0.0)
        self.assertEqual(result["cer"], 1.0)

    def test_cache_and_mode_restored_on_generation_exception(self):
        model = SimpleNamespace(config=SimpleNamespace(use_cache=False), training=True)
        model.train = lambda value: setattr(model, "training", value)
        def fail(*args, **kwargs):
            model.training = False
            raise ValueError("bad generation")
        api, _ = symbols("run_generation_benchmark", _run_generation_benchmark=fail)
        with self.assertRaises(ValueError):
            api.run_generation_benchmark(model)
        self.assertFalse(model.config.use_cache)
        self.assertTrue(model.training)

    def test_generation_benchmark_batches_and_reports_throughput(self):
        metrics_template = {
            "cer": 0.0,
            "wer": 0.0,
            "mean_iou": 1.0,
            "f1@0_5": 1.0,
            "avg_detection_rate": 1.0,
            "combined_score": 1.0,
            "avg_inference_time": 0.1,
            "per_sample": [],
        }
        api, _ = symbols(
            "benchmark_indices",
            "_run_generation_benchmark",
            RANDOM_SEED=42,
            random=__import__("random"),
            time=__import__("time"),
            parse_bbox_output=lambda _: [],
            extract_reference_text=lambda entry: entry["assistant_text"],
            compute_metrics_from_results=lambda _: dict(metrics_template),
        )
        model = torch.nn.Linear(1, 1)
        model.config = SimpleNamespace(use_cache=False)
        calls = []

        def batched(_model, _processor, entries):
            calls.append(len(entries))
            return [f"page-{entry['page_id']}" for entry in entries]

        dataset = [
            {"page_id": index, "image_file": f"{index}.jpg", "assistant_text": "A [0,0,1,1]"}
            for index in range(10)
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            metrics, results = api._run_generation_benchmark(
                model,
                None,
                dataset,
                split_name="val",
                generator=Mock(),
                model_label="TEST",
                batch_generator=batched,
                batch_size=4,
            )
        self.assertEqual(calls, [4, 4, 2])
        self.assertEqual(len(results), 10)
        self.assertEqual(metrics["generation_batch_size"], 4)
        self.assertGreater(metrics["pages_per_second"], 0)

    def test_generation_benchmark_halves_batch_after_oom(self):
        metrics_template = {
            "cer": 0.0,
            "wer": 0.0,
            "mean_iou": 1.0,
            "f1@0_5": 1.0,
            "avg_detection_rate": 1.0,
            "combined_score": 1.0,
            "avg_inference_time": 0.1,
            "per_sample": [],
        }
        api, _ = symbols(
            "benchmark_indices",
            "_run_generation_benchmark",
            RANDOM_SEED=42,
            random=__import__("random"),
            time=__import__("time"),
            parse_bbox_output=lambda _: [],
            extract_reference_text=lambda entry: entry["assistant_text"],
            compute_metrics_from_results=lambda _: dict(metrics_template),
        )
        model = torch.nn.Linear(1, 1)
        model.config = SimpleNamespace(use_cache=False)
        attempts = []

        def batched(_model, _processor, entries):
            attempts.append(len(entries))
            if len(entries) > 2:
                raise torch.cuda.OutOfMemoryError("synthetic")
            return ["ok"] * len(entries)

        dataset = [
            {"page_id": index, "image_file": f"{index}.jpg", "assistant_text": "A [0,0,1,1]"}
            for index in range(5)
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            metrics, results = api._run_generation_benchmark(
                model,
                None,
                dataset,
                split_name="val",
                generator=Mock(),
                model_label="TEST",
                batch_generator=batched,
                batch_size=8,
            )
        self.assertEqual(attempts[:3], [5, 4, 2])
        self.assertEqual(metrics["generation_batch_size"], 2)
        self.assertEqual(len(results), 5)

    def test_resume_requires_optimizer_and_scheduler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-3"
            checkpoint.mkdir()
            api, _ = symbols("resolve_resume_checkpoint", OUTPUT_DIR=root, FINAL_DIR=root / "final_merged", get_last_checkpoint=lambda _: str(checkpoint))
            with self.assertRaises(FileExistsError):
                api.resolve_resume_checkpoint("none")
            with self.assertRaises(ValueError):
                api.resolve_resume_checkpoint("auto")
            for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
                (checkpoint / name).write_text("{}")
            self.assertEqual(api.resolve_resume_checkpoint("auto"), str(checkpoint))

    def test_loss_only_selection_and_smoke_arguments(self):
        class Args:
            __dataclass_fields__ = {"train_sampling_strategy": None}
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        api, _ = symbols("env_bool", "estimate_total_update_steps", "make_training_args", Seq2SeqTrainingArguments=Args, OUTPUT_DIR=Path("unused"), GEN_EVAL_MAX_SAMPLES=0, RANDOM_SEED=42, math=__import__("math"))
        with patch.dict(os.environ, {"SURYA_BBOX_DATALOADER_WORKERS": "0"}):
            args = api.make_training_args(train_size=772)
            self.assertEqual(args.metric_for_best_model, "eval_loss")
            self.assertFalse(args.greater_is_better)
            self.assertIsNone(args.dataloader_prefetch_factor)
            self.assertEqual(args.warmup_steps, 30)
            smoke = api.make_training_args(3, train_size=772)
            self.assertEqual(smoke.max_steps, 3)
            self.assertEqual(smoke.save_strategy, "no")
            self.assertFalse(smoke.load_best_model_at_end)
            self.assertEqual(smoke.warmup_steps, 1)

    def test_training_step_estimate_matches_real_5090_profile(self):
        api, _ = symbols("estimate_total_update_steps", math=__import__("math"))
        self.assertEqual(api.estimate_total_update_steps(772, 8, 1, 6), 582)


class MonitorTests(unittest.TestCase):
    def state(self, step=0, rank_zero=True):
        return SimpleNamespace(global_step=step, max_steps=20, epoch=0, best_model_checkpoint=None, best_metric=None, is_world_process_zero=rank_zero)

    def test_redaction_strict_json_and_escaping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            atomic_json(path / "data.json", {"hub_token": "secret", "nested": {"api_key": "secret"}, "loss": float("nan"), "max_new_tokens": 2048})
            result = json.loads((path / "data.json").read_text())
            self.assertEqual(result["hub_token"], "[REDACTED]")
            self.assertEqual(result["nested"]["api_key"], "[REDACTED]")
            self.assertIsNone(result["loss"])
            self.assertEqual(result["max_new_tokens"], 2048)
            write_dashboard(path / "report.html", {"run_name": "<script>alert(1)</script>", "status": "complete"}, [])
            document = (path / "report.html").read_text()
            self.assertNotIn("<script>", document)
            self.assertIn("&lt;script&gt;", document)
            self.assertNotIn('http-equiv="refresh"', document)

    def test_sessions_append_and_resume_starts_at_current_step(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2):
                monitor = TrainingMonitorCallback(directory)
                state = self.state(10)
                monitor.on_train_begin(None, state, None)
                self.assertEqual(monitor.initial_step, 10)
                monitor.on_log(None, state, None, logs={"loss": 0.25})
                monitor.close("complete", state)
            records = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len({r["session_id"] for r in records}), 2)
            self.assertEqual(len(records), 6)

    def test_nonfinite_loss_is_recorded_then_stops(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            monitor = TrainingMonitorCallback(directory)
            with self.assertRaises(FloatingPointError):
                monitor.on_log(None, self.state(2), None, logs={"loss": float("inf")})
            self.assertIsNone(json.loads((Path(directory) / "training_summary.json").read_text())["loss"])

    def test_non_primary_rank_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = TrainingMonitorCallback(directory)
            monitor.on_train_begin(None, self.state(rank_zero=False), None)
            monitor.close("complete")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_duration_and_finite_scalars(self):
        self.assertEqual(duration(3661), "01:01:01")
        self.assertEqual(duration(None), "--")
        self.assertEqual(json_safe(torch.tensor(1.5)), 1.5)


if __name__ == "__main__":
    unittest.main()
