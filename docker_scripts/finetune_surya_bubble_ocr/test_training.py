"""Numerical and data-contract regressions. GPU kernel tests are explicitly opt-in."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image

import train_surya_bubble_ocr as training
from training_acceleration import add_packed_targets, install_training_acceleration, weighted_ocr_loss


class FakeProcessor:
    pad_token_id = eos_token_id = 9

    def __init__(self, side="left"):
        self.padding_side = side
        self.tokenizer = self

    def apply_chat_template(self, messages, **kwargs):
        return "P:" if len(messages) == 1 else "P:" + messages[1]["content"][0]["text"]

    def __call__(self, text, images=None, **kwargs):
        rows = [[1, 2] + ([3 + ord(c) % 5 for c in s[2:]] + [9] if s[2:] else []) for s in text]
        if images is None:
            return {"input_ids": rows}
        rows = [[1, 2, 20, 20, *r[2:]] for r in rows]
        width = max(map(len, rows)) + 3
        def pad(row, value):
            padding = [value] * (width - len(row))
            return padding + row if self.padding_side == "left" else row + padding
        return {"input_ids": torch.tensor([pad(r, 9) for r in rows]),
                "attention_mask": torch.tensor([pad([1]*len(r), 0) for r in rows])}


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 8)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.lm_head = torch.nn.Linear(8, 32, bias=False)
        self.config = SimpleNamespace(_attn_implementation="eager")

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids).last_hidden_state))


class DataAndLossTests(unittest.TestCase):
    def test_eos_and_exact_targets_for_both_padding_sides(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "crop.png"
            Image.new("RGB", (32, 32)).save(image)
            for side in ("left", "right"):
                with patch.object(training, "resolve_image_path", return_value=image):
                    batch = training.process_batch(
                        {"id": [1, 2], "image_file": ["crop.png"]*2, "text": ["A", "BCDE"]}, FakeProcessor(side))
                for row, ref in enumerate(("A", "BCDE")):
                    actual = batch["labels"][row]
                    self.assertEqual(actual[actual.ne(-100)].tolist(), [3 + ord(c) % 5 for c in ref] + [9])
                self.assertTrue(batch["labels"][batch["attention_mask"].eq(0)].eq(-100).all())
                self.assertEqual(batch["ocr_targets"].numel(), 7)
                torch.testing.assert_close(batch["ocr_token_weights"].sum(), batch["sample_weights"].sum())

    def test_sparse_head_matches_dense_loss_and_gradients_with_uneven_accumulation(self):
        torch.manual_seed(42)
        original = ToyModel()
        packed = copy.deepcopy(original)
        install_training_acceleration(packed)
        ids = torch.randint(0, 32, (3, 8))
        labels = ids.clone()
        labels[0, :6], labels[1, :2], labels[2, :4] = -100, -100, -100
        weights = torch.tensor([1.2, 1.0, 1.4])
        logits = original(ids).logits[:, :-1]
        targets = labels[:, 1:]
        token_loss = F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1), reduction="none").view_as(targets)
        expected = ((token_loss.sum(1) / targets.ne(-100).sum(1)) * weights).sum() / weights.sum()
        expected.backward()
        total = 0
        for start, stop in ((0, 2), (2, 3)):
            batch = add_packed_targets({"input_ids": ids[start:stop], "labels": labels[start:stop]}, weights[start:stop])
            loss = training.SuryaOCRTrainer.compute_loss(None, packed, batch, num_items_in_batch=weights.sum())
            total += loss.detach()
            loss.backward()
        torch.testing.assert_close(total, expected.detach())
        for a, b in zip(original.parameters(), packed.parameters()):
            torch.testing.assert_close(a.grad, b.grad)

    def test_rejects_empty_supervision_and_invalid_weights(self):
        with self.assertRaises(ValueError):
            add_packed_targets({"labels": torch.full((1, 4), -100)}, torch.ones(1))
        with self.assertRaises(ValueError):
            add_packed_targets({"labels": torch.tensor([[-100, 2, 3]])}, torch.tensor([float("nan")]))

    def test_generation_restores_model_state_after_failure(self):
        model = ToyModel().train()
        model.config.use_cache = False
        with patch.object(training, "_run_generation_benchmark", side_effect=RuntimeError("generation failed")):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                training.run_generation_benchmark(model)
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)

    def test_no_baseline_comparison_is_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            sample = {"reference": "A", "prediction": "A", "cer": 0., "levenshtein": 0}
            payload = training.save_benchmark(Path(folder)/"test.json", {"cer": 0.}, [sample])
            self.assertNotIn("baseline_comparisons", payload)


@unittest.skipUnless(os.getenv("SURYA_RUN_GPU_TESTS") == "1", "Set SURYA_RUN_GPU_TESTS=1 for CUDA backward tests")
class KernelTests(unittest.TestCase):
    def test_weighted_liger_ce_forward_backward(self):
        torch.manual_seed(42)
        logits = torch.randn(97, 65424, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        copy_logits = logits.detach().clone().requires_grad_()
        targets = torch.randint(0, 65424, (97,), device="cuda")
        weights = torch.rand(97, device="cuda") / 97
        expected = weighted_ocr_loss(logits, targets, weights)
        actual = weighted_ocr_loss(copy_logits, targets, weights, fused=True)
        expected.backward()
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=.001, atol=.001)
        torch.testing.assert_close(copy_logits.grad, logits.grad, rtol=.02, atol=2e-5)

    def test_swiglu_and_qwen_norm_backward(self):
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
        from liger_kernel.transformers.rms_norm import LigerRMSNormForQwen3Next
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
        torch.manual_seed(42)
        gate = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        up = torch.randn_like(gate, requires_grad=True)
        a, b = gate.detach().clone().requires_grad_(), up.detach().clone().requires_grad_()
        expected, actual = F.silu(gate)*up, LigerSiLUMulFunction.apply(a, b)
        grad = torch.randn_like(actual)
        expected.backward(grad)
        actual.backward(grad)
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
        for p, q in ((a, gate), (b, up)):
            torch.testing.assert_close(p.grad, q.grad, rtol=.03, atol=.025)
        norm = Qwen3_5RMSNorm(1024).to(device="cuda", dtype=torch.bfloat16)
        fused = LigerRMSNormForQwen3Next(1024).to(device="cuda", dtype=torch.bfloat16)
        fused.load_state_dict(norm.state_dict())
        x, y = gate.detach().clone().requires_grad_(), gate.detach().clone().requires_grad_()
        v, w = norm(x), fused(y)
        v.backward(grad)
        w.backward(grad)
        torch.testing.assert_close(v, w, rtol=.01, atol=.01)
        torch.testing.assert_close(x.grad, y.grad, rtol=.02, atol=.02)
        torch.testing.assert_close(norm.weight.grad, fused.weight.grad, rtol=.03, atol=.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
