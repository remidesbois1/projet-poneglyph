"""Autograd-safe equivalents of the inference fusions, plus a sparse OCR head.

The desktop GEMV/Flash KV-cache kernels are batch-one, inference-only kernels.
Here large GEMMs stay on cuBLAS, while Liger fuses SwiGLU/RMSNorm/CE with backward.
Weights and saved model keys keep the standard Transformers format.
"""

import os
import types
import inspect
import sys

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


def weighted_ocr_loss(logits, targets, token_weights, fused=False):
    if fused:
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

        losses = LigerCrossEntropyLoss(reduction="none")(logits, targets)
    else:
        losses = F.cross_entropy(logits.float(), targets, reduction="none")
    return (losses.float() * token_weights.float()).sum()


def install_training_acceleration(model):
    if getattr(model, "_surya_training_acceleration", None) is not None:
        return model._surya_training_acceleration
    enabled = os.getenv("SURYA_LIGER", "1" if sys.platform == "linux" else "0").lower() not in {"0", "false", "off"}
    enabled = enabled and next(model.parameters()).is_cuda
    if enabled:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

        # Do not globally replace cross_entropy or change the multimodal forward.
        apply_liger_kernel_to_qwen3_5(
            model=model, rope=False, rms_norm=True, swiglu=True,
            cross_entropy=False, fused_linear_cross_entropy=False,
        )
    original_forward = model.forward

    def sparse_forward(self, *args, ocr_positions=None, ocr_targets=None,
                       ocr_token_weights=None, **kwargs):
        if ocr_positions is None:
            return original_forward(*args, **kwargs)
        kwargs.pop("logits_to_keep", None)
        kwargs.pop("labels", None)
        kwargs["use_cache"] = False
        outputs = self.model(*args, **kwargs)
        hidden = outputs.last_hidden_state
        selected = hidden.reshape(-1, hidden.shape[-1]).index_select(0, ocr_positions)
        logits = self.lm_head(selected)
        loss = weighted_ocr_loss(logits, ocr_targets, ocr_token_weights, fused=enabled)
        return CausalLMOutputWithPast(loss=loss)

    # Generation/Trainer inspect the public HF signature (e.g. logits_to_keep).
    sparse_forward.__signature__ = inspect.signature(type(model).forward)
    model.forward = types.MethodType(sparse_forward, model)
    model._surya_training_acceleration = {
        "packed_supervised_lm_head": True, "liger_swiglu_rmsnorm_ce": enabled,
        "linear_attention": "FLA chunk Gated DeltaNet + causal-conv1d",
        "full_attention": getattr(model.config, "_attn_implementation", "sdpa"),
        "inference_kernels": "Same fusion principles, with autograd; no batch-one decode patch in training",
    }
    return model._surya_training_acceleration


def add_packed_targets(batch, sample_weights):
    """Build sparse, shifted prediction positions on CPU, avoiding GPU .item()."""
    labels = batch["labels"]
    rows, columns = labels[:, 1:].ne(-100).nonzero(as_tuple=True)
    counts = labels[:, 1:].ne(-100).sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Every OCR example must contain a supervised next token.")
    if not torch.isfinite(sample_weights).all() or (sample_weights <= 0).any():
        raise ValueError("OCR sample weights must be positive and finite.")
    batch["ocr_positions"] = rows * labels.shape[1] + columns
    batch["ocr_targets"] = labels[rows, columns + 1]
    batch["ocr_token_weights"] = (sample_weights / counts)[rows]
    batch["sample_weights"] = sample_weights
    batch["sample_weight_sum"] = float(sample_weights.sum())
    return batch
