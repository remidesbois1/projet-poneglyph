"""Training-only fusions; inference and checkpoint state dictionaries stay native."""

import inspect
import types

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


def install_acceleration(model):
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

    apply_liger_kernel_to_qwen3_5(model=model, rope=False, rms_norm=True, swiglu=True,
                                cross_entropy=False, fused_linear_cross_entropy=False)
    native_forward = model.forward
    fused_ce = LigerFusedLinearCrossEntropyLoss(reduction="sum", accum_dtype=torch.float32)

    def training_forward(self, *args, ocr_positions=None, ocr_targets=None, ocr_weights=None, **kwargs):
        if ocr_positions is None:
            return native_forward(*args, **kwargs)
        kwargs.pop("labels", None)
        kwargs.pop("logits_to_keep", None)
        kwargs["use_cache"] = False
        output = self.model(*args, **kwargs)
        hidden = output.last_hidden_state
        selected = hidden.reshape(-1, hidden.shape[-1]).index_select(0, ocr_positions).contiguous()
        # Do not allocate [batch, full visual+text sequence, 248320] logits.
        # Liger chunks even the assistant-only projection and accumulates in FP32.
        if ocr_weights is None:
            loss = fused_ce(self.lm_head.weight, selected, ocr_targets)
        else:
            # Split by token weight so each fused backward is correctly weighted,
            # including the tied LM-head gradient; no full-vocabulary logits.
            loss = selected.new_zeros((), dtype=torch.float32)
            for factor in torch.unique(ocr_weights).tolist():
                indices = ocr_weights.eq(factor).nonzero(as_tuple=True)[0]
                loss = loss + factor * fused_ce(self.lm_head.weight, selected.index_select(0, indices),
                                               ocr_targets.index_select(0, indices))
        return CausalLMOutputWithPast(loss=loss)

    training_forward.__signature__ = inspect.signature(type(model).forward)
    model.forward = types.MethodType(training_forward, model)
    return {"assistant_only_fused_linear_ce": True, "ce_accumulation_dtype": "float32",
            "liger_rmsnorm_swiglu": True, "linear_attention": "FLA Gated DeltaNet + SM120 causal-conv1d",
            "full_attention": "sdpa", "torch_compile": False}


def packed_targets(batch):
    labels = batch["labels"]
    rows, cols = labels[:, 1:].ne(-100).nonzero(as_tuple=True)
    if labels[:, 1:].ne(-100).sum(dim=1).eq(0).any():
        raise ValueError("A page has no supervised assistant tokens")
    batch["ocr_positions"] = rows * labels.shape[1] + cols
    batch["ocr_targets"] = labels[rows, cols + 1]
    if "ocr_token_weights" in batch:
        batch["ocr_weights"] = batch.pop("ocr_token_weights")[rows, cols + 1]
    return batch
