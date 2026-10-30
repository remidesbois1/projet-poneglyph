"""Compatibility checks for optional Surya GPU kernels."""

import types
import unittest
from unittest.mock import patch

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

import local_ocr_server as server


class SuryaKernelPolicyTests(unittest.TestCase):
    def test_cpu_and_bbox_never_instantiate_cuda_only_kernels(self):
        names = ("FusedRMSNormGated", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")
        sentinels = {name: object() for name in names}
        with patch.multiple(qwen, **sentinels):
            for model_key, device, dtype in [(server.SURYA_MODEL_KEY, "cpu", torch.float32),
                                             (server.SURYA_BBOX_MODEL_KEY, "cuda", torch.bfloat16)]:
                with self.assertRaisesRegex(RuntimeError, "load failed"):
                    with server.surya_kernel_policy(model_key, device, dtype):
                        self.assertTrue(all(getattr(qwen, name) is None for name in names))
                        raise RuntimeError("load failed")
                self.assertTrue(all(getattr(qwen, name) is value for name, value in sentinels.items()))
            with server.surya_kernel_policy(server.SURYA_MODEL_KEY, "cuda", torch.bfloat16):
                self.assertTrue(all(getattr(qwen, name) is value for name, value in sentinels.items()))

    def test_failed_fla_model_can_restore_reference_operations_and_weights(self):
        parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
        attention = types.SimpleNamespace(chunk_gated_delta_rule=object(), recurrent_gated_delta_rule=object(),
                                          norm=types.SimpleNamespace(weight=parameter), head_v_dim=8,
                                          layer_norm_epsilon=1e-6)
        layer = types.SimpleNamespace(layer_type="linear_attention", linear_attn=attention)
        model = types.SimpleNamespace(model=types.SimpleNamespace(language_model=types.SimpleNamespace(layers=[layer])))
        self.assertTrue(server.restore_surya_torch_kernels(model))
        self.assertIs(attention.chunk_gated_delta_rule, qwen.torch_chunk_gated_delta_rule)
        self.assertIs(attention.recurrent_gated_delta_rule, qwen.torch_recurrent_gated_delta_rule)
        self.assertIsInstance(attention.norm, qwen.Qwen3_5RMSNormGated)
        self.assertIs(attention.norm.weight, parameter)
        self.assertFalse(server.restore_surya_torch_kernels(model))


if __name__ == "__main__":
    unittest.main()
