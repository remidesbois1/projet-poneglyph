"""Verify fused loss/backward against ordinary CE before the expensive run."""

import torch
import torch.nn.functional as F
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss


def main():
    torch.manual_seed(42)
    for dtype in (torch.float32, torch.bfloat16):
        hidden = torch.randn(17, 128, device="cuda", dtype=dtype, requires_grad=True)
        weight = torch.randn(513, 128, device="cuda", dtype=torch.float32, requires_grad=True)
        targets = torch.randint(513, (17,), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fused = LigerFusedLinearCrossEntropyLoss(reduction="sum", accum_dtype=torch.float32)(weight, hidden, targets) / 17
        fused.backward()
        hgrad, wgrad = hidden.grad.clone(), weight.grad.clone()
        hidden.grad, weight.grad = None, None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = F.cross_entropy(F.linear(hidden, weight).float(), targets)
        reference.backward()
        torch.testing.assert_close(fused, reference, rtol=.005, atol=.01)
        torch.testing.assert_close(hgrad.float(), hidden.grad.float(), rtol=.06, atol=.006)
        torch.testing.assert_close(wgrad.float(), weight.grad.float(), rtol=.06, atol=.006)
        print(f"Fused CE/backward verified: hidden={dtype}, FP32 weights, loss={fused.item():.6f}", flush=True)
        hidden.grad, weight.grad = None, None
        token_weights = torch.tensor([.2, 1., 2.] * 5 + [1., 2.], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fused = hidden.new_zeros((), dtype=torch.float32)
            for factor in torch.unique(token_weights).tolist():
                indices = token_weights.eq(factor).nonzero(as_tuple=True)[0]
                fused = fused + factor * LigerFusedLinearCrossEntropyLoss(reduction="sum", accum_dtype=torch.float32)(
                    weight, hidden.index_select(0, indices), targets.index_select(0, indices))
            fused = fused / token_weights.sum()
        fused.backward()
        hgrad, wgrad = hidden.grad.clone(), weight.grad.clone()
        hidden.grad, weight.grad = None, None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = (F.cross_entropy(F.linear(hidden, weight).float(), targets, reduction="none") * token_weights).sum() / token_weights.sum()
        reference.backward()
        torch.testing.assert_close(fused, reference, rtol=.005, atol=.01)
        torch.testing.assert_close(hgrad.float(), hidden.grad.float(), rtol=.06, atol=.006)
        torch.testing.assert_close(wgrad.float(), weight.grad.float(), rtol=.06, atol=.006)
        print(f"Position-weighted fused CE/backward verified: {dtype}", flush=True)


if __name__ == "__main__":
    main()
