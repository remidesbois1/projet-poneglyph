"""Trainable PyTorch execution of an ONNX graph, with weight write-back.

Indra publishes rec.onnx (and its float weights) but no training code.  Rather
than re-implementing the architecture by hand, this module executes the ONNX
graph node by node with torch ops: float initializers become parameters, so the
exact published graph can be fine-tuned, and the trained values are written back
into a copy of the same graph.  The exported model is therefore structurally
identical to rec.onnx (same inputs, outputs, ops), which keeps it a drop-in
replacement for indra_ocr and onnxruntime(-web).

Only the operators used by Indra's rec.onnx and det.onnx (opset 17) are implemented.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

_DTYPES = {1: torch.float32, 6: torch.int32, 7: torch.int64, 9: torch.bool, 10: torch.float16, 11: torch.float64}
_INT64_MAX = 2**62


def _key(name: str) -> str:
    return "p_" + "".join(c if c.isalnum() else "_" for c in name)


def _attrs(node) -> dict:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


class OnnxModule(torch.nn.Module):
    def __init__(self, path: str | Path, outputs: list[str] | None = None, frozen: set[str] = frozenset()):
        super().__init__()
        model = onnx.load(str(path))
        g = model.graph
        self.input_names = [i.name for i in g.input]
        self.output_names = outputs or [o.name for o in g.output]
        # Keep only the nodes the requested outputs depend on.
        need, nodes = set(self.output_names), []
        for node in reversed(g.node):
            if any(o in need for o in node.output):
                nodes.append(node)
                need.update(i for i in node.input if i)
        nodes.reverse()
        bn_stats = {n.input[k] for n in nodes if n.op_type == "BatchNormalization" for k in (3, 4)}
        self.params = torch.nn.ParameterDict()
        self.param_names: dict[str, str] = {}  # onnx initializer -> parameter key
        self.consts: dict[str, torch.Tensor] = {}
        for t in g.initializer:
            if t.name not in need:
                continue
            arr = numpy_helper.to_array(t).copy()
            if arr.dtype == np.float32 and t.name not in bn_stats and t.name not in frozen:
                self.params[_key(t.name)] = torch.nn.Parameter(torch.from_numpy(arr))
                self.param_names[t.name] = _key(t.name)
            else:
                self.consts[t.name] = torch.from_numpy(arr)
        self.nodes = []
        for n in nodes:
            a = _attrs(n)
            if n.op_type == "Constant":
                self.consts[n.output[0]] = torch.from_numpy(numpy_helper.to_array(a["value"]).copy())
                continue
            self.nodes.append((n.op_type, list(n.input), list(n.output), a))
        for i, (k, v) in enumerate(self.consts.items()):
            self.register_buffer(f"c{i}", v, persistent=False)
        self._const_keys = {k: f"c{i}" for i, k in enumerate(self.consts)}

    # ------------------------------------------------------------------
    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        env: dict[str, torch.Tensor] = dict(inputs)
        for name, key in self._const_keys.items():
            env[name] = getattr(self, key)
        for name, key in self.param_names.items():
            env[name] = self.params[key]
        for op, ins, outs, a in self.nodes:
            x = [env[i] if i else None for i in ins]
            res = getattr(self, "_" + op)(x, a)
            if not isinstance(res, tuple):
                res = (res,)
            for o, v in zip(outs, res):
                env[o] = v
        return {o: env[o] for o in self.output_names}

    # --- operators -------------------------------------------------------
    @staticmethod
    def _ints(t) -> list[int]:
        return [int(v) for v in t.reshape(-1).tolist()]

    def _Slice(self, x, a):
        data, starts, ends = x[0], self._ints(x[1]), self._ints(x[2])
        axes = self._ints(x[3]) if len(x) > 3 and x[3] is not None else list(range(len(starts)))
        steps = self._ints(x[4]) if len(x) > 4 and x[4] is not None else [1] * len(starts)
        idx = [slice(None)] * data.dim()
        for s, e, ax, st in zip(starts, ends, axes, steps):
            if st != 1:
                raise NotImplementedError("Slice step")
            idx[ax] = slice(s, min(e, _INT64_MAX))
        return data[tuple(idx)]

    def _Conv(self, x, a):
        inp, w, b = x[0], x[1], x[2] if len(x) > 2 else None
        nd = w.dim() - 2
        pads = a.get("pads", [0] * 2 * nd)
        begin, end = pads[:nd], pads[nd:]
        kw = dict(stride=a.get("strides", [1] * nd), dilation=a.get("dilations", [1] * nd), groups=a.get("group", 1))
        if begin != end:
            inp = F.pad(inp, [p for pair in zip(reversed(begin), reversed(end)) for p in pair])
            begin = [0] * nd
        conv = F.conv1d if nd == 1 else F.conv2d
        return conv(inp, w, b, padding=begin, **kw)

    def _Squeeze(self, x, a):
        axes = self._ints(x[1]) if len(x) > 1 and x[1] is not None else None
        if axes is None:
            return x[0].squeeze()
        out = x[0]
        for ax in sorted((ax % out.dim() for ax in axes), reverse=True):
            out = out.squeeze(ax)
        return out

    def _Unsqueeze(self, x, a):
        out = x[0]
        rank = out.dim() + len(self._ints(x[1]))
        for ax in sorted(ax % rank for ax in self._ints(x[1])):
            out = out.unsqueeze(ax)
        return out

    def _HardSwish(self, x, a):
        return F.hardswish(x[0])

    def _Relu(self, x, a):
        return F.relu(x[0])

    def _Sigmoid(self, x, a):
        return torch.sigmoid(x[0])

    def _Exp(self, x, a):
        return torch.exp(x[0])

    def _Reciprocal(self, x, a):
        return torch.reciprocal(x[0])

    def _Softplus(self, x, a):
        return F.softplus(x[0])

    def _DepthToSpace(self, x, a):
        mode = a.get("mode", b"DCR")
        if (mode.decode() if isinstance(mode, bytes) else mode) != "CRD":
            raise NotImplementedError("DepthToSpace DCR")
        return F.pixel_shuffle(x[0], a["blocksize"])  # CRD == pixel_shuffle

    def _Resize(self, x, a):
        mode = a.get("mode", "nearest")
        mode = mode.decode() if isinstance(mode, bytes) else mode
        ctm = a.get("coordinate_transformation_mode", "half_pixel")
        ctm = ctm.decode() if isinstance(ctm, bytes) else ctm
        nm = a.get("nearest_mode", "round_prefer_floor")
        nm = nm.decode() if isinstance(nm, bytes) else nm
        if mode != "nearest" or ctm != "asymmetric" or nm != "floor" or len(x) < 4 or x[3] is None:
            raise NotImplementedError(f"Resize {mode}/{ctm}/{nm}")
        size = self._ints(x[3])[2:]
        return F.interpolate(x[0], size=size, mode="nearest")  # floor(dst * in/out): asymmetric + floor

    def _Shape(self, x, a):
        return torch.tensor(list(x[0].shape), dtype=torch.int64, device=x[0].device)

    def _Gather(self, x, a):
        data, idx, ax = x[0], x[1].long(), a.get("axis", 0)
        idx = torch.where(idx < 0, idx + data.shape[ax], idx)
        if idx.dim() == 0:
            return data.select(ax, int(idx))
        out = torch.index_select(data, ax, idx.reshape(-1))
        return out.reshape(data.shape[:ax] + idx.shape + data.shape[ax + 1:])

    def _Add(self, x, a):
        return x[0] + x[1]

    def _Sub(self, x, a):
        return x[0] - x[1]

    def _Mul(self, x, a):
        return x[0] * x[1]

    def _Div(self, x, a):
        if x[0].is_floating_point():
            return x[0] / x[1]
        return torch.div(x[0], x[1], rounding_mode="trunc")

    def _Mod(self, x, a):
        return torch.fmod(x[0], x[1]) if a.get("fmod", 0) else torch.remainder(x[0], x[1])

    def _Softmax(self, x, a):
        return torch.softmax(x[0], a.get("axis", -1))

    def _LogSoftmax(self, x, a):
        return torch.log_softmax(x[0], a.get("axis", -1))

    def _Cast(self, x, a):
        return x[0].to(_DTYPES[a["to"]])

    def _Concat(self, x, a):
        return torch.cat(x, a["axis"])

    def _Reshape(self, x, a):
        shape = self._ints(x[1])
        if not a.get("allowzero", 0):
            shape = [x[0].shape[i] if s == 0 else s for i, s in enumerate(shape)]
        return x[0].reshape(shape)

    def _ReduceSum(self, x, a):
        axes = self._ints(x[1]) if len(x) > 1 and x[1] is not None else []
        if not axes:
            return x[0] if a.get("noop_with_empty_axes", 0) else x[0].sum()
        return x[0].sum(dim=axes, keepdim=bool(a.get("keepdims", 1)))

    def _ReduceMax(self, x, a):
        axes = a.get("axes") or (self._ints(x[1]) if len(x) > 1 and x[1] is not None else list(range(x[0].dim())))
        return torch.amax(x[0], dim=axes, keepdim=bool(a.get("keepdims", 1)))

    def _ArgMax(self, x, a):
        if a.get("select_last_index", 0):
            raise NotImplementedError("ArgMax select_last_index")
        return torch.argmax(x[0], dim=a.get("axis", 0), keepdim=bool(a.get("keepdims", 1)))

    def _TopK(self, x, a):
        k = int(x[1].reshape(-1)[0])
        v, i = torch.topk(x[0], k, dim=a.get("axis", -1), largest=bool(a.get("largest", 1)), sorted=bool(a.get("sorted", 1)))
        return v, i

    def _Less(self, x, a):
        return x[0] < x[1]

    def _Equal(self, x, a):
        return x[0] == x[1]

    def _Not(self, x, a):
        return torch.logical_not(x[0])

    def _And(self, x, a):
        return torch.logical_and(x[0], x[1])

    def _Xor(self, x, a):
        return torch.logical_xor(x[0], x[1])

    def _Where(self, x, a):
        return torch.where(x[0], x[1], x[2])

    def _Clip(self, x, a):
        lo = x[1] if len(x) > 1 else None
        hi = x[2] if len(x) > 2 else None
        return torch.clamp(x[0], min=lo, max=hi)

    def _Range(self, x, a):
        return torch.arange(x[0].item(), x[1].item(), x[2].item(), dtype=x[0].dtype, device=x[0].device)

    def _BatchNormalization(self, x, a):
        if a.get("training_mode", 0):
            raise NotImplementedError("BatchNormalization training_mode")
        return F.batch_norm(x[0], x[3], x[4], x[1], x[2], training=False, eps=a.get("epsilon", 1e-5))

    def _Transpose(self, x, a):
        return x[0].permute(a["perm"])

    # ------------------------------------------------------------------
    def export(self, src: str | Path, dst: str | Path) -> None:
        """Write the current parameter values into a copy of ``src``."""
        model = onnx.load(str(src))
        for t in model.graph.initializer:
            key = self.param_names.get(t.name)
            if key is not None:
                arr = self.params[key].detach().float().cpu().numpy()
                if tuple(arr.shape) != tuple(t.dims):
                    raise ValueError(f"shape changed for {t.name}")
                t.CopyFrom(numpy_helper.from_array(arr, t.name))
        onnx.checker.check_model(model)
        onnx.save(model, str(dst))
