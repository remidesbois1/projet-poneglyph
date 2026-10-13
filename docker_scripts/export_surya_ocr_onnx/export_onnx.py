"""Surya OCR 2: portable vision + hybrid DeltaNet ONNX, batch one, chunks <=32.

All recurrent state stays explicit. A finite Neumann series solves the triangular
DeltaNet chunk recurrence (nilpotent at order 32); no Loop or custom operators.
Weights/matmuls are FP16, residuals, reductions and recurrent state are FP32.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper, external_data_helper
from safetensors import safe_open

END = 2**63 - 1


class Graph:
    def __init__(self, source, directory, name, shared=None):
        self.weights = safe_open(source / "model.safetensors", framework="pt")
        self.nodes, self.initializers, self.inputs, self.outputs = [], [], [], []
        self.serial, self.weight_names = 0, {}
        self.directory, self.name = directory, name
        self.data_name = shared.data_name if shared else f"{name}.onnx.data"
        self.data_index = shared.data_index if shared else {}
        self.stream = None if shared else (directory / self.data_name).open("wb")
        self.single_token = name == "decode"

    def fresh(self, stem):
        self.serial += 1
        return f"{stem}_{self.serial}"

    def const(self, value, dtype=np.float32, name=None):
        name = name or self.fresh("c")
        tensor = numpy_helper.from_array(np.asarray(value, dtype=dtype), name)
        if len(tensor.raw_data) > 4096:
            digest = hashlib.sha256(tensor.raw_data).hexdigest()
            if digest not in self.data_index:
                if self.stream is None:
                    raise RuntimeError("Decode graph introduced a weight absent from prefill")
                self.data_index[digest] = (self.stream.tell(), len(tensor.raw_data))
                self.stream.write(tensor.raw_data)
            offset, length = self.data_index[digest]
            external_data_helper.set_external_data(tensor, self.data_name, offset, length)
            tensor.ClearField("raw_data")
            tensor.data_location = T.EXTERNAL
        self.initializers.append(tensor)
        return name

    def array(self, name):
        return self.weights.get_tensor(name).float().numpy()

    def weight(self, name, transpose=False, dtype=np.float16):
        key = (name, transpose, str(dtype))
        if key not in self.weight_names:
            a = self.array(name)
            self.weight_names[key] = self.const(a.T if transpose else a, dtype, self.fresh(name))
        return self.weight_names[key]

    def op(self, kind, *inputs, **attrs):
        output = self.fresh(kind.lower())
        self.nodes.append(h.make_node(kind, list(inputs), [output], name=output, **attrs))
        return output

    def input(self, name, dtype, shape):
        self.inputs.append(h.make_tensor_value_info(name, dtype, shape))
        return name

    def output(self, value, name, dtype, shape):
        self.nodes.append(h.make_node("Identity", [value], [name]))
        self.outputs.append(h.make_tensor_value_info(name, dtype, shape))

    def cast(self, x, dtype=T.FLOAT):
        return self.op("Cast", x, to=dtype)

    def shape(self, x, shape):
        return self.op("Reshape", x, self.const(shape, np.int64))

    def trans(self, x, perm):
        return self.op("Transpose", x, perm=perm)

    def unsqueeze(self, x, axes):
        return self.op("Unsqueeze", x, self.const(axes, np.int64))

    def gather(self, x, indices, axis=0):
        return self.op("Gather", x, self.const(indices, np.int64), axis=axis)

    def slice(self, x, start, end, axis=-1):
        return self.op("Slice", x, self.const([start], np.int64), self.const([end], np.int64), self.const([axis], np.int64))

    def mm(self, a, b):
        return self.op("MatMul", a, b)

    def linear(self, x, prefix, bias=False):
        y = self.mm(x, self.weight(prefix + ".weight", True))
        return self.op("Add", y, self.weight(prefix + ".bias")) if bias else y

    def silu(self, x):
        return self.op("Mul", x, self.op("Sigmoid", x))

    def norm(self, x, prefix, kind="rms"):
        x = self.cast(x)
        if kind == "layer":
            mean = self.op("ReduceMean", x, axes=[-1], keepdims=1)
            x = self.op("Sub", x, mean)
        squared = self.op("Mul", x, x)
        if kind == "l2":
            var = self.op("ReduceSum", squared, self.const([-1], np.int64), keepdims=1)
        else:
            var = self.op("ReduceMean", squared, axes=[-1], keepdims=1)
        y = self.op("Div", x, self.op("Sqrt", self.op("Add", var, self.const(1e-6))))
        if prefix:
            weight = self.array(prefix + ".weight")
            if kind == "rms":
                weight = weight + 1
            y = self.op("Mul", y, self.const(weight))
            if kind == "layer":
                y = self.op("Add", y, self.weight(prefix + ".bias", dtype=np.float32))
        return y

    def rope(self, x, cos, sin, dim=64, partial=False):
        rot = self.cast(self.slice(x, 0, dim))
        half = self.op("Concat", self.op("Neg", self.slice(rot, dim // 2, dim)), self.slice(rot, 0, dim // 2), axis=-1)
        y = self.cast(self.op("Add", self.op("Mul", rot, cos), self.op("Mul", half, sin)), T.FLOAT16)
        return self.op("Concat", y, self.slice(x, dim, END), axis=-1) if partial else y

    def gelu(self, x, approximate=False):
        x = self.cast(x)
        if approximate:
            cube = self.op("Mul", self.op("Mul", x, x), x)
            inner = self.op("Mul", self.op("Add", x, self.op("Mul", cube, self.const(0.044715))), self.const(np.sqrt(2 / np.pi)))
            activation = self.op("Tanh", inner)
        else:
            activation = self.op("Erf", self.op("Mul", x, self.const(1 / np.sqrt(2))))
        return self.cast(self.op("Mul", self.op("Mul", x, self.const(0.5)), self.op("Add", activation, self.const(1))), T.FLOAT16)

    def save(self):
        if self.stream is not None:
            self.stream.close()
        model = h.make_model(h.make_graph(self.nodes, f"Surya-{self.name}", self.inputs, self.outputs, self.initializers),
                             opset_imports=[h.make_opsetid("", 17)], ir_version=9, producer_name="poneglyph-surya-webgpu")
        onnx.save(model, self.directory / f"{self.name}.onnx")
        onnx.checker.check_model(str(self.directory / f"{self.name}.onnx"))
        print(f"Exported {self.name}: {len(self.nodes)} nodes", flush=True)


def vision(source, out, attention_chunks=1):
    g = Graph(source, out, "vision")
    pixels = g.input("pixel_values", T.FLOAT16, ["patches", 1536])
    indices = g.input("position_indices", T.INT64, [4, "patches"])
    weights = g.input("position_weights", T.FLOAT, [4, "patches", 1])
    cos = g.input("vision_cos", T.FLOAT, [1, "patches", 64])
    sin = g.input("vision_sin", T.FLOAT, [1, "patches", 64])
    if attention_chunks > 1:
        assert attention_chunks == 16
        g.input("query_group_splits", T.INT64, [4])
        for group in range(4):
            g.input(f"query_splits_{group}", T.INT64, [4])
    prefix = "model.visual."
    patch_weight = g.array(prefix + "patch_embed.proj.weight").reshape(768, 1536).T
    x = g.op("Add", g.mm(pixels, g.const(patch_weight, np.float16)), g.weight(prefix + "patch_embed.proj.bias"))
    pos = g.cast(g.op("Gather", g.weight(prefix + "pos_embed.weight"), indices, axis=0))
    pos = g.op("ReduceSum", g.op("Mul", pos, weights), g.const([0], np.int64), keepdims=0)
    x = g.op("Add", g.cast(x), pos)
    for layer in range(12):
        p = prefix + f"blocks.{layer}."
        norm = g.cast(g.norm(x, p + "norm1", "layer"), T.FLOAT16)
        qkv = g.shape(g.linear(norm, p + "attn.qkv", True), [-1, 3, 12, 64])
        q, k, v = [g.trans(g.gather(qkv, i, 1), [1, 0, 2]) for i in range(3)]
        q, k = g.rope(q, cos, sin), g.rope(k, cos, sin)
        query_groups = [[q]]
        if attention_chunks > 1:
            groups = [g.fresh("query_group") for _ in range(4)]
            g.nodes.append(h.make_node("Split", [q, "query_group_splits"], groups, axis=1))
            query_groups = []
            for index, group in enumerate(groups):
                queries = [g.fresh("query_chunk") for _ in range(4)]
                g.nodes.append(h.make_node("Split", [group, f"query_splits_{index}"], queries, axis=1))
                query_groups.append(queries)
        transposed_key = g.trans(k, [0, 2, 1])
        attended_groups = []
        for queries in query_groups:
            attended_chunks = []
            for query in queries:
                scores = g.op("Mul", g.cast(g.mm(query, transposed_key)), g.const(0.125))
                probabilities = g.cast(g.op("Softmax", scores, axis=-1), T.FLOAT16)
                attended_chunks.append(g.mm(probabilities, v))
            attended_groups.append(attended_chunks[0] if len(queries) == 1 else g.op("Concat", *attended_chunks, axis=1))
        # Flatten each group before concatenation. Apart from bounding every
        # kernel's binding count, this prevents a Concat fusion into 16 inputs.
        flattened = [g.shape(g.trans(group, [1, 0, 2]), [-1, 768]) for group in attended_groups]
        attended = flattened[0] if len(flattened) == 1 else g.op("Concat", *flattened, axis=0)
        x = g.op("Add", x, g.cast(g.linear(attended, p + "attn.proj", True)))
        norm = g.cast(g.norm(x, p + "norm2", "layer"), T.FLOAT16)
        ff = g.gelu(g.linear(norm, p + "mlp.linear_fc1", True), True)
        x = g.op("Add", x, g.cast(g.linear(ff, p + "mlp.linear_fc2", True)))
    x = g.cast(g.shape(g.norm(x, prefix + "merger.norm", "layer"), [-1, 3072]), T.FLOAT16)
    x = g.gelu(g.linear(x, prefix + "merger.linear_fc1", True))
    x = g.linear(x, prefix + "merger.linear_fc2", True)
    g.output(x, "image_features", T.FLOAT16, ["image_tokens", 1024])
    g.save()


def delta(g, x, p, layer):
    conv = g.input(f"conv_{layer}", T.FLOAT16, [1, 3, 6144])
    state = g.input(f"state_{layer}", T.FLOAT, [16, 128, 128])
    packed = g.linear(x, p + "in_proj_qkv")
    history = g.op("Concat", conv, packed, axis=1)
    g.output(g.slice(history, -3, END, 1), f"present_conv_{layer}", T.FLOAT16, [1, 3, 6144])
    cw = g.array(p + "conv1d.weight")[:, 0, :]
    parts = [g.op("Mul", g.cast(g.slice(history, i, i - 3 if i < 3 else END, 1)), g.const(cw[:, i].reshape(1, 1, -1))) for i in range(4)]
    mixed = g.cast(g.silu(g.op("Add", g.op("Add", parts[0], parts[1]), g.op("Add", parts[2], parts[3]))), T.FLOAT16)
    q, k, v = [g.trans(g.shape(g.slice(mixed, i * 2048, (i + 1) * 2048), [-1, 16, 128]), [1, 0, 2]) for i in range(3)]
    q = g.op("Mul", g.norm(q, None, "l2"), g.const(128**-0.5))
    k, v = g.norm(k, None, "l2"), g.cast(v)
    a = g.op("Add", g.cast(g.linear(x, p + "in_proj_a")), g.weight(p + "dt_bias", dtype=np.float32))
    sp = g.op("Add", g.op("Relu", a), g.op("Log", g.op("Add", g.op("Exp", g.op("Neg", g.op("Abs", a))), g.const(1))))
    decay = g.op("Mul", sp, g.const(-np.exp(g.array(p + "A_log"))))
    decay = g.trans(g.shape(decay, [-1, 16]), [1, 0])
    beta = g.op("Sigmoid", g.cast(g.linear(x, p + "in_proj_b")))
    beta = g.unsqueeze(g.trans(g.shape(beta, [-1, 16]), [1, 0]), [-1])
    if g.single_token:
        decayed = g.op("Mul", state, g.op("Exp", g.unsqueeze(decay, [-1])))
        difference = g.op("Mul", g.op("Sub", v, g.mm(k, decayed)), beta)
        present = g.op("Add", decayed, g.mm(g.trans(k, [0, 2, 1]), difference))
        core = g.mm(q, present)
        g.output(present, f"present_state_{layer}", T.FLOAT, [16, 128, 128])
        core = g.trans(core, [1, 0, 2])
        z = g.shape(g.cast(g.linear(x, p + "in_proj_z")), [-1, 16, 128])
        gated = g.op("Mul", g.norm(core, p + "norm", "gated"), g.silu(z))
        return g.linear(g.cast(g.shape(gated, [1, -1, 2048]), T.FLOAT16), p + "out_proj")
    cumulative = g.op("CumSum", decay, g.const(1, np.int64))
    gc = g.unsqueeze(cumulative, [-1])
    expg = g.op("Exp", gc)
    diff = g.op("Sub", gc, g.unsqueeze(cumulative, [1]))
    decay_matrix = g.op("Mul", g.op("Exp", g.op("Mul", diff, "lower")), "lower")
    kb = g.op("Mul", k, beta)
    vb = g.op("Mul", v, beta)
    a = g.op("Neg", g.op("Mul", g.op("Mul", g.mm(kb, g.trans(k, [0, 2, 1])), decay_matrix), "strict_lower"))
    inverse = g.op("Add", "identity", a)
    for _ in range(4):
        a = g.mm(a, a)
        inverse = g.mm(inverse, g.op("Add", "identity", a))
    u = g.mm(inverse, vb)
    w = g.mm(inverse, g.op("Mul", kb, expg))
    updated_values = g.op("Sub", u, g.mm(w, state))
    attention = g.op("Mul", g.mm(q, g.trans(k, [0, 2, 1])), decay_matrix)
    core = g.op("Add", g.mm(g.op("Mul", q, expg), state), g.mm(attention, updated_values))
    lastg = g.slice(gc, -1, END, 1)
    weighted_k = g.op("Mul", k, g.op("Exp", g.op("Sub", lastg, gc)))
    present = g.op("Add", g.op("Mul", state, g.op("Exp", lastg)), g.mm(g.trans(weighted_k, [0, 2, 1]), updated_values))
    g.output(present, f"present_state_{layer}", T.FLOAT, [16, 128, 128])
    core = g.trans(core, [1, 0, 2])
    z = g.shape(g.cast(g.linear(x, p + "in_proj_z")), [-1, 16, 128])
    gated = g.op("Mul", g.norm(core, p + "norm", "gated"), g.silu(z))
    return g.linear(g.cast(g.shape(gated, [1, -1, 2048]), T.FLOAT16), p + "out_proj")


def decoder(source, out, config, shared=None):
    g = Graph(source, out, "decode" if shared else "decoder", shared=shared)
    seq = 1 if shared else "sequence"
    ids = g.input("input_ids", T.INT64, [1, seq])
    image = g.input("image_features", T.FLOAT16, ["image_tokens", 1024])
    index = g.input("image_indices", T.INT64, [seq])
    mask = g.input("image_mask", T.BOOL, [1, seq, 1])
    cos = g.input("rope_cos", T.FLOAT, [1, 1, seq, 64])
    sin = g.input("rope_sin", T.FLOAT, [1, 1, seq, 64])
    attention_mask = g.input("attention_mask", T.FLOAT16, [1, 1, seq, "total_sequence"])
    if not g.single_token:
        for name in ["identity", "lower", "strict_lower"]:
            g.input(name, T.FLOAT, [1, seq, seq])
    embed = g.weight("model.language_model.embed_tokens.weight")
    text = g.op("Gather", embed, ids, axis=0)
    visual = g.unsqueeze(g.op("Gather", image, index, axis=0), [0])
    x = g.cast(g.op("Where", mask, visual, text))
    for layer, kind in enumerate(config["text_config"]["layer_types"]):
        p = f"model.language_model.layers.{layer}."
        norm = g.cast(g.norm(x, p + "input_layernorm"), T.FLOAT16)
        if kind == "linear_attention":
            attended = delta(g, norm, p + "linear_attn.", layer)
        else:
            p2 = p + "self_attn."
            qg = g.shape(g.linear(norm, p2 + "q_proj"), [1, -1, 8, 512])
            q = g.cast(g.norm(g.slice(qg, 0, 256), p2 + "q_norm"), T.FLOAT16)
            gate = g.shape(g.slice(qg, 256, 512), [1, -1, 2048])
            k = g.cast(g.norm(g.shape(g.linear(norm, p2 + "k_proj"), [1, -1, 2, 256]), p2 + "k_norm"), T.FLOAT16)
            v = g.trans(g.shape(g.linear(norm, p2 + "v_proj"), [1, -1, 2, 256]), [0, 2, 1, 3])
            q = g.rope(g.trans(q, [0, 2, 1, 3]), cos, sin, partial=True)
            k = g.rope(g.trans(k, [0, 2, 1, 3]), cos, sin, partial=True)
            for label, value in [("key", k), ("value", v)]:
                past = g.input(f"{label}_{layer}", T.FLOAT16, [1, 2, "past_sequence", 256])
                present = g.op("Concat", past, value, axis=2)
                g.output(present, f"present_{label}_{layer}", T.FLOAT16, [1, 2, "total_sequence", 256])
                if label == "key":
                    k = g.gather(present, [0] * 4 + [1] * 4, 1)
                else:
                    v = g.gather(present, [0] * 4 + [1] * 4, 1)
            scores = g.op("Add", g.op("Mul", g.mm(q, g.trans(k, [0, 1, 3, 2])), g.const(256**-0.5, np.float16)), attention_mask)
            probabilities = g.cast(g.op("Softmax", g.cast(scores), axis=-1), T.FLOAT16)
            attended = g.shape(g.trans(g.mm(probabilities, v), [0, 2, 1, 3]), [1, -1, 2048])
            attended = g.op("Mul", attended, g.op("Sigmoid", gate))
            attended = g.linear(attended, p2 + "o_proj")
        x = g.op("Add", x, g.cast(attended))
        norm = g.cast(g.norm(x, p + "post_attention_layernorm"), T.FLOAT16)
        gate, up = g.linear(norm, p + "mlp.gate_proj"), g.linear(norm, p + "mlp.up_proj")
        ff = g.cast(g.op("Mul", g.silu(g.cast(gate)), g.cast(up)), T.FLOAT16)
        x = g.op("Add", x, g.cast(g.linear(ff, p + "mlp.down_proj")))
    last = g.cast(g.norm(g.slice(x, -1, END, 1), "model.language_model.norm"), T.FLOAT16)
    logits = g.cast(g.mm(last, g.trans(embed, [1, 0])))
    g.output(logits, "logits", T.FLOAT, [1, 1, config["text_config"]["vocab_size"]])
    g.output(g.op("ArgMax", logits, axis=-1, keepdims=0), "next_token", T.INT64, [1, 1])
    g.save()
    return g


def export(source, out):
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    text, visual = config["text_config"], config["vision_config"]
    assert (text["hidden_size"], text["num_attention_heads"], text["num_key_value_heads"], text["head_dim"],
            text["linear_num_key_heads"], text["linear_num_value_heads"], text["linear_key_head_dim"],
            text["linear_value_head_dim"], text["linear_conv_kernel_dim"], text["vocab_size"]) == (1024, 8, 2, 256, 16, 16, 128, 128, 4, 65425)
    assert text["layer_types"] == (["linear_attention"] * 3 + ["full_attention"]) * 6
    assert visual["deepstack_visual_indexes"] == []
    assert (visual["depth"], visual["hidden_size"], visual["num_heads"], visual["patch_size"],
            visual["spatial_merge_size"], visual["temporal_patch_size"]) == (12, 768, 12, 16, 2, 2)
    vision(source, out)
    prefill = decoder(source, out, config)
    decoder(source, out, config, shared=prefill)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(source)
    prompt = "Transcris exactement le texte visible dans cette bulle. Ne rajoute rien."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chunks = [processor.tokenizer.encode(s, add_special_tokens=False) for s in formatted.split("<|image_pad|>")]
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {"schema": 1, "model_id": "Remidesbois/surya-bubble-ocr-poneglyph",
                "source_sha256": digest(source / "model.safetensors"), "config": config,
                "format": "mixed-fp16-fp32", "chunk_size": 32, "max_new_tokens": 256,
                "prompt_chunks": chunks, "image_token_id": 11, "stop_ids": [2],
                "min_pixels": 65536, "max_pixels": 1048576,
                "files": {name: {"bytes": (out / name).stat().st_size, "sha256": digest(out / name)}
                          for name in ["vision.onnx", "vision.onnx.data", "decoder.onnx", "decode.onnx", "decoder.onnx.data"]}}
    (out / "browser_manifest.json").write_text(json.dumps(manifest, indent=2))
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        shutil.copy2(source / name, out / name)
    from export_decode_weights import export as export_decode_index
    from export_prefill import export as export_state_prefill
    export_decode_index(source, out)
    export_state_prefill(out)
    export_state_prefill(out, fused_norms=True)
    print(json.dumps(json.loads((out / "browser_manifest.json").read_text())["files"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)
