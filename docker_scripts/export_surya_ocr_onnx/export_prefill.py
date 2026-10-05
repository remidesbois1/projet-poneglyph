"""Prune the LM head and unused final-layer work from state-only prompt prefill."""
import argparse
import hashlib
import json
from pathlib import Path
import onnx


def export(directory: Path, fused_norms=False):
    model = onnx.load(directory / "decoder.onnx", load_external_data=False)
    if fused_norms:
        model = onnx.version_converter.convert_version(model, 23)
        producers = {out: node for node in model.graph.node for out in node.output}
        replacements = {}
        for node in model.graph.node:
            if node.op_type != "Mul":
                continue
            div = producers.get(node.input[0])
            if div is None or div.op_type != "Div":
                continue
            sqrt = producers.get(div.input[1])
            if sqrt is None or sqrt.op_type != "Sqrt":
                continue
            add = producers.get(sqrt.input[0])
            if add is None or add.op_type != "Add":
                continue
            mean = producers.get(add.input[0])
            if mean is None or mean.op_type != "ReduceMean":
                continue
            square = producers.get(mean.input[0])
            if square is None or square.op_type != "Mul" or list(square.input) != [div.input[0], div.input[0]]:
                continue
            replacements[node.name] = onnx.helper.make_node("RMSNormalization", [div.input[0], node.input[1]], list(node.output),
                name=node.name, epsilon=1e-6, axis=-1)
        nodes = [replacements.get(node.name, node) for node in model.graph.node]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        print(f"Fused {len(replacements)} RMS normalizations")
    outputs = [item for item in model.graph.output if item.name.startswith("present_")]
    live = {item.name for item in outputs}
    nodes = []
    for node in reversed(model.graph.node):
        if live.intersection(node.output):
            nodes.append(node)
            live.update(node.input)
    initializers = [item for item in model.graph.initializer if item.name in live]
    inputs = [item for item in model.graph.input if item.name in live]
    for name, values in [("node", reversed(nodes)), ("input", inputs), ("output", outputs), ("initializer", initializers)]:
        target = getattr(model.graph, name)
        del target[:]
        target.extend(values)
    path = directory / ("prefill-fast.onnx" if fused_norms else "prefill.onnx")
    onnx.save(model, path)
    onnx.checker.check_model(str(path))
    manifest_path = directory / "browser_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Exported state-only prefill: {len(nodes)} nodes, {len(outputs)} GPU outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--fused-norms", action="store_true")
    args = parser.parse_args()
    export(args.directory, args.fused_norms)
