"""Export dynamic-width browser models and verify exact prepared-input parity."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'finetune_paddleocr_line_rec'))
from train_ppocrv6_rec import decode_ctc, load_transformers_components
from evaluate_recognizer import prepared_tensor


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).last_hidden_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-manifest', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    _, processor, model = load_transformers_components(str(args.model))
    wrapper = Wrapper(model.eval()).eval()
    output = args.output / 'ppocrv6_bubble_line_rec.onnx'
    torch.onnx.export(wrapper, (torch.zeros(1, 3, 48, 960),), str(output), input_names=['pixel_values'], output_names=['logits'],
                      dynamic_axes={'pixel_values': {0: 'batch', 3: 'width'}, 'logits': {0: 'batch', 1: 'sequence'}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    # ORT WebGPU does not support ceil_mode=1. In this architecture these pools
    # all have stride 1, for which floor/ceil produce identical output sizes.
    patches = []
    for node in graph.graph.node:
        if node.op_type != 'MaxPool':
            continue
        attributes = {a.name: a for a in node.attribute}
        if attributes.get('ceil_mode') is not None and attributes['ceil_mode'].i == 1:
            if list(attributes['strides'].ints) != [1, 1]:
                raise ValueError('Cannot safely rewrite a strided ceil-mode pool')
            attributes['ceil_mode'].i = 0
            patches.append(node.name)
    webgpu = args.output / 'ppocrv6_bubble_line_rec_webgpu.onnx'
    onnx.checker.check_model(graph)
    onnx.save(graph, webgpu)
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    sessions = [ort.InferenceSession(str(p), options, providers=['CPUExecutionProvider']) for p in (output, webgpu)]
    manifest_rows = [json.loads(x) for x in (args.dataset / 'recognizer/line_manifest.jsonl').read_text(encoding='utf-8').splitlines() if x]
    val_rows = [r for r in manifest_rows if r['split'] == 'val']
    # Include the minimum and maximum real widths and several normal inputs.
    chosen = {r['single_line_image']: r for r in [*val_rows[:8], min(val_rows, key=lambda r: r['width']), max(val_rows, key=lambda r: r['width'])]}
    parity = []
    with torch.inference_mode():
        for row in chosen.values():
            tensor = prepared_tensor(args.dataset / 'recognizer' / row['single_line_image'])
            native = wrapper(tensor).cpu()
            native_text = decode_ctc(native, processor.character_list)[0]
            item = dict(image=row['single_line_image'], width=tensor.shape[-1], torch_text=native_text, variants=[])
            for session in sessions:
                pred = session.run(None, {session.get_inputs()[0].name: tensor.numpy()})[0]
                text = decode_ctc(torch.from_numpy(pred), processor.character_list)[0]
                delta = float(np.abs(native.numpy() - pred).max())
                if text != native_text or delta > .002:
                    raise ValueError(f'ONNX parity failed for {row["bubble_id"]}: {delta}, {text!r}, {native_text!r}')
                item['variants'].append(dict(text=text, max_abs_diff=delta))
            parity.append(item)
        # Widths not necessarily present in val still need dynamic-shape checks.
        for width in (960, 1200, 1600, 2400, 3200):
            tensor = torch.full((1, 3, 48, width), (128/255-.5)/.5)
            native = wrapper(tensor).cpu().numpy()
            for session in sessions:
                pred = session.run(None, {session.get_inputs()[0].name: tensor.numpy()})[0]
                if pred.shape != native.shape or not np.allclose(pred, native, atol=.002, rtol=.001):
                    raise ValueError(f'Dynamic-width parity failed at width {width}')
    manifest = json.loads(args.baseline_manifest.read_text(encoding='utf-8'))
    manifest.update(recognizer_onnx=output.name, input_shape=[1, 3, 48, 'width'], dynamic_width=True, image_height=48,
                    min_image_width=960, max_image_width=3200, resize='height_48_preserve_aspect_then_pad_to_min_width',
                    padding_color='#808080', channel_order='BGR', character_list=list(processor.character_list),
                    final_text_postprocess='ppocr_lexicon_spacing_case_only')
    report = dict(samples=len(parity), all_text_match=True, max_abs_diff=max(v['max_abs_diff'] for p in parity for v in p['variants']),
                  dynamic_widths=[960,1200,1600,2400,3200], webgpu_pool_patches=patches, items=parity)
    manifest['parity'] = report
    (args.output / 'browser_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'parity_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for name in ['config.json', 'preprocessor_config.json']:
        shutil.copy2(args.model / name, args.output / name)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
