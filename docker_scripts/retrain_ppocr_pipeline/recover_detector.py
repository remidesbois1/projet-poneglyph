"""Recover trainable legacy YOLO weights from the pinned production ONNX.

Fused Conv+BN layers are represented by an equivalent neutral BN. The
unexported one-to-many training head is initialized from the one-to-one head.
Recovery is accepted only after numeric inference parity on real pages.
"""
import argparse
import ast
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
import ultralytics
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pages', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    graph = onnx.load(args.onnx)
    metadata = {p.key: p.value for p in graph.metadata_props}
    if ultralytics.__version__ != metadata['version']:
        raise ValueError(f"Architecture must use Ultralytics {metadata['version']}, got {ultralytics.__version__}")
    model = DetectionModel('yolo26n.yaml', nc=1, verbose=False).float().eval()
    model.names = ast.literal_eval(metadata['names'])
    state = {k: v.clone() for k, v in model.state_dict().items()}
    mapped = set()
    for value in graph.graph.initializer:
        if not value.name.startswith('model.'):
            continue
        array = np.array(onnx.numpy_helper.to_array(value), copy=True)
        tensor = torch.from_numpy(array)
        key = value.name
        if key in state:
            if tuple(state[key].shape) != tuple(tensor.shape):
                raise ValueError(f'Shape mismatch: {key}')
            state[key] = tensor.to(state[key].dtype)
            mapped.add(key)
        elif key.endswith('.conv.bias'):
            prefix = key[:-len('.conv.bias')] + '.bn'
            norm = model.get_submodule(prefix)
            for suffix, replacement in [
                ('weight', torch.full_like(state[prefix+'.weight'], math.sqrt(1 + norm.eps))),
                ('bias', tensor.to(state[prefix+'.bias'].dtype)),
                ('running_mean', torch.zeros_like(state[prefix+'.running_mean'])),
                ('running_var', torch.ones_like(state[prefix+'.running_var'])),
                ('num_batches_tracked', torch.zeros_like(state[prefix+'.num_batches_tracked'])),
            ]:
                state[prefix+'.'+suffix] = replacement
                mapped.add(prefix+'.'+suffix)
        else:
            raise ValueError(f'Unmapped model initializer: {key}')
    copied = []
    for key in state:
        if key in mapped:
            continue
        source = key.replace('.cv2.', '.one2one_cv2.').replace('.cv3.', '.one2one_cv3.')
        if source != key and source in mapped and state[source].shape == state[key].shape:
            state[key] = state[source].clone()
            mapped.add(key)
            copied.append(key)
    missing = [k for k, _ in model.named_parameters() if k not in mapped]
    if missing:
        raise ValueError(f'Unrecovered trainable parameters: {missing}')
    model.load_state_dict(state, strict=True)
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.onnx), options, providers=['CPUExecutionProvider'])
    parity = []
    with torch.inference_mode():
        for image_path in sorted(args.pages.glob('*.png'))[:4]:
            image = Image.open(image_path).convert('RGB')
            ratio = min(800/image.width, 800/image.height)
            image = image.resize((round(image.width*ratio), round(image.height*ratio)))
            canvas = Image.new('RGB', (800, 800), (128, 128, 128))
            canvas.paste(image, ((800-image.width)//2, (800-image.height)//2))
            x = np.asarray(canvas).transpose(2,0,1).astype(np.float32)[None]/255
            result = model(torch.from_numpy(x))
            while isinstance(result, (tuple, list)):
                result = result[0]
            native = result.cpu().numpy()[0]
            expected = session.run(None, {session.get_inputs()[0].name:x})[0][0]
            a, b = native[native[:,4] >= .05], expected[expected[:,4] >= .05]
            if len(a) != len(b) or not len(a):
                raise ValueError(f'Prediction count mismatch for {image_path.name}: {len(a)}, {len(b)}')
            unused = set(range(len(b))); max_box_error = max_score_error = 0.
            for row in a:
                closest = min(unused, key=lambda i: float(np.abs(row[:4]-b[i,:4]).max()))
                max_box_error = max(max_box_error, float(np.abs(row[:4]-b[closest,:4]).max()))
                max_score_error = max(max_score_error, float(abs(row[4]-b[closest,4])))
                unused.remove(closest)
            if max_box_error > .1 or max_score_error > .001:
                raise ValueError(f'Recovery parity failed: {max_box_error}, {max_score_error}')
            parity.append(dict(page=image_path.name, detections=len(a), max_box_error=max_box_error, max_score_error=max_score_error))
    if len(parity) < 4:
        raise ValueError('Four real pages required for recovery parity')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train_args = {**DEFAULT_CFG_DICT, 'model':'yolo26n.yaml', 'task':'detect', 'imgsz':800, 'single_cls':True}
    model.args = train_args
    torch.save(dict(model=model, ema=None, optimizer=None, updates=None, epoch=-1,
                    best_fitness=None, train_args=train_args, version=ultralytics.__version__,
                    date=datetime.now(timezone.utc).isoformat()), args.output)
    report = dict(source_sha256=hashlib.sha256(args.onnx.read_bytes()).hexdigest(), version=ultralytics.__version__,
                  trainable_parameters_recovered=len(list(model.named_parameters())), training_head_tensors_copied=len(copied),
                  parity=parity, note='Original BN statistics are not recoverable. Freeze BN running statistics during fine-tuning. Original custom detector split is not fully recoverable; evaluation is not blind.')
    args.output.with_suffix('.recovery.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
