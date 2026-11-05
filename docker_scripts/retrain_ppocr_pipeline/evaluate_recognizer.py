"""Evaluate production-prepared inputs; save every prediction, including failures."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'finetune_paddleocr_line_rec'))
from train_ppocrv6_rec import decode_ctc, load_transformers_components, pairs_metric, read_label_file


def prepared_tensor(image_path):
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    if rgb.shape[0] != 48 or not 960 <= rgb.shape[1] <= 3200:
        raise ValueError(f'Unexpected input shape: {image_path}: {rgb.shape}')
    return (torch.from_numpy(np.ascontiguousarray(rgb[..., ::-1].transpose(2, 0, 1))).float() / 255 - .5).unsqueeze(0) / .5


def evaluate(model_source, dataset, output, split='val', limit=0, onnx_path=None):
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    _, processor, model = load_transformers_components(str(model_source))
    model.eval().to(device)
    session = None
    if onnx_path:
        import onnxruntime as ort
        options = ort.SessionOptions(); options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(onnx_path), options, providers=['CPUExecutionProvider'])
    labels = read_label_file(dataset, split)
    if limit:
        labels = labels[:limit]
    rows = []
    # Always start a new evaluation rather than resuming results from other weights.
    prediction_path = output / f'{split}_predictions.jsonl'
    with prediction_path.open('w', encoding='utf-8') as handle, torch.inference_mode():
        for image_path, reference, relative in tqdm(labels, desc=f'evaluate {split}'):
            inputs = prepared_tensor(image_path)
            probabilities = model(pixel_values=inputs.to(device)).last_hidden_state.float().cpu()
            raw = decode_ctc(probabilities, processor.character_list)[0]
            row = dict(image=relative, reference=reference, raw=raw, timesteps=probabilities.shape[1], width=inputs.shape[-1])
            if session:
                ort_probs = session.run(None, {session.get_inputs()[0].name: inputs.numpy()})[0]
                row.update(onnx_raw=decode_ctc(torch.from_numpy(ort_probs), processor.character_list)[0], max_abs_diff=float(np.abs(probabilities.numpy() - ort_probs).max()))
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    predictions, references = [r['raw'] for r in rows], [r['reference'] for r in rows]
    metrics = pairs_metric(predictions, references)
    metrics.update(model=str(model_source), split=split, device=str(device), empty_predictions=sum(not r['raw'] for r in rows),
                   label_sha256=hashlib.sha256((dataset / f'rec_gt_{split}.txt').read_bytes()).hexdigest())
    metrics['length_groups'] = {}
    for name, lower, upper in [('short_1_12', 0, 12), ('medium_13_40', 12, 40), ('long_over_40', 40, float('inf'))]:
        subset = [r for r in rows if lower < len(r['reference']) <= upper]
        metrics['length_groups'][name] = pairs_metric([r['raw'] for r in subset], [r['reference'] for r in subset])
    if session:
        metrics['onnx_parity'] = dict(samples=len(rows), text_matches=sum(r['raw'] == r['onnx_raw'] for r in rows), max_abs_diff=max((r['max_abs_diff'] for r in rows), default=0))
    (output / f'{split}_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--onnx', type=Path)
    args = parser.parse_args()
    evaluate(args.model, args.dataset, args.output, args.split, args.limit, args.onnx)
