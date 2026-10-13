"""Inspect difficult validation crops with a standard native BF16 reload."""
import argparse
import json
from pathlib import Path

import train
from training_monitor import atomic_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    train.core.configure_torch_runtime()
    benchmark = json.loads(args.validation.read_text(encoding='utf-8'))
    difficult = [s for s in benchmark['samples'] if s.get('hit_token_limit')][:3]
    rows = {str(r['id']): r for r in [json.loads(line) for line in train.core.VAL_FILE.read_text(encoding='utf-8').splitlines()]}
    processor = train.load_processor()
    model = train.load_model(args.checkpoint).eval()
    train.core.configure_generation(model, processor)
    result = []
    for sample in difficult:
        entry = rows[str(sample['id'])]
        prediction, metadata = train.core.generate_predictions(model, processor, [entry], return_metadata=True)
        result.append({'id':sample['id'], 'reference':sample['reference'], 'training_eval_prediction':sample['prediction'],
                       'native_bf16_batch1_prediction':prediction[0], **metadata[0]})
    atomic_json(args.output, {'checkpoint':args.checkpoint, 'split':'val', 'dtype':'bfloat16', 'batch_size':1, 'samples':result})
    print(json.dumps([{'id':s['id'],'hit_token_limit':s['hit_token_limit'],'native_prediction':s['native_bf16_batch1_prediction'][:100]} for s in result], ensure_ascii=True))
