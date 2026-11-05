"""Retrain the page bubble detector with the preserved OCR page splits."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--model', default='yolo26n.pt')
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--mosaic', type=float, default=.5)
    parser.add_argument('--scale', type=float, default=.25)
    parser.add_argument('--translate', type=float, default=.05)
    parser.add_argument('--close-mosaic', type=int, default=10)
    parser.add_argument('--warmup-epochs', type=float, default=3.)
    parser.add_argument('--warmup-bias-lr', type=float, default=.1)
    parser.add_argument('--freeze-batchnorm', action='store_true')
    parser.add_argument('--initial-eval', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--image-size', type=int, default=800)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Refusing full training without CUDA')
    summary = json.loads((args.dataset / 'dataset_summary.json').read_text())
    if summary['smoke'] or summary['limited_recognition']:
        raise ValueError('A complete production dataset is required')
    pages = json.loads((args.dataset / 'detector/pages.json').read_text())
    ids = [p['id'] for p in pages]
    if len(ids) != len(set(ids)):
        raise ValueError('Page leakage in detector dataset')
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(path=str((args.dataset / 'detector').resolve()), train='train/images', val='val/images', test='test/images', names={0: 'bubble'})
    yaml_path = args.output / 'data.yaml'
    yaml_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    model = YOLO(args.model)
    if args.freeze_batchnorm:
        def freeze_batchnorm(trainer):
            for module in trainer.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        model.add_callback('on_train_batch_start', freeze_batchnorm)
    # First stage starts from generic weights because the previous published
    # detector splits are not fully recoverable. A polish stage uses its best.pt.
    base_path = Path(model.ckpt_path)
    provenance = dict(base_model=str(base_path), base_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                      dataset_plan_sha256=hashlib.sha256((args.dataset / 'plan.json').read_bytes()).hexdigest(),
                      gpu=torch.cuda.get_device_name(0), seed=42, image_size=args.image_size,
                      annotation_note='All known annotations validated; completeness unproven. Historical holdouts were already inspected.')
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    started = time.time()
    initial_validation = None
    if args.initial_eval:
        # Ultralytics standalone validation fuses Conv+BN in place. Evaluate a
        # separate instance so training still loads every BN parameter/buffer.
        initial_validation = YOLO(args.model).val(data=str(yaml_path), split='val', imgsz=args.image_size, device=0, plots=False).results_dict
        (args.output / 'initial_validation.json').write_text(json.dumps(initial_validation, indent=2), encoding='utf-8')
        expected_state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
        def verify_loaded_state(trainer):
            loaded = trainer.model.state_dict()
            changed = [k for k, value in expected_state.items() if k not in loaded or not torch.equal(value, loaded[k].detach().cpu())]
            if changed:
                raise ValueError(f'Training did not preserve the recovered initial state: {changed[:12]}')
            print(json.dumps({'verified_initial_tensors': len(expected_state)}), flush=True)
        model.add_callback('on_pretrain_routine_end', verify_loaded_state)
    result = model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.image_size, batch=args.batch, patience=args.patience,
                         device=0, workers=4, seed=42, deterministic=True, amp=not args.no_amp,
                         project=str(args.output), name='run', exist_ok=False,
                         freeze=len(model.model.model)-1 if args.freeze_backbone else None,
                         optimizer='AdamW', lr0=args.learning_rate, lrf=.01, cos_lr=True,
                         warmup_epochs=args.warmup_epochs, warmup_bias_lr=args.warmup_bias_lr,
                         fliplr=0., flipud=0., degrees=0., hsv_h=0., hsv_s=0., hsv_v=.1,
                         translate=args.translate, scale=args.scale, mosaic=args.mosaic, close_mosaic=args.close_mosaic, cache=False, plots=False)
    best = Path(result.save_dir) / 'weights/best.pt'
    best_model = YOLO(best)
    val = best_model.val(data=str(yaml_path), split='val', imgsz=args.image_size, device=0, plots=False)
    selected_initial = bool(initial_validation and val.results_dict['metrics/mAP50-95(B)'] <= initial_validation['metrics/mAP50-95(B)'])
    if selected_initial:
        shutil.copy2(base_path, args.output / 'selected.pt')
        best_model = YOLO(args.output / 'selected.pt')
    onnx_path = best_model.export(format='onnx', imgsz=args.image_size, batch=1, dynamic=False, simplify=False, opset=12, nms=True, device='cpu')
    report = dict(best_checkpoint=str(best), onnx=str(onnx_path), seconds=time.time()-started, validation=val.results_dict,
                  initial_validation=initial_validation, selected_initial=selected_initial,
                  evaluation_note='Ultralytics validation; production preprocessing and threshold must be tested separately.')
    (args.output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
