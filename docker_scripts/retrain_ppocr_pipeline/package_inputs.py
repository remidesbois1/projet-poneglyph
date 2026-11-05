"""Package an explicit allowlist of private inputs for SSH transfer to Runpod."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument('--detector-only', action='store_true')
    stage.add_argument('--recognizer-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    entries = []
    folders = ['recognizer'] if args.recognizer_only else ['detector'] + ([] if args.detector_only else ['recognizer'])
    for folder in folders:
        entries += [(p, 'data/' + p.relative_to(args.dataset).as_posix()) for p in (args.dataset / folder).rglob('*') if p.is_file()]
    entries += [(args.dataset / name, 'data/' + name) for name in ['plan.json', 'dataset_summary.json', 'exclusions.json']]
    source_files = ['train_detector.py', 'evaluate_recognizer.py', 'export_recognizer.py', 'requirements.txt']
    entries += [(Path(__file__).parent / name, 'docker_scripts/retrain_ppocr_pipeline/' + name) for name in source_files]
    trainer = 'docker_scripts/finetune_paddleocr_line_rec/train_ppocrv6_rec.py'
    entries.append((root / trainer, trainer))
    baseline = Path(__file__).parent / 'outputs/baseline-recognizer'
    entries += [(baseline / name, 'baseline-recognizer/' + name) for name in ['model.safetensors', 'config.json', 'preprocessor_config.json']]
    manifest = root / 'docker_scripts/audit_ppocr_pipeline/cache/models/browser_manifest.json'
    entries.append((manifest, 'baseline-browser-manifest.json'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, 'w:gz', compresslevel=1) as archive:
        for source, target in entries:
            if not source.is_file() or source.is_symlink():
                raise ValueError(f'Unexpected archive source: {source}')
            if source.suffix in ['.env', '.toml'] or '.cache' in source.parts:
                raise ValueError('Credential/cache files must not enter the bundle')
            archive.add(source, arcname=target, recursive=False)
    summary = dict(files=len(entries), bytes=args.output.stat().st_size, sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                   dataset_plan_sha256=hashlib.sha256((args.dataset / 'plan.json').read_bytes()).hexdigest(), private=True)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
