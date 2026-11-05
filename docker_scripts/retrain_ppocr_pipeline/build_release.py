"""Build a reviewable, aggregate-only release from the completed 2026-11-03 run.

No upload occurs here. publish_release.py checks the exact file/hash allowlist.
Private scans, labels, prediction rows and account state never enter a release.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PP_REPO = 'Remidesbois/pp-ocrv6-one-piece-bubble-line-rec'
READER_REPO = 'Remidesbois/Poneglyph-ReaderNet'
DATE = '2026-11-03'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def order_metrics(predictions, annotations):
    """Score matched bubble ordering separately from GT-conditioned text CER."""
    orders = {r['id']: r.get('order') for r in annotations}
    total = dict(eligible_pages=0, exact_matched_order_pages=0, pairs=0, inversions=0,
                 skipped_incomplete_or_duplicate_order_pages=0, skipped_small_pages=0)
    for line in predictions.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        matched = [(m['j'], orders.get(row['bubbles'][m['i']]['id'])) for m in row['matches']]
        known = [v for _, v in matched]
        if any(not isinstance(v, (int, float)) for v in known) or len(set(known)) != len(known):
            total['skipped_incomplete_or_duplicate_order_pages'] += 1
            continue
        if len(matched) < 2:
            total['skipped_small_pages'] += 1
            continue
        actual = [v for _, v in sorted(matched)]
        inversions = sum(a > b for i, a in enumerate(actual) for b in actual[i + 1:])
        total['eligible_pages'] += 1
        total['exact_matched_order_pages'] += int(inversions == 0)
        total['pairs'] += len(actual) * (len(actual) - 1) // 2
        total['inversions'] += inversions
    total['note'] = ('Exploratory global order over spatially matched annotations with unique numeric order; '
                     'missed and extra bubbles are excluded. This is not a full reading-order success rate. '
                     'Panel and ordering weights are unchanged.')
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=Path, default=HERE / 'outputs')
    args = parser.parse_args()
    out = args.outputs.resolve()
    recognition = read(out / 'recognition-summary-final.json')
    selection = read(out / 'detector-selection.json')
    train = read(out / 'recognizer-results-v1/recognizer-v2/training_metrics.json')
    baseline = read(out / 'baseline-test-v1/summary.json')
    final = read(out / 'final-pipeline-test-v1/summary.json')
    assert train['best_epoch'] > 0 and train['skipped_optimizer_steps'] == 0
    assert final['pages'] == baseline['pages'] == 129
    assert final['bubbles'] == baseline['bubbles'] == 1221
    assert final['final']['cer'] < baseline['final']['cer'] and final['matched'] >= baseline['matched']
    assert recognition['splits']['test']['candidate']['final']['cer'] < recognition['splits']['test']['baseline']['final']['cer']
    assert recognition['browser']['candidate']['cer'] < recognition['browser']['baseline']['cer']
    assert selection['selected']['model'] == 'production' and selection['selected']['threshold'] == .25
    browser_detector = read(out / 'browser-active-detector-val20/result.json')
    assert browser_detector['status'] == 'complete' and len(browser_detector['rows']) == 20
    parity = read(out / 'recognizer-results-v1/recognizer-onnx/parity_report.json')
    parity.pop('items', None)
    assert parity['all_text_match']
    train_public = {k: v for k, v in train.items() if k not in ['reviewed_training_pair_exclusions', 'dataset_dir', 'model_id', 'resume_from']}
    train_public['reviewed_training_pair_exclusions'] = {'count': 28, 'split': 'train', 'reason': 'line_detector_omitted_visible_text', 'visually_reviewed': True}
    train_public['starting_checkpoint'] = {'repository': PP_REPO, 'revision': '10b932d4aadca2830850ccf5951116597404bef8'}
    annotations = read(ROOT / 'docker_scripts/audit_ppocr_pipeline/cache/validated.json')
    report = {
        'schema_version': 1, 'date': DATE,
        'protocol': {
            'dataset': 'Poneglyph historical page-level splits, already evaluated and inspected; not blind.',
            'normalization': recognition['normalization'],
            'text_matching': 'Greedy one-to-one spatial matching at IoU >= 0.5; missed GT bubbles receive empty text. Unmatched predictions counted separately, not included in CER.',
            'annotation_limit': 'Annotation completeness is unproven. Unmatched predictions are not verified false positives. Perfect annotated pages score text and GT coverage only, not extra regions or reading order.',
            'selection': 'OCR epoch, text policy and detector confidence selected on validation; detector selection frozen before final selected-pipeline test.',
            'native_crop_runtime': 'PyTorch 2.8.0+cu128 FP32, NVIDIA GeForce RTX 4090 24 GB on RunPod',
            'native_page_runtime': 'Actual production worker functions, Skia Canvas and ONNX Runtime Node 1.24.3 CPU',
            'browser_scope': '100 validation GT-crop recognizer inputs on Chrome WebGPU AMD RDNA3; 20 validation full-page active detector inputs on WASM; no full-corpus browser CER claim.',
            'sfx': 'SFX retained without a class-based exclusion or automatic skip; no SFX classifier trained. Short strings are not used as SFX labels.',
        },
        'dataset': {
            'pages': 1326, 'bubbles': 11670,
            'splits': read(out / 'dataset-v5/plan.json')['counts'],
            'plan_sha256': descriptor(out / 'dataset-v5/plan.json')['sha256'],
            'recognizer_train_effective': 9228, 'reviewed_train_pairs_excluded': 28,
            'zero_line_fallbacks_kept_all_splits': 24,
            'detector_pages': {'train': 1065, 'val': 127, 'test': 129},
            'detector_excluded_unvalidated_pages': 5, 'train_boxes_clipped_to_image': 2,
            'audit': 'All prepared PNG hashes and CTC capacity checked; no page hash duplicated across splits. Six multiline transcripts serialized with whitespace normalization; raw reference preserved privately.',
        },
        'recognizer_training': train_public,
        'recognition_gt_crops': {s: {k: v for k, v in r.items() if k != 'diagnostic_subset'} for s, r in recognition['splits'].items()},
        'postprocess': recognition['postprocess'],
        'browser_recognizer': recognition['browser'],
        'onnx_parity': parity,
        'detector_selection': selection,
        'full_page': {'test': {'baseline': baseline, 'selected': final},
                      'val_selected': read(out / 'new-ocr-old-detector-val025-lexicon.json'),
                      'val_1024_candidate_diagnostic': read(out / 'diagnostic-1024-pipeline-val-v1/summary.json')},
        'matched_reading_order': {name: order_metrics(out / folder / 'pages.jsonl', annotations)
                                 for name, folder in [('baseline', 'baseline-test-v1'), ('selected', 'final-pipeline-test-v1')]},
        'remaining_work': [
            'Annotate complete line boxes, including missed lines, and retrain the line detector. Twenty-eight visually confirmed train pairs currently omit visible text before recognition.',
            'Create a human-labeled ordinary/SFX evaluation set; the existing diagnostic subset is too small and error-focused for a reliable class-wide score.',
            'Acquire new unseen volumes and review all extra detections to establish recall and precision on an independent, complete holdout.',
            'Validate reading-order and panel annotations separately; text CER alone does not measure ordering quality.',
        ],
    }
    # Preserve only hashes/protocols of private per-page evaluation files.
    report['evidence_sha256'] = {name: descriptor(out / name)['sha256'] for name in [
        'final-pipeline-test-v1/pages.jsonl', 'baseline-test-v1/pages.jsonl',
        'final-pipeline-test-v1/protocol.json', 'recognition-summary-final.json',
        'browser-new-webgpu-val100-lexicon/result.json', 'browser-active-detector-val20/result.json']}
    report['production_artifacts'] = {name: descriptor(out / 'models-ppocr-v1' / name) for name in [
        'bubble_detector.onnx', 'bubble_line_detector_yolo26n.onnx', 'ppocrv6_bubble_line_rec_webgpu.onnx', 'panel_detector.onnx', 'ordering.onnx']}
    write(out / 'release-report.json', report)
    # Also keep an aggregate, reviewable report in Git. No private rows or paths.
    write(HERE / 'results-20261103.json', report)
    pp = out / 'releases/ppocr'
    reader = out / 'releases/readernet'
    pp.mkdir(parents=True, exist_ok=True)
    reader.mkdir(parents=True, exist_ok=True)

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    best = out / 'recognizer-results-v1/recognizer-v2/ppocrv6_medium_rec_line_finetune_best'
    exported = out / 'recognizer-results-v1/recognizer-onnx'
    for name in ['model.safetensors', 'config.json', 'preprocessor_config.json']:
        copy(best / name, pp / name)
    for name in ['ppocrv6_bubble_line_rec.onnx', 'ppocrv6_bubble_line_rec_webgpu.onnx', 'config.json', 'preprocessor_config.json']:
        copy(exported / name, pp / 'onnx' / name)
    manifest = read(exported / 'browser_manifest.json')
    manifest['parity'] = parity
    manifest['final_text_postprocess'] = 'ppocr_lexicon_spacing_case_only'
    manifest['source_revision'] = train_public['starting_checkpoint']['revision']
    manifest['training_date'] = DATE
    write(pp / 'onnx/browser_manifest.json', manifest)
    write(pp / 'onnx/ppocrv6_postprocess_rules.json', manifest['postprocess'])
    write(pp / 'onnx/pipeline_manifest.json', {
        'schema_version': 2, 'training_kind': 'ppocrv6_production_prepared_bubble_lines',
        'hf_repo': PP_REPO, 'date': DATE, 'training': train_public,
        'files': {'detector': 'onnx/bubble_line_detector_yolo26n.onnx', 'recognizer': 'onnx/ppocrv6_bubble_line_rec.onnx',
                  'recognizer_webgpu': 'onnx/ppocrv6_bubble_line_rec_webgpu.onnx', 'manifest': 'onnx/browser_manifest.json'},
        'preprocess': {'line_confidence': .10, 'line_padding': 2, 'line_gap': 8, 'image_height': 48,
                       'width_range': [960, 3200], 'padding': '#808080', 'channels': 'BGR', 'normalization': '(uint8 / 255 - 0.5) / 0.5'},
        'postprocess': {'policy': manifest['final_text_postprocess'], 'train_derived_rules_frozen': True},
        'artifacts': report['production_artifacts'], 'metrics': 'metrics/retraining_20261103.json'})
    write(pp / 'dataset_stats.json', report['dataset'])
    write(pp / 'training_metrics.json', train_public)
    write(pp / 'best_checkpoint_metrics.json', {'epoch': train['best_epoch'], 'selection': 'native raw validation CER', 'metrics': train['epochs'][train['best_epoch'] - 1]})
    for short, split in [('validation', 'val'), ('test', 'test')]:
        write(pp / (short + '_summary.json'), {'scope': 'Native CUDA FP32, production-prepared GT crops; historical inspected split', **report['recognition_gt_crops'][split]['candidate']})
    write(pp / 'postprocess_official_metrics.json', {'date': DATE, 'protocol': report['protocol'], 'splits': report['recognition_gt_crops']})
    write(pp / 'metrics/retraining_20261103.json', report)
    copy(HERE / 'cards/ppocr.md', pp / 'README.md')

    copy(out / 'recovered-bubble-detector-cuda-env.pt', reader / 'bubble_detector.pt')
    recovery = read(out / 'training-logs-v1/recovered-bubble-detector.recovery.json')
    cases = recovery.pop('parity')
    recovery['parity'] = {'pages': len(cases), 'positive_detections': sum(r['detections'] for r in cases),
                          'max_box_error_pixels': max(r['max_box_error'] for r in cases),
                          'max_score_error': max(r['max_score_error'] for r in cases)}
    recovery['checkpoint'] = descriptor(reader / 'bubble_detector.pt')
    recovery['note'] += ' This native file is a reconstruction, not the original training checkpoint. Active ONNX is retained byte-for-byte.'
    write(reader / 'retraining/2026-11-03/recovery.json', recovery)
    for name, folder, version in [('yolo800', 'detector-results-v1', '8.4.201'),
                                   ('yolo1024', 'detector-1024-results-v1', '8.4.201'),
                                   ('legacy-head800', 'detector-head-results-v1', '8.4.48')]:
        destination = reader / 'retraining/2026-11-03' / name
        for suffix in ['pt', 'onnx']:
            copy(out / folder / 'run/weights' / ('best.' + suffix), destination / ('bubble_detector.' + suffix))
        config = yaml.safe_load((out / folder / 'run/args.yaml').read_text(encoding='utf-8'))
        for field in ['model', 'data', 'project', 'name', 'save_dir', 'resume']:
            config.pop(field, None)
        rows = list(csv.DictReader((out / folder / 'run/results.csv').open(encoding='utf-8')))
        result = read(out / folder / 'result.json')
        write(destination / 'training.json', {'status': 'candidate_not_promoted', 'ultralytics': version,
              'completed_epochs': len(rows), 'config': config, 'validation': result['validation'],
              'seconds': result['seconds'], 'selection': 'metrics/retraining_20261103.json#detector_selection'})
    write(reader / 'metrics/retraining_20261103.json', report)
    copy(HERE / 'cards/readernet.md', reader / 'README.md')
    reader_manifest = read(out / 'release-sources/Poneglyph-ReaderNet/model_manifest.json')
    reader_manifest['runtime']['bubble_confidence'] = .25
    reader_manifest['runtime']['calibration_date'] = DATE
    reader_manifest['provenance'] = {'active_bubble_weights': 'Retained; candidates trained but did not improve validation operating point',
                                     'parent_revision': 'd97d4cd2903a7ebe49276a5269c4f3b7df608be7',
                                     'native_checkpoint': 'Recovered equivalent inference graph with neutral BN; Ultralytics 8.4.48; freeze running BN statistics for training'}
    # Preserve source layout so relative imports remain correct; inputs are private.
    sources = ['requirements.txt', 'requirements-yolo-legacy.txt', 'train_detector.py', 'recover_detector.py',
               'export_recognizer.py', 'evaluate_recognizer.py', 'test_training.py']
    for destination in [pp, reader]:
        for name in sources:
            copy(HERE / name, destination / 'training_source/docker_scripts/retrain_ppocr_pipeline' / name)
        copy(ROOT / 'docker_scripts/finetune_paddleocr_line_rec/train_ppocrv6_rec.py',
             destination / 'training_source/docker_scripts/finetune_paddleocr_line_rec/train_ppocrv6_rec.py')
    for file in sorted(reader.rglob('*')):
        if file.is_file() and file.name != 'model_manifest.json':
            reader_manifest['artifacts'][file.relative_to(reader).as_posix()] = descriptor(file)
    write(reader / 'model_manifest.json', reader_manifest)
    plan = []
    for repo, folder, parent, deletes in [
        (PP_REPO, pp, train_public['starting_checkpoint']['revision'], ['validation_predictions.csv', 'test_predictions.csv', 'yolo_metrics.json']),
        (READER_REPO, reader, reader_manifest['provenance']['parent_revision'], [])]:
        files = {p.relative_to(folder).as_posix(): descriptor(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        assert all(Path(n).suffix in ['.json', '.md', '.py', '.txt', '.onnx', '.pt', '.safetensors'] for n in files)
        plan.append({'repo_id': repo, 'parent_commit': parent, 'folder': str(folder.relative_to(out)), 'files': files, 'delete': deletes})
    write(out / 'release-plan.json', plan)
    print(json.dumps({'files': {p['repo_id']: len(p['files']) for p in plan}, 'report': str(HERE / 'results-20261103.json'), 'reading_order': report['matched_reading_order']}))


if __name__ == '__main__':
    main()
