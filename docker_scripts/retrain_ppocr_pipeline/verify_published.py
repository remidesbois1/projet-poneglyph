"""Fetch the complete active runtime bundle at the published immutable commits."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

from huggingface_hub import HfApi, hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=Path, default=Path(__file__).parent / 'outputs')
    args = parser.parse_args()
    out = args.outputs.resolve()
    state = json.loads((out / 'published-revisions.json').read_text(encoding='utf-8'))
    report = json.loads((out / 'release-report.json').read_text(encoding='utf-8'))
    destination = out / 'models-published'
    destination.mkdir(parents=True, exist_ok=True)
    verified = {}
    for repo, release in state.items():
        revision = release['revision']
        if HfApi().model_info(repo).sha != revision:
            raise RuntimeError('Published repository moved; inspect before updating pins')
        pp = repo.endswith('bubble-line-rec')
        files = ['onnx/browser_manifest.json', 'onnx/bubble_line_detector_yolo26n.onnx',
                 'onnx/ppocrv6_bubble_line_rec.onnx', 'onnx/ppocrv6_bubble_line_rec_webgpu.onnx'] if pp else ['bubble_detector.onnx', 'panel_detector.onnx', 'ordering.onnx']
        for name in files:
            cached = Path(hf_hub_download(repo, name, revision=revision))
            sha = hashlib.sha256(cached.read_bytes()).hexdigest()
            expected = report['production_artifacts'].get(Path(name).name)
            if expected and sha != expected['sha256']:
                raise RuntimeError('Active artifact differs from evaluated weights: ' + name)
            shutil.copyfile(cached, destination / Path(name).name)
            verified[repo + '/' + name] = sha
        # Check the public, unauthenticated URL actually used by the web app.
        probe = 'onnx/browser_manifest.json' if pp else 'model_manifest.json'
        origin = 'http://localhost:3000'
        request = Request(f'https://huggingface.co/{repo}/resolve/{revision}/{probe}', headers={'Origin': origin})
        with urlopen(request, timeout=30) as response:
            remote = json.load(response)
            if response.headers.get('Access-Control-Allow-Origin') not in ['*', origin]:
                raise RuntimeError('Published manifest lacks browser CORS access')
        if pp:
            if remote['final_text_postprocess'] != 'ppocr_lexicon_spacing_case_only' or 'items' in remote['parity']:
                raise RuntimeError('Published preprocessing/privacy contract is wrong')
        elif remote['runtime']['bubble_confidence'] != .25:
            raise RuntimeError('Published detector calibration differs')
    if len(verified) != 7:
        raise RuntimeError('Both published repositories are required')
    result = {'revisions': {repo: release['revision'] for repo, release in state.items()},
              'active_files_sha256': verified, 'public_manifest_access_and_cors': 'passed'}
    (out / 'published-verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
