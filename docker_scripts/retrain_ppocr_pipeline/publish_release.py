"""Publish the explicit reviewed file allowlist, atomically per HF repository."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=Path, default=Path(__file__).parent / 'outputs')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    out = args.outputs.resolve()
    plan = json.loads((out / 'release-plan.json').read_text(encoding='utf-8'))
    state_path = out / 'published-revisions.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    api = HfApi()
    for release in plan:
        repo = release['repo_id']
        folder = (out / release['folder']).resolve()
        if not folder.is_relative_to(out / 'releases'):
            raise ValueError('Release outside the curated directory')
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
        if actual != set(release['files']):
            raise ValueError('Release contains unreviewed or missing files')
        for name, metadata in release['files'].items():
            file = (folder / name).resolve()
            if not file.is_relative_to(folder) or digest(file) != metadata['sha256'] or file.stat().st_size != metadata['bytes']:
                raise ValueError('Release file changed: ' + name)
        prior = state.get(repo)
        if prior and prior['plan_sha256'] != digest(out / 'release-plan.json'):
            raise ValueError('Published state belongs to a different release plan')
        head = api.model_info(repo).sha
        if head != (prior['revision'] if prior else release['parent_commit']):
            raise ValueError('Repository changed since review: ' + repo)
        print(json.dumps({'repo': repo, 'parent': head, 'add': len(actual), 'delete': release['delete'], 'publish': args.publish}), flush=True)
        if not args.publish:
            continue
        if prior:
            revision = prior['revision']
        else:
            operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(folder / name)) for name in sorted(actual)]
            operations += [CommitOperationDelete(path_in_repo=name) for name in release['delete']]
            result = api.create_commit(repo_id=repo, parent_commit=head, operations=operations,
                                       commit_message='Retrain and validate production PPOCR pipeline on RunPod (2026-11-03)')
            revision = result.oid
            state[repo] = {'revision': revision, 'parent_commit': head, 'url': result.commit_url,
                           'plan_sha256': digest(out / 'release-plan.json')}
            state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        if api.model_info(repo).sha != revision:
            raise RuntimeError('Published revision is no longer the head; inspect before updating registry')
        # Redownload active weights and manifests at the immutable revision. Hash
        # every published asset against Hub metadata, including native candidates.
        info = api.model_info(repo, revision=revision, files_metadata=True)
        remote = {f.rfilename: f for f in info.siblings}
        for name, metadata in release['files'].items():
            lfs = remote[name].lfs
            if lfs and lfs.sha256 != metadata['sha256']:
                raise RuntimeError('Hub LFS hash mismatch: ' + name)
        names = ['model.safetensors', 'onnx/ppocrv6_bubble_line_rec_webgpu.onnx', 'onnx/browser_manifest.json', 'metrics/retraining_20261103.json'] if repo.endswith('bubble-line-rec') else ['bubble_detector.pt', 'bubble_detector.onnx', 'model_manifest.json', 'metrics/retraining_20261103.json']
        verified = {}
        for name in names:
            path = Path(hf_hub_download(repo, name, revision=revision, local_dir=out / 'published-downloads' / repo.split('/')[-1]))
            expected = release['files'].get(name)
            if expected is None:
                expected = json.loads((folder / 'model_manifest.json').read_text(encoding='utf-8'))['artifacts'][name]
            if digest(path) != expected['sha256']:
                raise RuntimeError('Downloaded artifact hash mismatch: ' + name)
            verified[name] = digest(path)
        state[repo]['downloaded_sha256'] = verified
        state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'repo': repo, 'revision': revision, 'downloaded_and_verified': list(verified)}), flush=True)


if __name__ == '__main__':
    main()
