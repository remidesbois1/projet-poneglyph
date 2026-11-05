"""SSH/SCP for this task's attributed pod; credentials never enter archives."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from runpod_cli import cli_env


def connect_args():
    state_path = Path(__file__).parent / 'outputs/runpod-state.json'
    state = json.loads(state_path.read_text())
    if state.get('created_by') != 'poneglyph-retrain-20261103':
        raise ValueError('Unknown pod provenance')
    info = state.get('ssh_direct')
    if not info:
        request = Request('https://api.runpod.io/v2/pods/' + state['pod_id'],
                          headers={'Authorization': 'Bearer ' + cli_env()['RUNPOD_API_KEY']})
        for attempt in range(3):
            try:
                pod = json.load(urlopen(request, timeout=30))
                break
            except (HTTPError, URLError) as exc:
                if isinstance(exc, HTTPError) and exc.code < 500 or attempt == 2:
                    raise
                time.sleep(attempt + 1)
        info = pod.get('ssh', {}).get('direct', {})
        if info.get('host') and info.get('port'):
            state['ssh_direct'] = {k: info[k] for k in ['host', 'port', 'username'] if k in info}
            state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
    if not info.get('host') or not info.get('port'):
        raise RuntimeError('No direct SSH yet')
    key = Path.home() / '.runpod/ssh/poneglyph-training-20261103'
    known_hosts = Path.home() / '.runpod/ssh/poneglyph-training-20261103-known-hosts'
    common = ['-i', str(key), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
              '-o', 'UserKnownHostsFile=' + str(known_hosts), '-o', 'ConnectTimeout=20']
    return common, str(info['port']), info.get('username', 'root') + '@' + info['host']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['exec', 'upload', 'download'])
    parser.add_argument('source')
    parser.add_argument('target', nargs='?')
    args = parser.parse_args()
    common, port, host = connect_args()
    if args.action == 'exec':
        command = ['ssh', *common, '-p', port, host, args.source]
    elif args.action == 'upload':
        if not args.target or not args.target.startswith('/workspace/'):
            raise ValueError('Uploads must stay inside /workspace')
        command = ['scp', *common, '-P', port, str(Path(args.source).resolve()), host + ':' + args.target]
    else:
        if not args.source.startswith('/workspace/') or not args.target:
            raise ValueError('Only retrieve task workspace artifacts')
        command = ['scp', *common, '-P', port, host + ':' + args.source, str(Path(args.target).resolve())]
    raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
    main()
