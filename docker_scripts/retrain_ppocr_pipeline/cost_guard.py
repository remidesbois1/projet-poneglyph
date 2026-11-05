"""Local independent deadline guard, stopping only the pod created by this run.

It is a secondary guard while the local computer remains awake and connected;
it is not a server-side Runpod scheduling guarantee.
"""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from runpod_cli import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True, type=Path)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding='utf-8'))
    if state.get('created_by') != 'poneglyph-retrain-20261103' or not state.get('pod_id'):
        raise ValueError('Refusing to manage an unattributed pod')
    deadline = datetime.fromisoformat(state['deadline_utc']).timestamp()
    while time.time() < deadline:
        if args.state.with_suffix('.cleaned').exists():
            return
        time.sleep(min(30, max(0, deadline - time.time())))
    # Keep checkpoints on the network volume. Never terminate or delete storage
    # automatically when local artifact retrieval might still be incomplete.
    result = run('pod', 'stop', state['pod_id'], capture_output=True, text=True)
    report = dict(time=datetime.now(timezone.utc).isoformat(), returncode=result.returncode, output=result.stdout, stderr=result.stderr)
    args.state.with_suffix('.guard-result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Pod deadline stop failed; see guard-result.json')


if __name__ == '__main__':
    main()
