"""Use runpodctl with the key saved by either current or legacy browser login."""
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def cli_env():
    env = os.environ.copy()
    config_path = Path.home() / '.runpod/config.toml'
    if not env.get('RUNPOD_API_KEY') and config_path.exists():
        config = tomllib.loads(config_path.read_text(encoding='utf-8'))
        key = config.get('apikey') or config.get('default', {}).get('api_key')
        if key:
            env['RUNPOD_API_KEY'] = key
    return env


def run(*args, **kwargs):
    binary = shutil.which('runpodctl') or str(Path.home() / '.local/bin/runpodctl.exe')
    return subprocess.run([binary, *args], env=cli_env(), **kwargs)


if __name__ == '__main__':
    sys.exit(run(*sys.argv[1:]).returncode)
