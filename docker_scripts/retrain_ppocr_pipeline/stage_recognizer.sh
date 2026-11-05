#!/usr/bin/env bash
set -euo pipefail
expected_sha="$1"
printf '%s  %s\n' "$expected_sha" /workspace/recognizer-inputs.tar.gz | sha256sum --check -
mkdir -p /opt/poneglyph-rec-data
# Images can be regenerated from the durable archive. Keep the many small
# training files on the container SSD; checkpoints always stay on /workspace.
tar -xzf /workspace/recognizer-inputs.tar.gz -C /workspace/poneglyph --exclude='data/recognizer/images/*'
tar -xzf /workspace/recognizer-inputs.tar.gz -C /opt/poneglyph-rec-data --wildcards 'data/recognizer/images/*'
ln -s /opt/poneglyph-rec-data/data/recognizer/images /workspace/poneglyph/data/recognizer/images
echo RECOGNIZER_INPUTS_READY
