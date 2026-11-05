#!/usr/bin/env bash
set -euo pipefail
trap 'code=$?; printf "%s\n" "$code" > /workspace/quality.exit' EXIT
cd /workspace/poneglyph
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HOME=/workspace/hf-cache
python3 docker_scripts/retrain_ppocr_pipeline/evaluate_recognizer.py --model baseline-recognizer --dataset data/recognizer --output /workspace/outputs/baseline-recognizer-eval --split train
