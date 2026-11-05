#!/usr/bin/env bash
set -euo pipefail
trap 'code=$?; printf "%s\n" "$code" > /workspace/detector.exit' EXIT
cd /workspace/poneglyph
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HOME=/workspace/hf-cache
export HF_HUB_DISABLE_TELEMETRY=1
python3 docker_scripts/retrain_ppocr_pipeline/train_detector.py --dataset data --output /workspace/outputs/detector --epochs 100 --batch 16 --patience 20
