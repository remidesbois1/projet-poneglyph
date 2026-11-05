#!/usr/bin/env bash
set -euo pipefail
trap 'code=$?; printf "%s\n" "$code" > /workspace/detector-warmstart-v2.exit' EXIT
cd /workspace/poneglyph
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HOME=/workspace/hf-cache
export PYTHONPATH=/workspace/legacy-ultralytics YOLO_CONFIG_DIR=/workspace/yolo-config-legacy
python3 docker_scripts/retrain_ppocr_pipeline/train_detector.py \
  --dataset data --output /workspace/outputs/detector-warmstart-v2 \
  --model /workspace/recovered-bubble-detector.pt \
  --epochs 40 --patience 12 --batch 16 --learning-rate .00003 \
  --mosaic 0 --scale .1 --translate .02 --close-mosaic 0 \
  --warmup-epochs 1 --warmup-bias-lr .00003 --freeze-batchnorm --initial-eval --no-amp
