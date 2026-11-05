#!/usr/bin/env bash
set -euo pipefail
trap 'code=$?; printf "%s\n" "$code" > /workspace/detector-1024.exit' EXIT
cd /workspace/poneglyph
unset PYTHONPATH
mkdir -p /workspace/yolo-config-modern
export YOLO_CONFIG_DIR=/workspace/yolo-config-modern
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HOME=/workspace/hf-cache
python3 docker_scripts/retrain_ppocr_pipeline/train_detector.py \
  --dataset data --output /workspace/outputs/detector-1024 \
  --model /workspace/outputs/detector/run/weights/best.pt --image-size 1024 \
  --epochs 40 --patience 12 --batch 16 --learning-rate .0001 \
  --mosaic 0 --scale .1 --translate .02 --close-mosaic 0 \
  --warmup-epochs 1 --warmup-bias-lr .0001 --initial-eval
