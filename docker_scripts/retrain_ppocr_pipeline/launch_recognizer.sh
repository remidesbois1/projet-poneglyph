#!/usr/bin/env bash
set -euo pipefail
trap 'code=$?; printf "%s\n" "$code" > /workspace/recognizer.exit' EXIT
cd /workspace/poneglyph
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HOME=/workspace/hf-cache
export HF_HUB_DISABLE_TELEMETRY=1
python3 docker_scripts/finetune_paddleocr_line_rec/train_ppocrv6_rec.py \
  --model-id baseline-recognizer --dataset-dir data/recognizer \
  --output-dir /workspace/outputs/recognizer-v2 \
  --prepared-inputs --batch-size 1 --grad-accum-steps 16 --epochs 5 \
  --learning-rate 6e-6 --backbone-learning-rate 6e-7 \
  --train-backbone --freeze-batchnorm --short-loss-weight 1 --short-oversample 1 \
  --lr-scheduler cosine --warmup-ratio .05 --image-width 960 --threads 4 \
  --initial-eval --patience 2 --require-cuda --no-amp --pin-memory --save-every-epoch \
  --exclude-train-pairs training-pair-exclusions.json
best=/workspace/outputs/recognizer-v2/ppocrv6_medium_rec_line_finetune_best
for split in val test; do
  python3 docker_scripts/retrain_ppocr_pipeline/evaluate_recognizer.py --model baseline-recognizer --dataset data/recognizer --output /workspace/outputs/baseline-recognizer-eval --split "$split"
  python3 docker_scripts/retrain_ppocr_pipeline/evaluate_recognizer.py --model "$best" --dataset data/recognizer --output /workspace/outputs/recognizer-eval --split "$split"
done
python3 docker_scripts/retrain_ppocr_pipeline/export_recognizer.py --model "$best" --dataset data --output /workspace/outputs/recognizer-onnx --baseline-manifest baseline-browser-manifest.json
echo RECOGNIZER_DONE
