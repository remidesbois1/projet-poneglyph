#!/usr/bin/env bash
# Runs on the pod, from /workspace/indra: two detector fine-tunes side by side on one GPU.
set -euo pipefail
cd /workspace/indra
python3 -c 'import torch; assert torch.cuda.is_available(); print("GPU", torch.cuda.get_device_name(0), torch.__version__)'
printf '%s\n' "torch==$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')" > constraints.txt
python3 -m pip install -q --break-system-packages -c constraints.txt -r code/requirements-train.txt
WORKERS=$(( $(nproc) / 2 )); WORKERS=$(( WORKERS > 6 ? 6 : WORKERS ))
mkdir -p det_runs
nohup python3 code/train_det.py --data data --out det_runs/d1_lr1e-3 --workers "$WORKERS" --lr 1e-3 > det_runs/d1_lr1e-3.log 2>&1 &
nohup python3 code/train_det.py --data data --out det_runs/d2_lr3e-4 --workers "$WORKERS" --lr 3e-4 > det_runs/d2_lr3e-4.log 2>&1 &
wait
echo DET_TRAINING_DONE
