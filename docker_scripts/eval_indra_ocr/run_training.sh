#!/usr/bin/env bash
# Runs on the pod, from /workspace/indra: three recognizer fine-tunes side by side on one GPU
# (the model has 1.25M parameters; data loading, not the GPU, is the bottleneck).
set -euo pipefail
cd /workspace/indra
python3 -c 'import torch; assert torch.cuda.is_available(); print("GPU", torch.cuda.get_device_name(0), torch.__version__)'
printf '%s\n' "torch==$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')" > constraints.txt
python3 -m pip install -q --break-system-packages -c constraints.txt -r code/requirements-train.txt
WORKERS=$(( $(nproc) / 3 )); WORKERS=$(( WORKERS > 12 ? 12 : WORKERS ))
mkdir -p runs
run() {  # name, extra args
  name=$1; shift
  nohup python3 code/train_rec.py --data data --out "runs/$name" --workers "$WORKERS" "$@" > "runs/$name.log" 2>&1 &
}
run r1_lr1e-3 --lr 1e-3 --epochs 40
run r2_lr3e-4 --lr 3e-4 --epochs 40
run r3_upper_lr1e-3 --lr 1e-3 --epochs 40 --case upper
wait
echo TRAINING_DONE
