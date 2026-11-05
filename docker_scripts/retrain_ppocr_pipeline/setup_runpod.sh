#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/poneglyph
python3 --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /workspace /dev/shm
python3 -c 'import torch; assert torch.cuda.is_available(); print("TEMPLATE_TORCH",torch.__version__,torch.version.cuda)'
# Reuse the CUDA wheels in the official template. Dependency resolution must
# never replace them with CPU wheels or download another Torch release.
printf '%s\n' 'torch==2.8.0' 'torchvision==0.23.0' > /workspace/template-constraints.txt
python3 -m pip install --break-system-packages -c /workspace/template-constraints.txt -r /workspace/retrain-requirements.txt
python3 -c 'import torch,transformers,ultralytics,onnxruntime; assert torch.cuda.is_available(); print("GPU_READY",torch.cuda.get_device_name(0));print("VERSIONS",torch.__version__,transformers.__version__,ultralytics.__version__,onnxruntime.__version__)'
