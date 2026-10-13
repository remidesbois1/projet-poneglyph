ARG TRAINING_BASE=poneglyph/surya-onnx:20261005
FROM ${TRAINING_BASE}
WORKDIR /app/finetune_lighton3_ocr_bbox
COPY docker_scripts/finetune_lighton3_ocr_bbox/requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY docker_scripts/finetune_lighton3_ocr_bbox/*.py ./
COPY docker_scripts/finetune_lighton_ocr_bbox/training_monitor.py /app/finetune_lighton_ocr_bbox/training_monitor.py
COPY docker_scripts/finetune_surya_ocr_bbox/export_dataset.py /app/finetune_surya_ocr_bbox/export_dataset.py
COPY docker_scripts/common_training /app/common_training
COPY packages/shared/src/llm-prompts.json /app/packages/shared/src/llm-prompts.json
ENV PONEGLYPH_PROMPTS_FILE=/app/packages/shared/src/llm-prompts.json
ENV HF_HOME=/workspace/hf-cache
ENV HF_XET_HIGH_PERFORMANCE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TRITON_CACHE_DIR=/workspace/hf-cache/triton
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/hf-cache/torchinductor
ENV CUDA_MODULE_LOADING=LAZY
ENV TRACKIO_DIR=/workspace/trackio
ENV LIGHTON_BBOX_RICH=0
ENTRYPOINT ["python", "-u"]
CMD ["train.py", "--dataset", "/workspace/dataset", "--output", "/workspace/outputs/rtx5090-20261011"]
