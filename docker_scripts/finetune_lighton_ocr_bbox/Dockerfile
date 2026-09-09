FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_MODULE_LOADING=LAZY
ENV HF_HOME=/workspace/hf-cache
ENV HF_HUB_DISABLE_XET=1
ENV LIGHTON_BBOX_DATASET_DIR=/workspace/lighton_bbox_dataset
ENV LIGHTON_BBOX_OUTPUT_DIR=/workspace/outputs_lighton_bbox
ENV LIGHTON_BBOX_PROFILE=auto
ENV LIGHTON_BBOX_IMAGE_LONGEST_EDGE=1500
ENV PONEGLYPH_PROMPTS_FILE=/app/packages/shared/src/llm-prompts.json

WORKDIR /app

COPY docker_scripts/finetune_lighton_ocr_bbox/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY docker_scripts/common_training /app/common_training
COPY packages/shared/src/llm-prompts.json /app/packages/shared/src/llm-prompts.json
RUN python -c "from common_training.prompts import get_prompt; p=get_prompt('ocr_page_bbox_training_lines'); assert '[x1,y1,x2,y2]' in p and 'JSON' in p; print('LightOn bbox prompt registry OK')"

COPY docker_scripts/finetune_lighton_ocr_bbox /app/finetune_lighton_ocr_bbox
WORKDIR /app/finetune_lighton_ocr_bbox

CMD ["python", "run_pipeline.py"]
