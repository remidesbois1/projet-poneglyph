FROM poneglyph/lighton3-bbox:5090
WORKDIR /app/finetune_lighton3_bubble_ocr
COPY docker_scripts/finetune_surya_bubble_ocr/*.py /app/finetune_surya_bubble_ocr/
COPY docker_scripts/common_training /app/common_training
COPY packages/shared/src/llm-prompts.json /packages/shared/src/llm-prompts.json
COPY docker_scripts/finetune_lighton3_bubble_ocr/*.py ./
ENV LIGHTON3_BUBBLE_DATASET_DIR=/data/bubbles
ENV LIGHTON3_BUBBLE_OUTPUT_DIR=/workspace/outputs/rtx5090-20261012-bubble
ENV HF_HOME=/hf-cache
ENV TRACKIO_DIR=/workspace/trackio
ENV TRITON_CACHE_DIR=/kernel-cache/triton
ENV TORCHINDUCTOR_CACHE_DIR=/kernel-cache/torchinductor
ENV LIGHTON3_BUBBLE_REQUIRE_FAST_LINEAR_ATTENTION=1
ENV LIGHTON3_BUBBLE_EXPECTED_COMPUTE_CAPABILITY=12.0
ENV SURYA_RICH=0
ENTRYPOINT ["python", "-u"]
