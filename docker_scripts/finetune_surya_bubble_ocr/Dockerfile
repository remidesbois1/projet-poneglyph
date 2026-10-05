# Build with the repository root as context. Default: RTX 5090 / SM120.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ARG TORCH_CUDA_ARCH_LIST=12.0
ARG CAUSAL_CONV_CUDA_ARCH=120

ENV CUDA_HOME=/usr/local/cuda
ENV VIRTUAL_ENV=/opt/surya-venv
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=8

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    python3-venv \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv --system-site-packages ${VIRTUAL_ENV}
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

RUN python -m pip install --no-cache-dir --upgrade pip "setuptools<82" wheel

COPY docker_scripts/finetune_surya_bubble_ocr/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY docker_scripts/finetune_surya_bubble_ocr/patch_causal_conv1d_arch.py /tmp/patch_causal_conv1d_arch.py
RUN mkdir -p /tmp/causal-conv1d-src \
    && curl -fsSLo /tmp/causal-conv1d.tar.gz \
       https://files.pythonhosted.org/packages/source/c/causal-conv1d/causal_conv1d-1.6.2.post1.tar.gz \
    && tar -xzf /tmp/causal-conv1d.tar.gz -C /tmp/causal-conv1d-src --strip-components=1 \
    && python /tmp/patch_causal_conv1d_arch.py \
       /tmp/causal-conv1d-src/setup.py \
       --compute-capability ${CAUSAL_CONV_CUDA_ARCH} \
    && CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install \
       --no-cache-dir --no-build-isolation /tmp/causal-conv1d-src \
    && rm -rf /tmp/causal-conv1d-src /tmp/causal-conv1d.tar.gz \
       /tmp/patch_causal_conv1d_arch.py
RUN python -c "import causal_conv1d, fla; print('Qwen3.5 linear-attention kernels installed')"

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/hf-cache
ENV HF_XET_HIGH_PERFORMANCE=1
ENV SURYA_DATASET_DIR=/workspace/surya_bubble_dataset
ENV SURYA_OUTPUT_DIR=/workspace/outputs_surya_bubble_ocr
ENV SURYA_REQUIRE_FAST_LINEAR_ATTENTION=1
ENV SURYA_EXPECTED_COMPUTE_CAPABILITY=${TORCH_CUDA_ARCH_LIST}
ENV SURYA_PROFILE=auto
ENV SURYA_GEN_EVAL_BATCH=32
ENV SURYA_GEN_EVAL_MAX_SAMPLES=512
ENV SURYA_LIGER=1
ENV SURYA_IMAGE_CACHE_ITEMS=8192
ENV SURYA_SKIP_UPLOAD=1
ENV SURYA_MAX_NEW_TOKENS=256
ENV SURYA_AUTO_FIND_BATCH_SIZE=0
ENV SURYA_DATALOADER_WORKERS=0
ENV SURYA_EVAL_ON_START=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TOKENIZERS_PARALLELISM=false
ENV TORCHINDUCTOR_COMPILE_THREADS=1
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/hf-cache/torchinductor
ENV TRITON_CACHE_DIR=/workspace/hf-cache/triton
ENV CUDA_MODULE_LOADING=LAZY

COPY docker_scripts/common_training /app/common_training
COPY packages/shared/src/llm-prompts.json /app/packages/shared/src/llm-prompts.json
ENV PONEGLYPH_PROMPTS_FILE=/app/packages/shared/src/llm-prompts.json
COPY docker_scripts/finetune_surya_bubble_ocr /app/finetune_surya_bubble_ocr

WORKDIR /app/finetune_surya_bubble_ocr

CMD ["python", "run_pipeline.py"]
