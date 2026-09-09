# Build with the project root as context so the shared prompt registry can be
# embedded alongside the training code without duplicating it.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ARG TORCH_CUDA_ARCH_LIST=12.0
ARG CAUSAL_CONV_CUDA_ARCH=120

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONUNBUFFERED=1
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV HF_HOME=/workspace/hf-cache
ENV HF_XET_HIGH_PERFORMANCE=1
ENV SURYA_BBOX_DATASET_DIR=/workspace/surya_bbox_dataset
ENV SURYA_BBOX_OUTPUT_DIR=/workspace/outputs_surya_bbox
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=4
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV SURYA_BBOX_PROFILE=auto
ENV SURYA_BBOX_REQUIRE_FAST_LINEAR_ATTENTION=1
ENV SURYA_BBOX_EXPECTED_COMPUTE_CAPABILITY=12.0
ENV PONEGLYPH_PROMPTS_FILE=/app/packages/shared/src/llm-prompts.json
ENV TRITON_CACHE_DIR=/workspace/hf-cache/triton
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/hf-cache/torchinductor
ENV TORCHINDUCTOR_COMPILE_THREADS=1
ENV CUDA_MODULE_LOADING=LAZY

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip "setuptools<82" wheel

COPY docker_scripts/finetune_surya_ocr_bbox/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
# Use the existing, version-checked build helper, with SM120 instead of SM86.
COPY docker_scripts/finetune_surya_bubble_ocr/patch_causal_conv1d_arch.py /tmp/patch_causal_conv1d_arch.py
RUN mkdir -p /tmp/causal-conv1d-src \
    && curl -fsSLo /tmp/causal-conv1d.tar.gz \
       https://files.pythonhosted.org/packages/source/c/causal-conv1d/causal_conv1d-1.6.2.post1.tar.gz \
    && tar -xzf /tmp/causal-conv1d.tar.gz -C /tmp/causal-conv1d-src --strip-components=1 \
    && python /tmp/patch_causal_conv1d_arch.py /tmp/causal-conv1d-src/setup.py \
       --compute-capability ${CAUSAL_CONV_CUDA_ARCH} \
    && CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install \
       --no-cache-dir --no-build-isolation /tmp/causal-conv1d-src \
    && rm -rf /tmp/causal-conv1d-src /tmp/causal-conv1d.tar.gz /tmp/patch_causal_conv1d_arch.py
# Docker build stages do not have access to the host GPU, so FLA deliberately
# selects its CPU device here. Checking Qwen3.5's runtime fast-path flag at build
# time would therefore produce a false failure even when the CUDA extensions were
# compiled correctly. Import the dependencies here; the trainer performs the
# strict CUDA + SM120 + Qwen3.5 fast-path check once the container has a GPU.
RUN python -c "import importlib.util, causal_conv1d; assert importlib.util.find_spec('fla') is not None; assert importlib.util.find_spec('transformers.models.qwen3_5') is not None; print('Qwen3.5 DeltaNet dependencies installed; GPU fast-path will be validated at runtime')"

COPY docker_scripts/common_training /app/common_training
COPY packages/shared/src/llm-prompts.json /app/packages/shared/src/llm-prompts.json
RUN python -c "from common_training.prompts import get_prompt; p=get_prompt('ocr_page_bbox_training_lines'); assert '[x1,y1,x2,y2]' in p; print('Shared Surya bbox training prompt registry OK')"
COPY docker_scripts/finetune_surya_ocr_bbox /app/finetune_surya_ocr_bbox
WORKDIR /app/finetune_surya_ocr_bbox

CMD ["python", "run_pipeline.py"]
