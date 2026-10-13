FROM poneglyph/lighton3-bbox:5090
# Match the base image's PyTorch CUDA 12.8/cuDNN 9 libraries. ORT 1.27's
# default Linux wheel requires CUDA 13; the independent browser runtime is 1.27.
RUN python -m pip uninstall -y onnxruntime \
    && python -m pip install --no-cache-dir onnxruntime-gpu==1.26.0
WORKDIR /export
ENTRYPOINT ["python", "-u"]
