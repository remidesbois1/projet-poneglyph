FROM remidesbois/surya-ocr-bbox-finetune:latest@sha256:905b90f47799d163aacd8af4fbaa98b852e875495adb82c283dd0ca62559c0cd
WORKDIR /export
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY *.py ./
COPY README.md ./
ENTRYPOINT ["python", "-u", "export_onnx.py"]
