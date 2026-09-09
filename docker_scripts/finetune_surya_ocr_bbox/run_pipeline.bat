@echo off
setlocal
cd /d "%~dp0"
echo ==========================================================
echo  Starting Surya OCR 2 Poneglyph BBox Fine-Tuning Pipeline
echo ==========================================================
echo.
echo Required: .env with SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, optional HF_TOKEN
echo.

docker run --rm -it --gpus all --shm-size=8g ^
  --env-file ../../backend/.env ^
  --env-file ../../.env ^
  -v "%cd%\surya_bbox_dataset:/workspace/surya_bbox_dataset" ^
  -v "%cd%\outputs_surya_bbox:/workspace/outputs_surya_bbox" ^
  -v "%cd%\hf-cache:/workspace/hf-cache" ^
  remidesbois/surya-ocr-bbox-finetune:latest

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Pipeline execution failed or was interrupted!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Job done. Check the output folder and Hugging Face repo if upload was enabled.
echo.
pause
