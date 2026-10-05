@echo off
setlocal
pushd "%~dp0"

if "%IMAGE_NAME%"=="" set IMAGE_NAME=surya-bubble-ocr-finetune
if "%TAG%"=="" set TAG=latest
set ENV_ARGS=
if exist "..\..\backend\.env" set ENV_ARGS=--env-file "..\..\backend\.env"
if exist "..\..\.env" set ENV_ARGS=%ENV_ARGS% --env-file "..\..\.env"
if exist ".env" set ENV_ARGS=%ENV_ARGS% --env-file ".env"

echo ==========================================================
echo Running Surya bubble OCR pipeline in Docker
echo ==========================================================
echo.
echo Requires ..\..\.env or a local .env with Supabase credentials.
echo.

docker run --gpus all --ipc=host --shm-size 24g %ENV_ARGS% ^
    -v "%cd%\surya_bubble_dataset:/workspace/surya_bubble_dataset" ^
    -v "%cd%\outputs_surya_bubble_ocr:/workspace/outputs_surya_bubble_ocr" ^
    -v "%USERPROFILE%\.cache\huggingface:/workspace/hf-cache" ^
    -v "poneglyph-surya-kernels:/workspace/kernel-cache" ^
    -e TRITON_CACHE_DIR=/workspace/kernel-cache/triton ^
    -e TORCHINDUCTOR_CACHE_DIR=/workspace/kernel-cache/torchinductor ^
    %IMAGE_NAME%:%TAG%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Pipeline failed or was interrupted.
    exit /b %ERRORLEVEL%
)

echo.
echo Pipeline complete.
popd
echo.
