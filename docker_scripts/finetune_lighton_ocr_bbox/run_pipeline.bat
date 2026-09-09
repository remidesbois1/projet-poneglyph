@echo off
chcp 65001 >nul
echo ==========================================================
echo Starting LightOnOCR-2-1B BBox pipeline - RTX 5090
echo ==========================================================
echo.
echo Resolution fixe : 1500 px cote long
echo Requis : ../../.env avec SUPABASE_URL et SUPABASE_SERVICE_ROLE_KEY
echo.

docker run --rm -it --gpus all --shm-size=8g --env-file ../../backend/.env --env-file ../../.env ^
    -v "%cd%\lighton_bbox_dataset:/workspace/lighton_bbox_dataset" ^
    -v "%cd%\outputs_lighton_bbox:/workspace/outputs_lighton_bbox" ^
    -v "%cd%\hf-cache:/workspace/hf-cache" ^
    lighton-ocr-bbox-finetune:latest

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Pipeline execution failed or was interrupted!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Pipeline termine. Le modele final, son benchmark et le dashboard sont dans outputs_lighton_bbox.
echo.
pause
