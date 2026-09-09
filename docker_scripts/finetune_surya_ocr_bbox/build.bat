@echo off
setlocal
cd /d "%~dp0"
set DOCKER_USER=remidesbois
set IMAGE_NAME=surya-ocr-bbox-finetune
set TAG=latest

echo ==========================================================
echo    Building %DOCKER_USER%/%IMAGE_NAME%:%TAG% (no push)
echo ==========================================================
echo.

docker build -f Dockerfile -t %DOCKER_USER%/%IMAGE_NAME%:%TAG% ..\..

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo    Build failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo    Image built successfully. Nothing was pushed.
echo.
pause
