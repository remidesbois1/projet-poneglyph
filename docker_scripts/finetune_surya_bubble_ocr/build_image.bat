@echo off
setlocal

if "%IMAGE_NAME%"=="" set IMAGE_NAME=surya-bubble-ocr-finetune
if "%TAG%"=="" set TAG=latest

echo ==========================================================
echo Building Docker image %IMAGE_NAME%:%TAG%
echo ==========================================================
echo.

docker build -f "%~dp0Dockerfile" -t %IMAGE_NAME%:%TAG% "%~dp0..\.."

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Build failed.
    exit /b %ERRORLEVEL%
)

echo.
echo Build successful: %IMAGE_NAME%:%TAG%
echo.
