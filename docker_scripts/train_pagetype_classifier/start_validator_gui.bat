@echo off
title Page Type Dataset Validator
cd /d "%~dp0"
python dataset_validator_gui.py
if %ERRORLEVEL% NEQ 0 (
    "C:\Users\remis\AppData\Local\Programs\Python\Python311\python.exe" dataset_validator_gui.py
)
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Une erreur est survenue lors du lancement de l'application.
    pause
)
