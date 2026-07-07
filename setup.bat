@echo off
chcp 65001 >nul
title ABSA Web System - Setup

echo ============================================
echo   ABSA 方面级情感分析 Web 系统 - 一键启动
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ first.
    pause
    exit /b 1
)
echo [OK] Python found:
python --version

:: Check Git LFS (for BERT model)
echo.
echo [INFO] Checking Git LFS for BERT model...
git lfs version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Git LFS not found. BERT model may not download correctly.
    echo        Install from: https://git-lfs.com
) else (
    git lfs pull
    echo [OK] Git LFS ready
)

:: Check if state_dict exists
if not exist "state_dict\bert_spc_laptop_val_acc_0.7868" (
    echo [ERROR] BERT model not found!
    echo        Please run: git lfs install ^&^& git lfs pull
    pause
    exit /b 1
)
echo [OK] Model weights found

:: Install dependencies
echo.
echo [INFO] Installing Python dependencies...
pip install -q torch transformers flask flask-cors scikit-learn numpy Pillow
if %errorlevel% neq 0 (
    echo [ERROR] Package installation failed.
    pause
    exit /b 1
)
echo [OK] Dependencies installed

:: Set environment
set PYTHONIOENCODING=utf-8
set HF_ENDPOINT=https://hf-mirror.com

:: Start
echo.
echo ============================================
echo   Starting ABSA Web Server...
echo   Open http://127.0.0.1:5000 in browser
echo   Press Ctrl+C to stop
echo ============================================
echo.

python app.py

pause
