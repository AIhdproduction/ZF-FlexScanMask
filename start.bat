@echo off
:: ══════════════════════════════════════════════════════════════════
::  Z+F FlexScanMask - Setup and Launcher
::  Window stays open on errors. On success it closes automatically.
:: ══════════════════════════════════════════════════════════════════
if "%~1"=="__run__" goto :MAIN
cmd /k ""%~f0" __run__"
exit /b 0

:MAIN
setlocal enabledelayedexpansion
title Z+F FlexScanMask Setup
mode con cols=80 lines=50

echo.
echo  ================================================================
echo    Z+F FlexScanMask
echo    Anonymization for Z+F fisheye images
echo    Powered by SAM3
echo  ================================================================
echo.

cd /d "%~dp0"
echo [INFO] Working directory: %cd%
echo.

:: ─── Skip setup if already done ────────────────────────────────────
if exist ".setup_complete" (
    echo [OK] Setup already complete.
    echo.
    if not exist "venv\Scripts\activate.bat" (
        echo  [ERROR] Virtual environment missing!
        echo          Delete ".setup_complete" and run start.bat again.
        echo.
        echo  Type "exit" to close.
        goto :EOF
    )
    call venv\Scripts\activate.bat
    echo  [CHECK] Verifying dependencies ...
    pip install -r requirements.txt --quiet --timeout 60
    echo  [OK] Dependencies verified.
    echo.
    goto :LAUNCH
)

echo  IMPORTANT: Do NOT close this window during setup!
echo  First-time setup takes 15-30 minutes depending on your connection.
echo.
echo  ================================================================
echo.
echo  === FIRST-TIME SETUP ===
echo.
echo    Step 1/6  Check Python
echo    Step 2/6  Create virtual environment
echo    Step 3/6  Install PyTorch with CUDA
echo    Step 4/6  Install dependencies
echo    Step 5/6  Install SAM3 from GitHub
echo    Step 6/6  Download SAM3 checkpoint
echo.
echo  ----------------------------------------------------------------
echo   Starting in 5 seconds ...
echo  ----------------------------------------------------------------
ping -n 6 127.0.0.1 >nul 2>nul
echo.

:: ─── Step 1: Check Python ──────────────────────────────────────────
echo.
echo  [Step 1/6] Checking Python ...
echo  ----------------------------------------------------------------
python --version 2>nul
if errorlevel 1 (
    echo.
    echo  [ERROR] Python not found!
    echo          Install Python 3.12: https://www.python.org/downloads/
    echo.
    echo  Type "exit" to close.
    goto :EOF
)
for /f "tokens=2" %%v in ('python --version 2^>nul') do set PYVER=%%v
echo  [OK] Python !PYVER! found.

:: ─── Step 2: Virtual environment ───────────────────────────────────
echo.
echo  [Step 2/6] Setting up virtual environment ...
echo  ----------------------------------------------------------------
if exist "venv\Scripts\activate.bat" (
    echo  [OK] Virtual environment already exists.
) else (
    python -m venv venv
    if not exist "venv\Scripts\activate.bat" (
        echo  [ERROR] Could not create virtual environment.
        goto :EOF
    )
    echo  [OK] Virtual environment created.
)
call venv\Scripts\activate.bat
echo  [OK] Virtual environment activated.

:: ─── Step 3: PyTorch with CUDA ─────────────────────────────────────
echo.
echo  [Step 3/6] Installing PyTorch 2.10 with CUDA 12.8 ...
echo  ----------------------------------------------------------------
python -c "import torch" 2>nul
if not errorlevel 1 (
    echo  [OK] PyTorch already installed - skipping.
    goto :TORCH_OK
)
echo  [INFO] Download ~3 GB. Do NOT click inside this window!
echo.
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128 --progress-bar raw --timeout 120
if errorlevel 1 (
    echo  [WARNING] CUDA version failed. Trying CPU-only ...
    pip install torch torchvision --progress-bar raw --timeout 120
    if errorlevel 1 (
        echo  [ERROR] PyTorch installation failed.
        goto :EOF
    )
    echo  [OK] PyTorch CPU-only installed.
) else (
    echo  [OK] PyTorch with CUDA installed.
)
:TORCH_OK

:: ─── Step 4: Dependencies ──────────────────────────────────────────
echo.
echo  [Step 4/6] Installing dependencies ...
echo  ----------------------------------------------------------------
pip install -r requirements.txt --progress-bar raw --timeout 120
if errorlevel 1 (
    echo  [ERROR] Dependency installation failed.
    goto :EOF
)
echo  [OK] Dependencies installed.
:DEPS_OK

:: ─── Step 5: SAM3 ──────────────────────────────────────────────────
echo.
echo  [Step 5/6] Installing SAM3 from GitHub ...
echo  ----------------------------------------------------------------
python -c "import sam3" 2>nul
if not errorlevel 1 (
    echo  [OK] SAM3 already installed - skipping.
    goto :SAM3_INST_OK
)
echo  [INFO] Requires Git: https://git-scm.com/downloads
pip install git+https://github.com/facebookresearch/sam3.git --progress-bar raw
if errorlevel 1 (
    echo  [ERROR] SAM3 installation failed. Make sure Git is installed.
    goto :EOF
)
echo  [OK] SAM3 installed.
:SAM3_INST_OK

:: ─── Step 6: SAM3 Checkpoint ───────────────────────────────────────
if exist "checkpoints\sam3\model.safetensors" goto :SAM3_OK
if exist "checkpoints\sam3\config.json"       goto :SAM3_OK

echo.
echo  [Step 6/6] Downloading SAM3 model checkpoint (~5 GB) ...
echo  ----------------------------------------------------------------
echo.
echo  ================================================================
echo   HUGGINGFACE ACCESS REQUIRED
echo  ================================================================
echo.
echo   1. Create a free account: https://huggingface.co
echo   2. Request model access:  https://huggingface.co/facebook/sam3
echo      Click "Agree and access repository"
echo   3. Create an access token: https://huggingface.co/settings/tokens
echo      Under "Repositories" enable ALL 3 checkboxes:
echo        [x] Read access to contents of all repos under your namespace
echo        [x] View access requests for all gated repos under your namespace
echo        [x] Read access to contents of all public gated repos you can access
echo      Then click "Create token" and copy it.
echo   4. Paste the token below and press ENTER.
echo.
echo  ----------------------------------------------------------------
echo   Your token is stored locally only at:
echo   %USERPROFILE%\.cache\huggingface\token
echo  ----------------------------------------------------------------
echo.
set /p HF_TOKEN="  HuggingFace Token: "
echo.

if "!HF_TOKEN!"=="" (
    echo  [ERROR] No token entered.
    goto :EOF
)

echo  [INFO] Logging in to HuggingFace ...
cmd /c "venv\Scripts\hf.exe auth login --token !HF_TOKEN!"
if errorlevel 1 (
    echo  [ERROR] HuggingFace login failed. Check your token.
    goto :EOF
)
echo  [OK] Login successful.
echo.

mkdir checkpoints\sam3 2>nul
echo  [DOWNLOAD] Downloading SAM3 checkpoint - do NOT close this window!
cmd /c "venv\Scripts\hf.exe download facebook/sam3 --local-dir checkpoints\sam3 --token !HF_TOKEN!"
if errorlevel 1 (
    echo  [ERROR] SAM3 download failed.
    goto :EOF
)
echo  [OK] SAM3 checkpoint downloaded.

:SAM3_OK
echo  [OK] SAM3 checkpoint present.

echo done > .setup_complete
echo  [OK] Setup complete. Future starts skip installation.
echo.

:LAUNCH
echo.
echo  ================================================================
echo    Launching Z+F FlexScanMask ...
echo  ================================================================
echo.

python flexscanmask.py

if errorlevel 1 (
    echo.
    echo  ================================================================
    echo  [ERROR] FlexScanMask exited with an error.
    echo  ================================================================
    echo.
    echo  Type "exit" to close.
    goto :EOF
)

echo.
echo  [OK] FlexScanMask closed normally.
echo.
endlocal
exit
