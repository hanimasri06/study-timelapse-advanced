@echo off
echo =========================================
echo Study Timelapse App Initialization
echo =========================================

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [INFO] Creating Python virtual environment...
    python -m venv .venv
    echo [INFO] Virtual environment created.
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

python bootstrap.py
if errorlevel 1 (
    echo [ERROR] Dependency setup failed.
    pause
    exit /b 1
)

echo [INFO] Starting Application...
python main.py

pause
