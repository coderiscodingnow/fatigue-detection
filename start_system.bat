@echo off
REM Start System - Launches main.py and dashboard.py in parallel
REM
REM This script starts the Driver Monitoring System with both backend (main.py)
REM and frontend (dashboard) running simultaneously.

setlocal enabledelayedexpansion

echo.
echo ================================================================
echo DRIVER MONITORING SYSTEM - STARTUP
echo ================================================================
echo.

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then: .venv\Scripts\activate.bat
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo [1/2] Starting data collection and model pipeline (main.py)...
echo       This process collects sensor data, runs vision model, and calculates fatigue metrics...

REM Start main.py in a separate window
start "Main Pipeline" cmd /k python main.py

REM Wait for main.py to initialize
echo       Status: RUNNING
timeout /t 3 /nobreak

echo.
echo [2/2] Starting dashboard interface (Streamlit)...
echo       Dashboard will open in your default browser...
echo.
echo ================================================================
echo System is now running!
echo ================================================================
echo.
echo To stop the system, close the dashboard window.
echo.

REM Start dashboard
python -m streamlit run dashboard/dashboard.py --logger.level=warning --client.showErrorDetails=false

echo.
echo System stopped.
pause
