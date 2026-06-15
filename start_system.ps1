# Start System - Launches main.py and dashboard.py in parallel
# 
# This script starts the Driver Monitoring System with both backend (main.py)
# and frontend (dashboard) running simultaneously.

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "DRIVER MONITORING SYSTEM - STARTUP" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
& ".\.venv\Scripts\Activate.ps1"

Write-Host "[1/2] Starting data collection and model pipeline (main.py)..." -ForegroundColor Green
Write-Host "      This process collects sensor data, runs vision model, and calculates fatigue metrics..." -ForegroundColor Gray

# Start main.py in a separate process
$mainProcess = Start-Process -FilePath "python" -ArgumentList "main.py" -PassThru -WindowStyle Normal

Write-Host "      Status: RUNNING (PID: $($mainProcess.Id))" -ForegroundColor Green
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "[2/2] Starting dashboard interface (Streamlit)..." -ForegroundColor Green
Write-Host "      Dashboard will open in your default browser..." -ForegroundColor Gray
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "System is now running!" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Start dashboard (this will run in foreground)
try {
    & python -m streamlit run dashboard/dashboard.py --logger.level=warning --client.showErrorDetails=false
}
finally {
    Write-Host ""
    Write-Host "Shutting down system..." -ForegroundColor Yellow
    
    # Terminate main.py
    if ($mainProcess -and -not $mainProcess.HasExited) {
        $mainProcess | Stop-Process -Force -ErrorAction SilentlyContinue
        Write-Host "Data collection process terminated." -ForegroundColor Yellow
    }
    
    Write-Host "System stopped." -ForegroundColor Green
}
