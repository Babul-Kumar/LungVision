@echo off
setlocal enabledelayedexpansion
title LungVision AI Launcher

echo ===========================================================================
echo                       LUNGVISION AI WORKSTATION
echo            Thoracic Pathology Intelligence ^& 3D CT Exploration
echo ===========================================================================
echo.

:: Ensure working directory is always project root
cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"

:: 1. Detect Python Runtime (Prioritize Python 3.11 with verified torch/SimpleITK)
set "PYTHON_CMD="
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    echo [INFO] Detected verified Python 3.11 runtime: !PYTHON_CMD!
) else (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=python"
        echo [INFO] Using system Python runtime.
    ) else (
        echo [ERROR] Python not found in PATH or standard installation directory.
        echo Please install Python 3.11+ or add it to your system PATH.
        pause
        exit /b 1
    )
)

:: 2. Check Node / npm runtime for frontend
where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Node.js / npm not found in system PATH.
    echo Please install Node.js 18+ from https://nodejs.org/ to run the frontend.
    pause
    exit /b 1
)

:: 3. Check and install frontend node_modules if missing
if not exist "%PROJECT_ROOT%\frontend\node_modules" (
    echo [INFO] First-time setup: installing frontend node_modules...
    pushd "%PROJECT_ROOT%\frontend"
    call npm install
    popd
)

echo.
echo [1/3] Launching FastAPI Backend Gateway on http://127.0.0.1:8000 ...
start "LungVision AI - Backend Gateway (Port 8000)" cmd /k "cd /d "%PROJECT_ROOT%" && "!PYTHON_CMD!" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/3] Launching Vite React Frontend Workstation on http://localhost:5173 ...
start "LungVision AI - Frontend UI (Port 5173)" cmd /k "cd /d "%PROJECT_ROOT%\frontend" && npm run dev"

echo [3/3] Opening LungVision AI in your default browser...
timeout /t 3 /nobreak >nul
start http://localhost:5173

echo.
echo ===========================================================================
echo                 LUNGVISION AI SERVICES ARE NOW RUNNING
echo ===========================================================================
echo  * Frontend Workstation: http://localhost:5173
echo  * Backend REST API:     http://127.0.0.1:8000
echo  * Interactive Docs:     http://127.0.0.1:8000/docs
echo  * Health Probe:         http://127.0.0.1:8000/api/v1/health
echo.
echo  Keep the opened backend and frontend console windows running.
echo  To stop the application, close both console windows.
echo ===========================================================================
echo.
pause

