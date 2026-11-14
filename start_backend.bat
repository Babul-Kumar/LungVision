@echo off
set "PYTHON_CMD=python"
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)
echo ========================================================
echo Starting LungVision AI FastAPI Backend Gateway (Port 8000)
echo Using Python: %PYTHON_CMD%
echo ========================================================
"%PYTHON_CMD%" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
pause
