@echo off
REM Backend startup script for Windows
REM This script starts the FastAPI backend server with uvicorn

REM Check if virtual environment exists
if not exist "venv" (
    echo Error: Virtual environment not found at .\venv
    echo Please run setup first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    exit /b 1
)

REM Check if checkpoint exists
set CHECKPOINT_PATH=..\checkpoints\all_features\checkpoint_epoch_25.pth
if not exist "%CHECKPOINT_PATH%" (
    echo Error: Model checkpoint not found at %CHECKPOINT_PATH%
    echo Please ensure the checkpoint file is present before starting the server.
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Set default ALLOWED_ORIGIN if not set
if not defined ALLOWED_ORIGIN (
    set ALLOWED_ORIGIN=http://localhost:5173
    echo Using default ALLOWED_ORIGIN: %ALLOWED_ORIGIN%
)

REM Start server
echo Starting FastAPI backend server...
echo API will be available at: http://localhost:8000
echo API docs available at: http://localhost:8000/docs
echo.
uvicorn main:app --host 0.0.0.0 --port 8000
