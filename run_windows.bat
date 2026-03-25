@echo off
REM Hand Tracking - Run Script for Windows

REM Check if environment exists
if not exist ".venv\" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_windows.bat first.
    echo.
    pause
    exit /b 1
)

REM Activate environment and run the script
call .venv\Scripts\activate.bat
python handtracking.py %*

REM Keep window open if there was an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Script exited with an error.
    pause
)
