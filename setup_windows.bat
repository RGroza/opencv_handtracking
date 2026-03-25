@echo off
REM Hand Tracking Setup Script for Windows
REM This script will automatically install Python and dependencies

echo ========================================
echo Hand Tracking - Windows Setup
echo ========================================
echo.

REM Check if uv is already installed
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [1/3] Downloading uv package manager...
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install uv. Please check your internet connection.
        pause
        exit /b 1
    )
    
    REM Add uv to PATH for this session
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
) else (
    echo [1/3] uv package manager already installed
)

echo [2/3] Creating Python environment and installing dependencies...
uv venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [3/3] Installing required packages (this may take a minute)...
uv pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo You can now run the hand tracking script by double-clicking:
echo   run_windows.bat
echo.
echo Or from this command prompt:
echo   run_windows.bat
echo.
pause
