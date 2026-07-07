@echo off
REM ============================================================
REM  Investment Analysis Workbench - first-time setup
REM  Automates the SETUP.md flow: checks Python + Node, creates
REM  the backend venv, installs all dependencies, creates .env.
REM  Run once; afterwards use start.bat.
REM ============================================================

where python >nul 2>nul
if errorlevel 1 (
    echo [error] Python not found. Install Python 3.11 from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH" in the installer, then re-run setup.bat.
    pause & exit /b 1
)
where node >nul 2>nul
if errorlevel 1 (
    echo [error] Node.js not found. Install the LTS version from https://nodejs.org/
    echo         then re-run setup.bat.
    pause & exit /b 1
)

echo [1/4] Creating Python environment...
cd /d "%~dp0backend"
if not exist .venv python -m venv .venv
if errorlevel 1 (echo [error] venv creation failed & pause & exit /b 1)

echo [2/4] Installing backend dependencies (a few minutes the first time)...
.venv\Scripts\python.exe -m pip install --upgrade pip -q
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (echo [error] pip install failed - check the output above & pause & exit /b 1)

echo [3/4] Creating backend\.env (your keys/settings live here)...
if not exist .env copy .env.example .env >nul

echo [4/4] Installing frontend dependencies...
cd /d "%~dp0frontend"
call npm install
if errorlevel 1 (echo [error] npm install failed - check the output above & pause & exit /b 1)

echo.
echo ============================================================
echo  Setup complete. Two things left to do by hand:
echo   1. Install + log in to your broker gateway(s):
echo        Moomoo OpenD:  https://www.moomoo.com/download/OpenAPI
echo        IB Gateway:    https://www.interactivebrokers.com/en/trading/ibgateway-stable.php
echo      (either one is enough - the app uses whichever is linked)
echo   2. Edit backend\.env - broker settings and optional API keys
echo      (Finnhub / FRED / Gemini / Claude; see SETUP.md for details)
echo  Then double-click start.bat to launch everything.
echo ============================================================
pause
