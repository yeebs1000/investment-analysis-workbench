@echo off
REM ============================================================
REM  Investment Analysis Workbench - one-click launcher
REM  Starts the broker gateways + backend + frontend, then opens
REM  the dashboard. Edit the two paths below if your installs
REM  live elsewhere (leave a path empty to skip that gateway).
REM ============================================================
set "OPEND_EXE=%APPDATA%\moomoo_OpenD\moomoo_OpenD.exe"
set "IBGATEWAY_EXE=C:\Jts\ibgateway\1048\ibgateway.exe"

REM --- gateways (you still log in to each once; they stay open) ---
if exist "%OPEND_EXE%" (
    start "" "%OPEND_EXE%"
) else (
    echo [skip] moomoo OpenD not found at %OPEND_EXE%
)
if exist "%IBGATEWAY_EXE%" (
    start "" "%IBGATEWAY_EXE%"
) else (
    echo [skip] IB Gateway not found at %IBGATEWAY_EXE%
)

REM --- backend (FastAPI on :8010 -- deliberately not the common 8000 default,
REM     so this can run alongside other local FastAPI projects without a
REM     port clash) ---
start "Workbench backend" cmd /k "cd /d "%~dp0backend" && .venv\Scripts\python.exe -m uvicorn app.main:app --port 8010"

REM --- frontend (Vite on :5173) ---
start "Workbench frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

REM --- open the dashboard once the servers have had a moment ---
timeout /t 8 /nobreak >nul
start http://localhost:5173
echo Dashboard opening at http://localhost:5173 - log in to OpenD / IB Gateway if prompted.
