@echo off
REM ============================================================
REM  Auto-start launcher for the Audio Capturer web UI.
REM  Runs the Flask server (app.py) in the background, no admin.
REM  Invoked hidden at login by the .vbs in the Startup folder,
REM  but you can also double-click this file to start the server.
REM ============================================================

REM cd into this script's own folder (the project root).
cd /d "%~dp0"

set "LOGDIR=%LOCALAPPDATA%\audio_capturer"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "LOG=%LOGDIR%\startup.log"

echo.>> "%LOG%"
echo ==== %date% %time% : launching Audio Capturer web UI (port 5000) ====>> "%LOG%"

REM Single-instance guard: if something is already listening on :5000
REM (e.g. a server you started manually), skip launching another one.
netstat -an -p tcp | findstr /c:":5000 " | findstr /c:"LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo ==== %date% %time% : port 5000 already in use - skipping launch ====>> "%LOG%"
    goto :eof
)

REM pythonw.exe = no console window. Redirection still captures Flask logs.
".venv\Scripts\pythonw.exe" app.py --no-reload --port 5000 >> "%LOG%" 2>&1

echo ==== %date% %time% : server process exited (code %errorlevel%) ====>> "%LOG%"
