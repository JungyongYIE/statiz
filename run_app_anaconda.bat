@echo off
setlocal

set "APP_DIR=%~dp0"
set "ANACONDA_PY=C:\Users\User\anaconda3\python.exe"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "PYTHONPATH=%APP_DIR%;%PYTHONPATH%"

cd /d "%APP_DIR%"

if not exist "%ANACONDA_PY%" (
    echo [ERROR] Anaconda Python not found:
    echo         %ANACONDA_PY%
    pause
    exit /b 1
)

echo [INFO] Project directory: %CD%
echo [INFO] Python: %ANACONDA_PY%
"%ANACONDA_PY%" -c "import sys, streamlit; print('[INFO] Python version:', sys.version.split()[0]); print('[INFO] Streamlit version:', streamlit.__version__)"

if errorlevel 1 (
    echo [ERROR] Failed to import Streamlit from the Anaconda environment.
    pause
    exit /b 1
)

set "APP_PORT="
for %%P in (8501 8502 8503 8504 8505 8506 8507 8508 8509 8510) do (
    netstat -ano -p tcp | findstr /R /C:":%%P .*LISTENING" >nul 2>nul
    if errorlevel 1 (
        set "APP_PORT=%%P"
        goto :PORT_FOUND
    )
)

:PORT_FOUND

if "%APP_PORT%"=="" (
    echo [ERROR] No available local port found in 8501-8510.
    echo         Close an existing Streamlit window or stop the process using that port.
    pause
    exit /b 1
)

if not "%APP_PORT%"=="8501" (
    echo [WARN] Port 8501 is already in use. Using %APP_PORT% instead.
)

echo [INFO] Starting Streamlit at http://127.0.0.1:%APP_PORT%
"%ANACONDA_PY%" -m streamlit run "%APP_DIR%app.py" --server.address 127.0.0.1 --server.port %APP_PORT% --server.headless false --browser.gatherUsageStats false

pause
