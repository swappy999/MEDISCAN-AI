@echo off
title MediScan AI - Master Startup Launcher
color 0b
echo ======================================================================
echo                     MEDISCAN AI - SYSTEM LAUNCHER
echo ======================================================================
echo.
echo  Starting all four MediScan AI services in dedicated windows:
echo.
echo    [1/4] Chest X-Ray FastAPI ML Service (Port 8000 - PyTorch CUDA)
echo    [2/4] Brain MRI FastAPI ML Service   (Port 8001 - PyTorch CUDA)
echo    [3/4] Node.js Express Backend API    (Port 5000 - SQLite ^& PDF Reporting)
echo    [4/4] React Vite Frontend UI         (Port 5173 - Clinical Workstation)
echo.
echo ======================================================================

set "ROOT_DIR=%~dp0"

:: 1. Launch Chest X-Ray Python FastAPI ML Service
echo [1/4] Launching Chest X-Ray FastAPI ML Service...
start "MediScan AI - Chest X-Ray Service (FastAPI :8000)" cmd /k "cd /d "%ROOT_DIR%DL\ml-services\chest-xray" && title MediScan AI - Chest X-Ray Service :8000 && color 0a && echo =================================================== && echo   MediScan AI - Chest X-Ray Service (FastAPI :8000) && echo =================================================== && "%ROOT_DIR%DL\chest-xray\venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000"

:: Brief delay to allow Chest X-Ray ML service to initialize
timeout /t 2 /nobreak >nul

:: 2. Launch Brain MRI Python FastAPI ML Service
echo [2/4] Launching Brain MRI FastAPI ML Service...
start "MediScan AI - Brain MRI Service (FastAPI :8001)" cmd /k "cd /d "%ROOT_DIR%DL\ml-services\brain-mri" && title MediScan AI - Brain MRI Service :8001 && color 0d && echo =================================================== && echo   MediScan AI - Brain MRI Service (FastAPI :8001) && echo =================================================== && "%ROOT_DIR%DL\brain-mri-tumor\venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8001"

:: Brief delay to allow Brain MRI ML service to initialize
timeout /t 2 /nobreak >nul

:: 3. Launch Node.js Express Backend
echo [3/4] Launching Express Backend API...
start "MediScan AI - Backend API (Express :5000)" cmd /k "cd /d "%ROOT_DIR%backend" && title MediScan AI - Backend API :5000 && color 0b && echo =================================================== && echo   MediScan AI - Backend API (Express on Port 5000) && echo =================================================== && npm run dev"

:: Brief delay
timeout /t 2 /nobreak >nul

:: 4. Launch React Frontend
echo [4/4] Launching React Desktop Frontend...
start "MediScan AI - Frontend (Vite :5173)" cmd /k "cd /d "%ROOT_DIR%frontend" && title MediScan AI - Frontend :5173 && color 0e && echo =================================================== && echo   MediScan AI - Frontend UI (Vite on Port 5173) && echo =================================================== && npm run dev"

echo.
echo ======================================================================
echo  All services have been started!
echo.
echo  Chest X-Ray API : http://127.0.0.1:8000/health
echo  Brain MRI API   : http://127.0.0.1:8001/health
echo  Express API     : http://localhost:5000/api/health
echo  Web Interface   : http://localhost:5173
echo.
echo  Opening http://localhost:5173 in your default browser...
echo ======================================================================
echo.

timeout /t 3 /nobreak >nul
start http://localhost:5173

echo  Done! To stop all services later, run stop_all.bat or close their windows.
timeout /t 5 >nul

