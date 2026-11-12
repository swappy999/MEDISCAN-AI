@echo off
title MediScan AI - Stop Services
color 0c
echo ======================================================================
echo                     MEDISCAN AI - STOP ALL SERVICES
echo ======================================================================
echo.
echo  Stopping services running on ports:
echo    - Port 8000 (Chest X-Ray FastAPI Service)
echo    - Port 8001 (Brain MRI FastAPI Service)
echo    - Port 5000 (Express Backend API)
echo    - Port 5173 (React Vite Frontend)
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ports = @(8000, 8001, 5000, 5173); foreach ($port in $ports) { $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue; if ($connections) { $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique; foreach ($pidToKill in $pids) { try { Stop-Process -Id $pidToKill -Force -ErrorAction SilentlyContinue; Write-Host ('  [OK] Stopped PID ' + $pidToKill + ' on port ' + $port) -ForegroundColor Green } catch {} } } else { Write-Host ('  [--] Port ' + $port + ' was not in use.') -ForegroundColor Gray } }"

echo.
echo ======================================================================
echo  All MediScan AI services have been stopped.
echo ======================================================================
echo.
timeout /t 3 >nul

