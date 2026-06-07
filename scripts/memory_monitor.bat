@echo off
chcp 65001 > nul

cd /d %~dp0\..

set INTERVAL_SECONDS=%1
set DURATION_MINUTES=%2

if "%INTERVAL_SECONDS%"=="" set INTERVAL_SECONDS=30
if "%DURATION_MINUTES%"=="" set DURATION_MINUTES=480

powershell -ExecutionPolicy Bypass -File scripts\memory_monitor.ps1 -IntervalSeconds %INTERVAL_SECONDS% -DurationMinutes %DURATION_MINUTES%

pause
