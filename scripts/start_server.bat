@echo off
chcp 65001 > nul

echo ================================
echo Starting Vending Vision Module
echo Mode: Real Camera / Normal
echo URL: ws://127.0.0.1:7892/ws
echo Config: config.json
echo ================================

cd /d %~dp0\..

set VISION_MOCK_SCENARIO=off

python -m uvicorn app:app --host 127.0.0.1 --port 7892

pause
