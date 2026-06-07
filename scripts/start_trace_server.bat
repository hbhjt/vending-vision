@echo off
chcp 65001 > nul

echo ================================
echo Starting Vending Vision Module
echo Mode: Real Camera / Trace
echo URL: ws://127.0.0.1:7892/ws
echo Config: config.json
echo Process trace: on
echo Output: debug_outputs/process_traces
echo ================================

cd /d %~dp0\..

set VISION_MOCK_SCENARIO=off
set VISION_PROCESS_TRACE_ENABLED=true
set VISION_PROCESS_TRACE_OUTPUT_DIR=debug_outputs/process_traces

python -m uvicorn app:app --host 127.0.0.1 --port 7892

pause
