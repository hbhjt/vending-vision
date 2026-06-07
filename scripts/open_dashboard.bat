@echo off
chcp 65001 > nul

echo ================================
echo Opening Vending Vision Dashboard
echo URL: http://127.0.0.1:7892/dashboard
echo ================================

start http://127.0.0.1:7892/dashboard
