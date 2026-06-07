@echo off
chcp 65001 > nul

echo ================================
echo Stopping Vending Vision Module
echo Port: 7892
echo ================================

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$connections = Get-NetTCPConnection -LocalPort 7892 -State Listen -ErrorAction SilentlyContinue; " ^
  "if (-not $connections) { Write-Host 'No vision server is listening on port 7892.'; exit 0 }; " ^
  "$stopped = @(); " ^
  "foreach ($connection in $connections) { " ^
  "  $pidValue = $connection.OwningProcess; " ^
  "  $proc = Get-CimInstance Win32_Process -Filter \"ProcessId=$pidValue\"; " ^
  "  if ($proc.CommandLine -like '*uvicorn app:app*' -and $proc.CommandLine -like '*7892*') { " ^
  "    Stop-Process -Id $pidValue -Force; " ^
  "    $stopped += $pidValue; " ^
  "  } " ^
  "}; " ^
  "if ($stopped.Count -eq 0) { Write-Host 'Port 7892 is used, but not by uvicorn app:app. Nothing stopped.' } else { Write-Host ('Stopped PID(s): ' + ($stopped -join ', ')) }"

pause
