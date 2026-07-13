param(
    [string]$SourceRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$InstallRoot = 'C:\VEM\vision',
    [string]$TaskName = 'StartVisionServer',
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$SourceRoot = (Resolve-Path $SourceRoot).Path
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null

# Copy only the immutable release payload. Site config and logs remain outside
# the source tree so a later ISO/runtime update cannot overwrite calibration.
robocopy $SourceRoot $InstallRoot /E /XD .git __pycache__ logs reports /XF '*.pyc' | Out-Null
if ($LASTEXITCODE -gt 7) { throw "robocopy failed with exit code $LASTEXITCODE" }

& $Python -m pip install --upgrade --requirement (Join-Path $InstallRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'vision dependency installation failed' }

$visionOrigins = 'http://127.0.0.1:7892,http://localhost:7892,http://tauri.localhost'
$command = "set VISION_MOCK_SCENARIO=off&& set VISION_HOST=127.0.0.1&& set VISION_ALLOWED_ORIGINS=$visionOrigins&& `"$Python`" -m uvicorn app:app --host 127.0.0.1 --port 7892"
$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/d /c `"$command`"" -WorkingDirectory $InstallRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 3650) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "Installed VEM vision at $InstallRoot and started task $TaskName"
