param(
    [int]$IntervalSeconds = 30,
    [int]$DurationMinutes = 480,
    [string]$OutputDir = "test_reports\memory"
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputPath = Join-Path $OutputDir "memory_$timestamp.csv"

"time,pid,process_name,working_set_mb,private_memory_mb,cpu_seconds,system_available_mb" | Out-File -Encoding utf8 $outputPath

$endAt = (Get-Date).AddMinutes($DurationMinutes)

Write-Host "Monitoring Vending Vision memory usage"
Write-Host "Interval seconds: $IntervalSeconds"
Write-Host "Duration minutes: $DurationMinutes"
Write-Host "Output: $outputPath"
Write-Host ""

while ((Get-Date) -lt $endAt) {
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $os = Get-CimInstance Win32_OperatingSystem
    $availableMb = [math]::Round($os.FreePhysicalMemory / 1024, 2)

    $candidates = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match "^(python|pythonw)\.exe$" -and
            $_.CommandLine -and
            ($_.CommandLine -like "*uvicorn*" -or $_.CommandLine -like "*app:app*")
        }

    if (-not $candidates) {
        Write-Host "$now no uvicorn python process found"
    }

    foreach ($candidate in $candidates) {
        $proc = Get-Process -Id $candidate.ProcessId -ErrorAction SilentlyContinue
        if (-not $proc) {
            continue
        }

        $workingSetMb = [math]::Round($proc.WorkingSet64 / 1MB, 2)
        $privateMb = [math]::Round($proc.PrivateMemorySize64 / 1MB, 2)
        $cpuSeconds = [math]::Round($proc.CPU, 2)

        "$now,$($proc.Id),$($proc.ProcessName),$workingSetMb,$privateMb,$cpuSeconds,$availableMb" |
            Out-File -Append -Encoding utf8 $outputPath

        Write-Host "$now pid=$($proc.Id) working_set=${workingSetMb}MB private=${privateMb}MB available=${availableMb}MB"
    }

    Start-Sleep -Seconds $IntervalSeconds
}

Write-Host ""
Write-Host "Done. CSV saved to $outputPath"
