$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Invoke-Checked {
    & $args[0] $args[1..($args.Count - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($args -join ' ')"
    }
}

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Venv = Join-Path $Root ".venv-packaging"
$Python = Join-Path $Venv "Scripts\python.exe"
$PipIndexUrl = if ($env:PIP_INDEX_URL) { $env:PIP_INDEX_URL } else { "https://pypi.org/simple" }
$PipIndexArgs = @("--index-url", $PipIndexUrl)

if (-not (Test-Path $Python)) {
    Invoke-Checked python -m venv $Venv
}

Invoke-Checked $Python -m pip install @PipIndexArgs --upgrade pip setuptools wheel
Invoke-Checked $Python -m pip install @PipIndexArgs -r (Join-Path $Root "requirements-packaging.txt")
Invoke-Checked $Python -m pip install @PipIndexArgs "pyinstaller==6.16.0"

Invoke-Checked $Python -m PyInstaller --clean --noconfirm (Join-Path $Root "vending_vision.spec")

Write-Host ""
Write-Host "Build complete:"
Write-Host (Join-Path $Root "dist\vending-vision\vending-vision.exe")
