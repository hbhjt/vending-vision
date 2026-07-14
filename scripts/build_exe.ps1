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
$ExpectedPythonVersion = (Get-Content (Join-Path $Root ".python-version") -Raw).Trim()
$PipIndexUrl = if ($env:PIP_INDEX_URL) { $env:PIP_INDEX_URL } else { "https://pypi.org/simple" }
$PipIndexArgs = @("--index-url", $PipIndexUrl)

$ActualPythonVersion = (& python -c "import platform; print(platform.python_version())").Trim()
if ($ActualPythonVersion -cne $ExpectedPythonVersion) {
    throw "Python $ExpectedPythonVersion is required, found $ActualPythonVersion"
}

if (Test-Path $Python) {
    $VenvPythonVersion = (& $Python -c "import platform; print(platform.python_version())").Trim()
    if ($VenvPythonVersion -cne $ExpectedPythonVersion) {
        Remove-Item -LiteralPath $Venv -Recurse -Force
    }
}

if (-not (Test-Path $Python)) {
    Invoke-Checked python -m venv $Venv
}

Invoke-Checked $Python -m pip install @PipIndexArgs -r (Join-Path $Root "requirements-packaging.txt")

Invoke-Checked $Python -m PyInstaller --clean --noconfirm (Join-Path $Root "vending_vision.spec")

Write-Host ""
Write-Host "Build complete:"
Write-Host (Join-Path $Root "dist\vending-vision\vending-vision.exe")
