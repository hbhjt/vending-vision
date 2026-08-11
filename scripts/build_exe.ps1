param(
    [Parameter(Mandatory = $true)]
    [string]$Wheelhouse,
    [Parameter(Mandatory = $true)]
    [string]$AiWheelhouse,
    [Parameter(Mandatory = $true)]
    [string]$AiWheelhouseDescriptor
)

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
$BuildDir = Join-Path $Root "build"
$AiReleaseRequirements = Join-Path $BuildDir "requirements-ai-release.txt"
$ExpectedPythonVersion = (Get-Content (Join-Path $Root ".python-version") -Raw).Trim()
if (-not (Test-Path -LiteralPath $Wheelhouse -PathType Container)) {
    throw "A pre-validated offline core wheelhouse is required: $Wheelhouse"
}
if (-not (Test-Path -LiteralPath $AiWheelhouse -PathType Container)) {
    throw "A verified exact offline AI wheelhouse is required: $AiWheelhouse"
}
if (-not (Test-Path -LiteralPath $AiWheelhouseDescriptor -PathType Leaf)) {
    throw "AI wheelhouse release descriptor is required: $AiWheelhouseDescriptor"
}

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

Invoke-Checked $Python -m pip install --no-index --find-links $Wheelhouse --require-hashes -r (Join-Path $Root "requirements.txt")
New-Item -ItemType Directory -Force $BuildDir | Out-Null
Invoke-Checked $Python (Join-Path $Root "scripts\verify_ai_wheelhouse.py") --descriptor $AiWheelhouseDescriptor --wheelhouse $AiWheelhouse --runtime-descriptor (Join-Path $Root "ai-runtime-descriptor.json") --requirements-output $AiReleaseRequirements
Invoke-Checked $Python -m pip install --no-index --find-links $AiWheelhouse --require-hashes -r $AiReleaseRequirements

Invoke-Checked $Python -m PyInstaller --clean --noconfirm (Join-Path $Root "vending_vision.spec")
Invoke-Checked $Python -m PyInstaller --clean --noconfirm (Join-Path $Root "vending_vision_ai_worker.spec")

$MainExe = Join-Path $Root "dist\vending-vision\vending-vision.exe"
$WorkerExe = Join-Path $Root "dist\vending-vision-ai-worker\vending-vision-ai-worker.exe"
if (-not (Test-Path -LiteralPath $MainExe -PathType Leaf)) { throw "main Vision exe missing after build: $MainExe" }
if (-not (Test-Path -LiteralPath $WorkerExe -PathType Leaf)) { throw "AI worker exe missing after build: $WorkerExe" }

Invoke-Checked $Python (Join-Path $Root "scripts\verify_packaged_exe.py") $MainExe --require-ai-worker

Write-Host ""
Write-Host "Build complete:"
Write-Host $MainExe
Write-Host $WorkerExe
