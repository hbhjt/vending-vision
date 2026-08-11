param(
    [Parameter(Mandatory = $true)]
    [string]$Wheelhouse,
    [Parameter(Mandatory = $true)]
    [string]$AiWheelhouse,
    [Parameter(Mandatory = $true)]
    [string]$AiWheelhouseDescriptor,
    [string]$SourceRoot
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Invoke-Checked {
    & $args[0] $args[1..($args.Count - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($args -join ' ')"
    }
}

$ToolRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Root = if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $ToolRoot
} else {
    Resolve-Path -LiteralPath $SourceRoot
}
$BasePython = (Get-Command python -ErrorAction Stop).Source
$CoreVenv = Join-Path $Root ".venv-packaging-core"
$AiVenv = Join-Path $Root ".venv-packaging-ai"
$CorePython = Join-Path $CoreVenv "Scripts\python.exe"
$AiPython = Join-Path $AiVenv "Scripts\python.exe"
$BuildDir = Join-Path $Root "build"
$CoreWork = Join-Path $BuildDir "pyinstaller-core"
$AiWork = Join-Path $BuildDir "pyinstaller-ai"
$CoreDist = Join-Path $BuildDir "dist-core"
$AiDist = Join-Path $BuildDir "dist-ai"
$FinalDist = Join-Path $Root "dist"
$AiReleaseRequirements = Join-Path $BuildDir "requirements-ai-release.txt"
$AiBuildRequirements = Join-Path $BuildDir "requirements-ai-build-tools.txt"
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

$ActualPythonVersion = (& $BasePython -c "import platform; print(platform.python_version())").Trim()
if ($ActualPythonVersion -cne $ExpectedPythonVersion) {
    throw "Python $ExpectedPythonVersion is required, found $ActualPythonVersion"
}

New-Item -ItemType Directory -Force $BuildDir | Out-Null
Invoke-Checked $BasePython (Join-Path $ToolRoot "scripts\materialize_ai_wheelhouse.py") --descriptor $AiWheelhouseDescriptor --wheelhouse $AiWheelhouse --runtime-descriptor (Join-Path $Root "ai-runtime-descriptor.json") --requirements-output $AiReleaseRequirements
Invoke-Checked $BasePython (Join-Path $ToolRoot "scripts\render_ai_build_requirements.py") --core-requirements (Join-Path $Root "requirements.txt") --output $AiBuildRequirements

foreach ($Environment in @($CoreVenv, $AiVenv)) {
    if (Test-Path -LiteralPath $Environment) {
        Remove-Item -LiteralPath $Environment -Recurse -Force
    }
}
foreach ($Output in @($CoreWork, $AiWork, $CoreDist, $AiDist, $FinalDist)) {
    if (Test-Path -LiteralPath $Output) {
        Remove-Item -LiteralPath $Output -Recurse -Force
    }
}

Invoke-Checked $BasePython (Join-Path $ToolRoot "scripts\bootstrap_build_envs.py") `
    --base-python $BasePython `
    --core-env $CoreVenv `
    --core-wheelhouse $Wheelhouse `
    --core-requirements (Join-Path $Root "requirements.txt") `
    --ai-env $AiVenv `
    --ai-wheelhouse $AiWheelhouse `
    --ai-requirements $AiReleaseRequirements

if (-not (Test-Path -LiteralPath $CorePython -PathType Leaf)) { throw "core build Python missing: $CorePython" }
if (-not (Test-Path -LiteralPath $AiPython -PathType Leaf)) { throw "AI build Python missing: $AiPython" }

Invoke-Checked $CorePython (Join-Path $ToolRoot "scripts\dependency_lock.py") --requirements-lock (Join-Path $Root "requirements.txt") --wheelhouse $Wheelhouse --python $CorePython --target-sys-platform win32
Invoke-Checked $CorePython -c "from vision.model_manifest import verify_model_manifest; result=verify_model_manifest(); assert result['ok'], result"
Invoke-Checked $CorePython -m PyInstaller --clean --noconfirm --workpath $CoreWork --distpath $CoreDist (Join-Path $Root "vending_vision.spec")

Invoke-Checked $AiPython -m pip install --disable-pip-version-check --no-index --find-links $Wheelhouse --require-hashes --no-deps -r $AiBuildRequirements
Invoke-Checked $AiPython (Join-Path $ToolRoot "scripts\verify_ai_wheelhouse.py") --descriptor $AiWheelhouseDescriptor --wheelhouse $AiWheelhouse --runtime-descriptor (Join-Path $Root "ai-runtime-descriptor.json") --requirements-output $AiReleaseRequirements
Invoke-Checked $AiPython (Join-Path $Root "run_ai_attempt_worker.py") "--probe-runtime"
Invoke-Checked $AiPython -m PyInstaller --clean --noconfirm --workpath $AiWork --distpath $AiDist (Join-Path $Root "vending_vision_ai_worker.spec")

New-Item -ItemType Directory -Force $FinalDist | Out-Null
Copy-Item -LiteralPath (Join-Path $CoreDist "vending-vision") -Destination $FinalDist -Recurse
Copy-Item -LiteralPath (Join-Path $AiDist "vending-vision-ai-worker") -Destination $FinalDist -Recurse

$MainExe = Join-Path $FinalDist "vending-vision\vending-vision.exe"
$WorkerExe = Join-Path $FinalDist "vending-vision-ai-worker\vending-vision-ai-worker.exe"
if (-not (Test-Path -LiteralPath $MainExe -PathType Leaf)) { throw "main Vision exe missing after build: $MainExe" }
if (-not (Test-Path -LiteralPath $WorkerExe -PathType Leaf)) { throw "AI worker exe missing after build: $WorkerExe" }

Invoke-Checked $CorePython (Join-Path $ToolRoot "scripts\verify_packaged_exe.py") $MainExe

Write-Host ""
Write-Host "Build complete:"
Write-Host $MainExe
Write-Host $WorkerExe
