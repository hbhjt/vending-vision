param(
    [Parameter(Mandatory = $true)]
    [string]$Wheelhouse,
    [string]$SourceRoot,
    [string]$SourceCommit
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
$CorePython = Join-Path $CoreVenv "Scripts\python.exe"
$BuildDir = Join-Path $Root "build"
$CoreWork = Join-Path $BuildDir "pyinstaller-core"
$CoreDist = Join-Path $BuildDir "dist-core"
$FinalDist = Join-Path $Root "dist"
$ExpectedPythonVersion = (Get-Content (Join-Path $Root ".python-version") -Raw).Trim()
$RepositoryRoot = (Invoke-Checked git -C $Root rev-parse --show-toplevel).Trim()
$ActualSourceCommit = (Invoke-Checked git -C $Root rev-parse HEAD).Trim()
$TrackedStatus = (Invoke-Checked git -C $Root status --porcelain --untracked-files=no)
if ([IO.Path]::GetFullPath($RepositoryRoot) -cne [IO.Path]::GetFullPath($Root)) {
    throw "Build source must be the Git repository root"
}
if (-not [string]::IsNullOrWhiteSpace($TrackedStatus)) {
    throw "Build source has tracked changes"
}
if ([string]::IsNullOrWhiteSpace($SourceCommit)) {
    $SourceCommit = $ActualSourceCommit
}
if ($SourceCommit -cne $ActualSourceCommit -or $SourceCommit -notmatch '^[0-9a-f]{40}$') {
    throw "Build source commit does not match clean Git HEAD"
}

if (-not (Test-Path -LiteralPath $Wheelhouse -PathType Container)) {
    throw "A pre-validated offline core wheelhouse is required: $Wheelhouse"
}

$ActualPythonVersion = (Invoke-Checked $BasePython -c "import platform; print(platform.python_version())").Trim()
if ($ActualPythonVersion -cne $ExpectedPythonVersion) {
    throw "Python $ExpectedPythonVersion is required, found $ActualPythonVersion"
}

New-Item -ItemType Directory -Force $BuildDir | Out-Null

foreach ($Environment in @($CoreVenv)) {
    if (Test-Path -LiteralPath $Environment) {
        Remove-Item -LiteralPath $Environment -Recurse -Force
    }
}
foreach ($Output in @($CoreWork, $CoreDist, $FinalDist)) {
    if (Test-Path -LiteralPath $Output) {
        Remove-Item -LiteralPath $Output -Recurse -Force
    }
}

Invoke-Checked $BasePython (Join-Path $ToolRoot "scripts\bootstrap_build_envs.py") `
    --base-python $BasePython `
    --core-env $CoreVenv `
    --core-wheelhouse $Wheelhouse `
    --core-requirements (Join-Path $Root "requirements.txt")

if (-not (Test-Path -LiteralPath $CorePython -PathType Leaf)) { throw "core build Python missing: $CorePython" }

Invoke-Checked $CorePython (Join-Path $ToolRoot "scripts\dependency_lock.py") --requirements-lock (Join-Path $Root "requirements.txt") --wheelhouse $Wheelhouse --python $CorePython --target-sys-platform win32
Invoke-Checked $CorePython -c "from vision.model_manifest import verify_model_manifest; result=verify_model_manifest(); assert result['ok'], result"
Invoke-Checked $CorePython -m PyInstaller --clean --noconfirm --workpath $CoreWork --distpath $CoreDist (Join-Path $Root "vending_vision.spec")

New-Item -ItemType Directory -Force $FinalDist | Out-Null
Copy-Item -LiteralPath (Join-Path $CoreDist "vending-vision") -Destination $FinalDist -Recurse

$SourceBuildMarker = Join-Path $Root "vision\_build_version.py"
$PackagedBuildIdentity = Join-Path $FinalDist "vending-vision\_internal\vision\_build_identity.json"
Invoke-Checked $CorePython (Join-Path $ToolRoot "scripts\write_packaged_build_identity.py") `
    --version-marker $SourceBuildMarker `
    --identity-output $PackagedBuildIdentity `
    --source-commit $SourceCommit

$MainExe = Join-Path $FinalDist "vending-vision\vending-vision.exe"
if (-not (Test-Path -LiteralPath $MainExe -PathType Leaf)) { throw "main Vision exe missing after build: $MainExe" }

Invoke-Checked $CorePython (Join-Path $ToolRoot "scripts\verify_packaged_exe.py") $MainExe

Write-Host ""
Write-Host "Build complete:"
Write-Host $MainExe
