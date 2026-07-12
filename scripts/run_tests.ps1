$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv-packaging\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python -m unittest discover -s (Join-Path $Root "tests")
