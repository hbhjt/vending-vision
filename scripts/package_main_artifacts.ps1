param(
    [Parameter(Mandatory = $true)]
    [string]$Commit,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$RuntimeDirectory = Join-Path $Root "dist\vending-vision"
$FixtureDirectory = Join-Path $Root "fixtures\recorded-video"
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)

if (-not (Test-Path -LiteralPath $RuntimeDirectory -PathType Container)) {
    throw "Windows runtime build is missing: $RuntimeDirectory"
}
if (-not (Test-Path -LiteralPath $FixtureDirectory -PathType Container)) {
    throw "Recorded-video fixtures are missing: $FixtureDirectory"
}

New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$StageDirectory = Join-Path $OutputDirectory ".stage"
Remove-Item -LiteralPath $StageDirectory -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $StageDirectory | Out-Null

$manifest = @{
    schemaVersion = "vending-vision-main-artifacts/v1"
    commit = $Commit
    runtimeArchive = "vending-vision-windows-x86_64.zip"
    fixtureArchive = "vending-vision-test-fixtures.zip"
} | ConvertTo-Json -Depth 4

$RuntimeStage = Join-Path $StageDirectory "runtime"
Copy-Item -LiteralPath $RuntimeDirectory -Destination $RuntimeStage -Recurse
Set-Content -LiteralPath (Join-Path $RuntimeStage "vision-artifact.json") -Value $manifest -Encoding utf8NoBOM

$FixtureStage = Join-Path $StageDirectory "fixtures"
New-Item -ItemType Directory -Force $FixtureStage | Out-Null
Copy-Item -LiteralPath $FixtureDirectory -Destination (Join-Path $FixtureStage "recorded-video") -Recurse
Set-Content -LiteralPath (Join-Path $FixtureStage "vision-artifact.json") -Value $manifest -Encoding utf8NoBOM

$RuntimeArchive = Join-Path $OutputDirectory "vending-vision-windows-x86_64.zip"
$FixtureArchive = Join-Path $OutputDirectory "vending-vision-test-fixtures.zip"
Remove-Item -LiteralPath $RuntimeArchive, $FixtureArchive -Force -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $RuntimeStage "*") -DestinationPath $RuntimeArchive -CompressionLevel Optimal
Compress-Archive -Path (Join-Path $FixtureStage "*") -DestinationPath $FixtureArchive -CompressionLevel Optimal

Add-Type -AssemblyName System.IO.Compression.FileSystem
$runtimeZip = [IO.Compression.ZipFile]::OpenRead($RuntimeArchive)
try {
    $fixtureEntries = @($runtimeZip.Entries | Where-Object {
        $_.FullName -match '(^|/)fixtures(/|$)' -or $_.Name -match '\.mp4$'
    })
    if ($fixtureEntries.Count -gt 0) {
        throw "Runtime archive includes recorded-video fixtures: $($fixtureEntries.FullName -join ', ')"
    }
} finally {
    $runtimeZip.Dispose()
}

$delivery = @{
    schemaVersion = "vending-vision-main-artifacts/v1"
    commit = $Commit
    runtime = @{
        file = [IO.Path]::GetFileName($RuntimeArchive)
        sha256 = (Get-FileHash -Algorithm SHA256 $RuntimeArchive).Hash.ToLowerInvariant()
    }
    fixtures = @{
        file = [IO.Path]::GetFileName($FixtureArchive)
        sha256 = (Get-FileHash -Algorithm SHA256 $FixtureArchive).Hash.ToLowerInvariant()
    }
} | ConvertTo-Json -Depth 5
Set-Content -LiteralPath (Join-Path $OutputDirectory "vending-vision-main-artifacts.json") -Value $delivery -Encoding utf8NoBOM

Remove-Item -LiteralPath $StageDirectory -Recurse -Force
