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
function Read-ZipEntries([string]$Archive) {
    $zip = [IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        return @($zip.Entries | ForEach-Object { $_.FullName })
    } finally {
        $zip.Dispose()
    }
}

function Read-ZipJson([string]$Archive, [string]$EntryName) {
    $zip = [IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        $entry = $zip.GetEntry($EntryName)
        if ($null -eq $entry) {
            throw "Archive $Archive is missing $EntryName"
        }
        $reader = [IO.StreamReader]::new($entry.Open())
        try {
            return $reader.ReadToEnd() | ConvertFrom-Json
        } finally {
            $reader.Dispose()
        }
    } finally {
        $zip.Dispose()
    }
}

$runtimeEntries = Read-ZipEntries $RuntimeArchive
$fixtureEntries = Read-ZipEntries $FixtureArchive
if ($runtimeEntries -notcontains "vending-vision.exe") {
    throw "Runtime archive must contain vending-vision.exe at its root"
}
if ($runtimeEntries -notcontains "vision-artifact.json") {
    throw "Runtime archive is missing vision-artifact.json"
}
if ($fixtureEntries -notcontains "recorded-video/expected-results.json" -or
    $fixtureEntries -notcontains "recorded-video/top.mp4" -or
    $fixtureEntries -notcontains "recorded-video/front.mp4" -or
    $fixtureEntries -notcontains "vision-artifact.json") {
    throw "Fixture archive layout is incomplete"
}
$runtimeFixtureEntries = @($runtimeEntries | Where-Object {
    $_ -match '(^|/)fixtures(/|$)' -or $_ -match '(^|/)recorded-video(/|$)' -or $_ -match '\.mp4$'
})
if ($runtimeFixtureEntries.Count -gt 0) {
    throw "Runtime archive includes recorded-video fixtures: $($runtimeFixtureEntries -join ', ')"
}

foreach ($archive in @($RuntimeArchive, $FixtureArchive)) {
    $embeddedManifest = Read-ZipJson $archive "vision-artifact.json"
    if ($embeddedManifest.schemaVersion -ne "vending-vision-main-artifacts/v1" -or
        $embeddedManifest.commit -ne $Commit -or
        $embeddedManifest.runtimeArchive -ne "vending-vision-windows-x86_64.zip" -or
        $embeddedManifest.fixtureArchive -ne "vending-vision-test-fixtures.zip") {
        throw "Archive manifest does not match delivery contract: $archive"
    }
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
$DeliveryPath = Join-Path $OutputDirectory "vending-vision-main-artifacts.json"
Set-Content -LiteralPath $DeliveryPath -Value $delivery -Encoding utf8NoBOM

$deliveryCheck = Get-Content -LiteralPath $DeliveryPath -Raw | ConvertFrom-Json
if ($deliveryCheck.schemaVersion -ne "vending-vision-main-artifacts/v1" -or
    $deliveryCheck.commit -ne $Commit -or
    $deliveryCheck.runtime.file -ne [IO.Path]::GetFileName($RuntimeArchive) -or
    $deliveryCheck.fixtures.file -ne [IO.Path]::GetFileName($FixtureArchive) -or
    $deliveryCheck.runtime.sha256 -ne (Get-FileHash -Algorithm SHA256 $RuntimeArchive).Hash.ToLowerInvariant() -or
    $deliveryCheck.fixtures.sha256 -ne (Get-FileHash -Algorithm SHA256 $FixtureArchive).Hash.ToLowerInvariant()) {
    throw "Delivery manifest does not match packaged archives"
}

Remove-Item -LiteralPath $StageDirectory -Recurse -Force
