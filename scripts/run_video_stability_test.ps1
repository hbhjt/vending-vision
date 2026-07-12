param(
    [string]$OutputDir = "reports/video_dataset/test01",
    [int]$FrameStep = 5,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv-packaging\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

$TopVideo = Get-ChildItem -LiteralPath $Root -Filter "*顶部*.mp4" -File |
    Select-Object -First 1
$FrontVideo = Get-ChildItem -LiteralPath $Root -Filter "*中部*.mp4" -File |
    Select-Object -First 1

if (-not $TopVideo -or -not $FrontVideo) {
    throw "根目录必须各有一个名称包含‘顶部’和‘中部’的 MP4 文件。"
}

$env:PYTHONIOENCODING = "utf-8"

& $Python (Join-Path $PSScriptRoot "build_video_dataset.py") `
    --top-video $TopVideo.FullName `
    --front-video $FrontVideo.FullName `
    --output-dir $OutputDir `
    --top-frame-step $FrameStep `
    --front-frame-step $FrameStep
if ($LASTEXITCODE -ne 0) {
    throw "视频数据集构建失败，退出码：$LASTEXITCODE"
}

& $Python (Join-Path $PSScriptRoot "analyze_video_stability.py") `
    --dataset-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "稳定性分析失败，退出码：$LASTEXITCODE"
}

$Report = Join-Path $Root "$OutputDir\stability_report.html"
Write-Host "稳定性报告：$Report"

if (-not $NoOpen) {
    Start-Process $Report
}
