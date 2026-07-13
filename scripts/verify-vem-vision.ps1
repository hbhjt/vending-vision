param([int]$TimeoutSeconds = 30)
$ErrorActionPreference = 'Stop'
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do {
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:7892/health' -TimeoutSec 3
        $version = Invoke-RestMethod -Uri 'http://127.0.0.1:7892/version' -TimeoutSec 3
        [pscustomobject]@{
            status = $health.status
            protocol = $health.protocol
            version = $health.version
            mockScenario = $health.mockScenario
            cameraReady = $health.cameraReady
            modelReady = $health.modelReady
            profilePush = $version.profile_push.enabled
        } | ConvertTo-Json -Compress
        exit 0
    } catch {
        Start-Sleep -Milliseconds 500
    }
} while ((Get-Date) -lt $deadline)
throw 'VEM vision health endpoint did not become reachable'
