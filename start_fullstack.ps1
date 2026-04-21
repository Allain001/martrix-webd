$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendDir = Join-Path $root "frontend"
$backendDir = Join-Path $root "backend"
$siteUrl = "http://127.0.0.1:8000"

Write-Host "[MatrixVis] checking backend dependencies..." -ForegroundColor Cyan
$backendReady = $true
try {
    Push-Location $backendDir
    python -c "import fastapi, uvicorn, numpy" | Out-Null
}
catch {
    $backendReady = $false
}
finally {
    Pop-Location
}

if (-not $backendReady) {
    Write-Host "[MatrixVis] installing backend dependencies..." -ForegroundColor Cyan
    Push-Location $backendDir
    python -m pip install -r requirements.txt
    Pop-Location
}

Write-Host "[MatrixVis] installing frontend dependencies if needed..." -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
    Push-Location $frontendDir
    npm.cmd install
    Pop-Location
}

Write-Host "[MatrixVis] building frontend for single-site serving..." -ForegroundColor Cyan
Push-Location $frontendDir
npm.cmd run build
Pop-Location

Write-Host "[MatrixVis] starting website on $siteUrl" -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoProfile",
    "-Command",
    "Set-Location '$backendDir'; python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
) | Out-Null

Start-Sleep -Seconds 4
Start-Process $siteUrl | Out-Null

Write-Host "[MatrixVis] browser opened. This is now a single website served by FastAPI." -ForegroundColor Green
