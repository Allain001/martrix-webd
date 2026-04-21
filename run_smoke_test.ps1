$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendDir = Join-Path $root "frontend"
$backendDir = Join-Path $root "backend"
$scriptPath = Join-Path $root "scripts\smoke_test.py"

Write-Host "[MatrixVis] checking backend dependencies..." -ForegroundColor Cyan
Push-Location $backendDir
python -c "import fastapi, uvicorn, numpy" | Out-Null
Pop-Location

Write-Host "[MatrixVis] building frontend..." -ForegroundColor Cyan
Push-Location $frontendDir
npm.cmd run build
Pop-Location

Write-Host "[MatrixVis] running smoke test..." -ForegroundColor Cyan
python $scriptPath
