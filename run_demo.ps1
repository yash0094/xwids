# XWIDS demo launcher (Windows PowerShell)
#   .\run_demo.ps1                       # dashboard on http://127.0.0.1:5000 (synthetic data)
#   .\run_demo.ps1 -Dataset awid3        # after preparing + training on AWID3
#   .\run_demo.ps1 -Public               # also opens a free Cloudflare tunnel -> public https link for judges
param([string]$Dataset = "synthetic", [int]$Port = 5000, [switch]$Public, [switch]$KeepData)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (Test-Path ".venv\Scripts\Activate.ps1") { . .venv\Scripts\Activate.ps1 }
if (-not (Test-Path "models\$Dataset\registry.json")) {
  Write-Host "No trained model for '$Dataset' yet - preparing data and training first (a few minutes)..." -ForegroundColor Yellow
  python scripts\01_prepare.py --dataset $Dataset; if ($LASTEXITCODE) { exit 1 }
  python scripts\02_train.py --dataset $Dataset;   if ($LASTEXITCODE) { exit 1 }
}
if (-not $env:XWIDS_USERS)      { $env:XWIDS_USERS = "analyst:xwids-demo,reviewer:xwids-review" }
if (-not $env:XWIDS_SECRET_KEY) { $env:XWIDS_SECRET_KEY = [guid]::NewGuid().ToString() + [guid]::NewGuid().ToString() }
$env:XWIDS_DATASET = $Dataset; $env:PORT = "$Port"; $env:HOST = "127.0.0.1"
if (-not $KeepData) { $env:XWIDS_RESET = "1" }
if ($Public) {
  if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "Install the tunnel once:  winget install --id Cloudflare.cloudflared" -ForegroundColor Yellow; exit 1 }
  Start-Process cloudflared -ArgumentList "tunnel --url http://127.0.0.1:$Port"
  Write-Host "The public https link appears in the new cloudflared window." -ForegroundColor Cyan
}
Write-Host "Logins: $env:XWIDS_USERS" -ForegroundColor Cyan
Write-Host "Open http://127.0.0.1:$Port  (phone view: http://127.0.0.1:$Port/doctor/)" -ForegroundColor Green
python serve.py
