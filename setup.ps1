# One-time setup on Windows: creates .venv and installs everything.   .\setup.ps1
Set-Location $PSScriptRoot
py -3.11 -m venv .venv 2>$null; if ($LASTEXITCODE) { python -m venv .venv }
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q tests
Write-Host "Setup done. Next: .\run_demo.ps1" -ForegroundColor Green
