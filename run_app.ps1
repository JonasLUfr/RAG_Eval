$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  Write-Host "No .venv detected; creating virtual environment..."
  uv venv .venv
}

Write-Host "Installing/checking dependencies..."
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot ".uv-cache"
$requirementsFile = "requirements.txt"
if (Test-Path "requirements.lock.txt") {
  $requirementsFile = "requirements.lock.txt"
  Write-Host "requirements.lock.txt detected; installing pinned dependencies."
}
uv pip install -r $requirementsFile --python .venv\Scripts\python.exe

Write-Host "Starting RAG Eval Workbench: http://localhost:8501"
.\.venv\Scripts\python.exe -m streamlit run app/main.py --server.port 8501 --server.address localhost
