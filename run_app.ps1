$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  Write-Host "未检测到 .venv，正在创建虚拟环境..."
  uv venv .venv
}

Write-Host "正在安装/检查依赖..."
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot ".uv-cache"
uv pip install -r requirements.txt --python .venv\Scripts\python.exe

Write-Host "启动 RAG 评测工作台：http://localhost:8501"
.\.venv\Scripts\python.exe -m streamlit run app/main.py --server.port 8501 --server.address localhost

