$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$tracked = git ls-files
$targets = $tracked | Where-Object {
  $_ -like ".pip-cache/*" -or
  $_ -like "app/data/*" -or
  $_ -like "app/exports/*" -or
  $_ -like "*__pycache__/*"
}

if (-not $targets) {
  Write-Host "No tracked runtime artifacts found."
  exit 0
}

foreach ($path in $targets) {
  git rm --cached -- $path
}

Write-Host "Removed $($targets.Count) runtime artifacts from the Git index; local files were kept."
