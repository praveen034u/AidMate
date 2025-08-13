param(
  [int]$Port = 0,
  [string]$Workspace = "C:\SourceCode\AidMate"
)

$ErrorActionPreference = "Stop"
Write-Host "=== Patch Ollama Port in local files ===" -ForegroundColor Cyan

function Test-OllamaApi { param([int]$Port)
  try { Invoke-RestMethod -Uri ("http://localhost:" + $Port + "/api/tags") -Method GET -TimeoutSec 1 | Out-Null; return $true }
  catch { return $false }
}
function Find-OllamaPort { param([int]$Start=11434,[int]$End=11460)
  for ($p=$Start; $p -le $End; $p++) { if (Test-OllamaApi -Port $p) { return $p } }
  return 0
}
function Patch-File-Port { param([string]$Path,[int]$Port)
  if (-not (Test-Path $Path)) { return }
  $backup = "$Path.bak"; if (-not (Test-Path $backup)) { Copy-Item $Path $backup }
  $content = Get-Content $Path -Raw
  $content = $content -replace "http://localhost:\d+/api/generate",  ("http://localhost:" + $Port + "/api/generate")
  $content = $content -replace "http://localhost:\d+/api/embeddings", ("http://localhost:" + $Port + "/api/embeddings")
  $content = $content -replace "http://localhost:11434/api/generate",  ("http://localhost:" + $Port + "/api/generate")
  $content = $content -replace "http://localhost:11434/api/embeddings", ("http://localhost:" + $Port + "/api/embeddings")
  Set-Content -Path $Path -Value $content -Encoding UTF8
  Write-Host "[OK] Patched port in $Path (backup at $backup)" -ForegroundColor Green
}

if (-not (Test-Path $Workspace)) { throw "Workspace not found: $Workspace" }
Set-Location $Workspace

if ($Port -eq 0) {
  if ($env:OLLAMA_HOST -and ($env:OLLAMA_HOST -match ":(\d+)$")) {
    $Port = [int]$Matches[1]
  } else {
    $det = Find-OllamaPort -Start 11434 -End 11460
    if ($det -eq 0) { Write-Host "[WARN] Ollama not detected; defaulting to 11434." -ForegroundColor Yellow; $Port = 11434 }
    else { $Port = $det; Write-Host "[OK] Detected Ollama on port $Port." -ForegroundColor Green }
  }
}

Patch-File-Port -Path (Join-Path $Workspace "build_index.py") -Port $Port
Patch-File-Port -Path (Join-Path $Workspace "app.py")         -Port $Port
Write-Host "[DONE] Files patched to port $Port." -ForegroundColor Cyan
