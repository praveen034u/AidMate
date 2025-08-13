<#
  run_aidmate.ps1  (ASCII-safe)
  One-click launcher for AidMate
  - Starts Ollama on a free port (11434+), honoring OLLAMA_HOST if already set
  - Activates .venv and installs requirements (if present)
  - Mode "api": runs FastAPI (uvicorn) and opens web/index.html
  - Mode "voice": runs test_pipeline.ps1 (mic -> STT -> LLM -> TTS)
  Workspace: C:\SourceCode\AidMate
#>

param(
  [ValidateSet("api","voice")]
  [string]$Mode = "api",

  # Web server port for FastAPI (your browser UI talks to this)
  [int]$ApiPort = 7860
)

$ErrorActionPreference = "Stop"
$workspace = "C:\SourceCode\AidMate"
$venvDir   = Join-Path $workspace ".venv"
$reqFile   = Join-Path $workspace "requirements.txt"
$appFile   = Join-Path $workspace "app.py"
$uiFile    = Join-Path $workspace "web\index.html"
$pipeline  = Join-Path $workspace "test_pipeline.ps1"

Write-Host "=== AidMate Launcher ($Mode) ===" -ForegroundColor Cyan
Set-Location $workspace

# --------------------------
# Helpers
# --------------------------
function Test-PortFree { param([int]$Port)
  $net = netstat -ano | Select-String ":$Port\s"
  if (-not $net) { return $true } else { return $false }
}
function Find-FreePort { param([int]$Start,[int]$Tries=20)
  for ($p=$Start; $p -lt ($Start+$Tries); $p++) { if (Test-PortFree -Port $p) { return $p } }
  throw "Could not find a free port starting at $Start"
}
function Test-OllamaApi { param([string]$HostPort)
  try { Invoke-RestMethod -Uri "http://$HostPort/api/tags" -Method GET -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}
function Wait-ForOllama { param([string]$HostPort,[int]$Seconds=25)
  $deadline=(Get-Date).AddSeconds($Seconds); while((Get-Date)-lt $deadline){ if(Test-OllamaApi -HostPort $HostPort){return $true}; Start-Sleep -Milliseconds 800 }; return $false
}
function Start-Ollama {
  # Honor existing OLLAMA_HOST if the API is up
  if ($env:OLLAMA_HOST -and (Test-OllamaApi -HostPort $env:OLLAMA_HOST)) {
    Write-Host "[OK] Ollama already running at $($env:OLLAMA_HOST)" -ForegroundColor Green
    return $env:OLLAMA_HOST
  }

  # Choose a host:port
  $hostPort = if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST } else { "127.0.0.1:11434" }
  $host,$port = $hostPort.Split(":")
  if (-not (Test-PortFree -Port [int]$port)) {
    $port = Find-FreePort -Start 11434 -Tries 20
    $hostPort = "$host`:$port"
    Write-Host "[INFO] Default port busy; using $hostPort" -ForegroundColor Yellow
  }

  # Start with explicit --host (more reliable than env on some Windows builds)
  $exe = (Get-Command "ollama" -ErrorAction SilentlyContinue).Source
  if (-not $exe) {
    $cands = @("$Env:LOCALAPPDATA\Programs\Ollama\ollama.exe","$Env:ProgramFiles\Ollama\ollama.exe")
    $exe = ($cands | Where-Object { Test-Path $_ } | Select-Object -First 1)
  }
  if (-not $exe) { throw "Ollama not found. Install with winget or launch the Ollama app once." }

  Write-Host "[INFO] Starting Ollama: $exe serve --host $hostPort" -ForegroundColor Yellow
  Start-Process -FilePath $exe -ArgumentList "serve --host $hostPort" -WindowStyle Minimized | Out-Null
  if (-not (Wait-ForOllama -HostPort $hostPort -Seconds 20)) {
    # Try visible window once
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c start ""Ollama Serve"" `"$exe`" serve --host $hostPort" | Out-Null
    if (-not (Wait-ForOllama -HostPort $hostPort -Seconds 20)) {
      throw "Could not start Ollama on $hostPort. Try running: `"$exe`" serve --host $hostPort"
    }
  }
  # Export OLLAMA_HOST for this session (do not set globally here)
  $env:OLLAMA_HOST = $hostPort
  Write-Host "[OK] Ollama running at http://$hostPort" -ForegroundColor Green
  return $hostPort
}
function Ensure-Venv-And-Install {
  if (-not (Test-Path $venvDir)) {
    Write-Host "[INFO] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
  }
  & "$venvDir\Scripts\Activate.ps1"
  Write-Host "[OK] venv activated" -ForegroundColor Green

  if (Test-Path $reqFile) {
    Write-Host "[INFO] Installing requirements..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "[OK] Requirements installed" -ForegroundColor Green
  } else {
    Write-Host "[WARN] requirements.txt not found; skipping installs" -ForegroundColor Yellow
  }
}

# --------------------------
# 1) Start Ollama
# --------------------------
$ollamaHost = Start-Ollama

# --------------------------
# 2) Mode selection
# --------------------------
switch ($Mode) {
  "api" {
    # Prepare Python env
    Ensure-Venv-And-Install

    # Warn if app.py missing
    if (-not (Test-Path $appFile)) {
      Write-Host "[ERROR] app.py not found in $workspace" -ForegroundColor Red
      Write-Host "Create app.py (FastAPI) or run: .\run_aidmate.ps1 -Mode voice" -ForegroundColor Yellow
      exit 1
    }

    # Start API (non-blocking so we can open UI)
    Write-Host "[INFO] Starting FastAPI on http://localhost:$ApiPort ..." -ForegroundColor Yellow
    $uv = Start-Process -FilePath "$venvDir\Scripts\python.exe" -ArgumentList "-m uvicorn app:app --host 0.0.0.0 --port $ApiPort --reload" -PassThru
    Start-Sleep -Seconds 2
    Write-Host "[OK] FastAPI started (PID $($uv.Id))" -ForegroundColor Green

    # Open browser UI if present
    if (Test-Path $uiFile) {
      Write-Host "[INFO] Opening web UI..." -ForegroundColor Yellow
      Start-Process $uiFile | Out-Null
    } else {
      Write-Host "[WARN] web\index.html not found. You can still call the API at http://localhost:$ApiPort" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "Stop server:  taskkill /PID $($uv.Id) /F" -ForegroundColor Cyan
  }

  "voice" {
    if (-not (Test-Path $pipeline)) {
      Write-Host "[ERROR] test_pipeline.ps1 not found in $workspace" -ForegroundColor Red
      exit 1
    }
    Write-Host "[INFO] Running voice pipeline (you will be prompted for your mic)..." -ForegroundColor Yellow
    powershell -ExecutionPolicy Bypass -File $pipeline
  }
}
