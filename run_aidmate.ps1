<#
  run_aidmate.ps1  (ASCII-safe)
  One-click launcher for AidMate

  Modes:
    - api   : Start Ollama, ensure Piper, activate .venv, run FastAPI (uvicorn), open web/index.html
    - voice : Start Ollama, run test_pipeline.ps1 (mic -> STT -> LLM -> TTS)

  Workspace: C:\SourceCode\AidMate
#>

param(
  [ValidateSet("api","voice")]
  [string]$Mode = "api",

  # FastAPI port
  [int]$ApiPort = 8003,

  # Preferred Ollama host:port (leave empty to auto-pick a free port starting at 11434)
  [string]$OllamaHost = ""
)

$ErrorActionPreference = "Stop"

# --------------------------
# Paths
# --------------------------
$workspace = "C:\SourceCode\AidMate"
$venvDir   = Join-Path $workspace ".venv"
$reqFile   = Join-Path $workspace "requirements.txt"
$appFile   = Join-Path $workspace "app.py"
$uiFile    = Join-Path $workspace "web\index.html"
$pipeline  = Join-Path $workspace "test_pipeline.ps1"

# Piper locations (some installs unzip to piper\piper\piper.exe)
$piperRoot1 = Join-Path $workspace "piper\piper.exe"
$piperRoot2 = Join-Path $workspace "piper\piper\piper.exe"
$voiceOnnx  = Join-Path $workspace "piper\en_US-amy-low.onnx"
$voiceCfg   = Join-Path $workspace "piper\en_US-amy-low.onnx.json"

Write-Host "=== AidMate Launcher ($Mode) ===" -ForegroundColor Cyan
Set-Location $workspace

# --------------------------
# Helpers
# --------------------------
function Test-HttpOk { param([string]$Url,[int]$TimeoutSec=2)
  try {
    $resp = Invoke-WebRequest -Uri $Url -Method GET -TimeoutSec $TimeoutSec -UseBasicParsing
    return ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 400)
  } catch { return $false }
}
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
  $deadline=(Get-Date).AddSeconds($Seconds)
  while((Get-Date) -lt $deadline) {
    if (Test-OllamaApi -HostPort $HostPort) { return $true }
    Start-Sleep -Milliseconds 800
  }
  return $false
}
function Start-Ollama {
  param([string]$PrefHostPort)

  # If the API is already up, honor it
  if ($env:OLLAMA_HOST -and (Test-OllamaApi -HostPort $env:OLLAMA_HOST)) {
    Write-Host "[OK] Ollama already running at $($env:OLLAMA_HOST)" -ForegroundColor Green
    return $env:OLLAMA_HOST
  }

  # Decide host:port
  if ([string]::IsNullOrWhiteSpace($PrefHostPort)) {
    $hostAddr = "127.0.0.1"
    $port = 11434   
    $hostPort = "$hostAddr`:$port"
  
  }

  # Resolve ollama.exe
  $exe = (Get-Command "ollama" -ErrorAction SilentlyContinue).Source
  if (-not $exe) {
    $cands = @("$Env:LOCALAPPDATA\Programs\Ollama\ollama.exe","$Env:ProgramFiles\Ollama\ollama.exe")
    $exe = ($cands | Where-Object { Test-Path $_ } | Select-Object -First 1)
  }
  if (-not $exe) { throw "Ollama not found. Install it or run the Ollama app once." }

  # Start with explicit --host (more reliable than only env)
  # Write-Host "[INFO] Starting Ollama: $exe serve --host $hostPort" -ForegroundColor Yellow
  # Start-Process -FilePath $exe -ArgumentList "serve --host $hostPort" -WindowStyle Minimized | Out-Null
  # if (-not (Wait-ForOllama -HostPort $hostPort -Seconds 20)) {
  #   Start-Process -FilePath "cmd.exe" -ArgumentList "/c start ""Ollama Serve"" `"$exe`" serve --host $hostAddr --port  $port" | Out-Null
  #   if (-not (Wait-ForOllama -HostPort $hostPort -Seconds 20)) {
  #     throw "Could not start Ollama on $hostPort. Try running: `"$exe`" serve --host $hostAddr --port  $port"
  #   }
  # }

  $env:OLLAMA_HOST = $hostPort
  Write-Host "[OK] Ollama running at http://$hostPort" -ForegroundColor Green
  return $hostPort
}
function Ensure-Piper {
  if (-not (Test-Path $piperRoot1) -and -not (Test-Path $piperRoot2)) {
    throw "Piper not found. Expected at: `n  $piperRoot1 `nor `n  $piperRoot2"
  }
  if (-not (Test-Path $voiceOnnx)) { throw "Missing voice model: $voiceOnnx" }
  if (-not (Test-Path $voiceCfg))  { throw "Missing voice config: $voiceCfg" }
  Write-Host "[OK] Piper files present." -ForegroundColor Green
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
function Open-UI {
  param(
    [int]$ApiPort,
    [string]$UiFile
  )
  $url = "http://localhost/"

  # Try the API URL first (a few quick retries while uvicorn finishes booting)
  $maxTries = 10
  for ($i=1; $i -le $maxTries; $i++) {
    if (Test-HttpOk -Url $url -TimeoutSec 2) {
      Write-Host "[INFO] Opening web UI at $url ..." -ForegroundColor Yellow
      Start-Process $url | Out-Null
      return
    }
    Start-Sleep -Milliseconds 500
  }

  # Fallback: open the static HTML file directly if present
  if (Test-Path $UiFile) {
    Write-Host "[WARN] API not reachable yet; opening local UI file instead: $UiFile" -ForegroundColor Yellow
    Start-Process $UiFile | Out-Null
  } else {
    Write-Host "[WARN] Neither API ($url) nor UI file ($UiFile) available. You can still hit the API later at $url" -ForegroundColor Yellow
  }
}

# --------------------------
# 1) Start Ollama
# --------------------------
$ollamaHost = Start-Ollama -PrefHostPort $OllamaHost

# --------------------------
# 2) Mode handling
# --------------------------
switch ($Mode) {
  "api" {
    # Make sure Piper assets exist (app.py depends on them)
    Ensure-Piper

    # Prepare Python venv + deps
    Ensure-Venv-And-Install

    if (-not (Test-Path $appFile)) {
      Write-Host "[ERROR] app.py not found in $workspace" -ForegroundColor Red
      exit 1
    }

    # Start FastAPI (uvicorn) in background so we can open the UI
    Write-Host "[INFO] Starting FastAPI at http://localhost:$ApiPort ..." -ForegroundColor Yellow
    $pythonExe = "$venvDir\Scripts\python.exe"
    $uv = Start-Process -FilePath $pythonExe -ArgumentList "-m uvicorn app:app --host 0.0.0.0 --port $ApiPort --reload" -PassThru
    Start-Sleep -Seconds 2
    Write-Host "[OK] FastAPI started (PID $($uv.Id))" -ForegroundColor Green

    # Open the HTML UI if present
    Open-UI -ApiPort 8080 -UiFile $uiFile

    Write-Host ""
    Write-Host "API:    http://localhost:$ApiPort" -ForegroundColor Cyan
    Write-Host "Stop:   taskkill /PID $($uv.Id) /F" -ForegroundColor Cyan
  }

  "voice" {
    if (-not (Test-Path $pipeline)) {
      Write-Host "[ERROR] test_pipeline.ps1 not found in $workspace" -ForegroundColor Red
      exit 1
    }
    Write-Host "[INFO] Running voice pipeline..." -ForegroundColor Yellow
    powershell -ExecutionPolicy Bypass -File $pipeline
  }
}
