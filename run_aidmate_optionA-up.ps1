<#
  run_aidmate_optionA.ps1
  One-click setup/launch for AidMate using Option A (Ollama alias -> small open model).

  USAGE (PowerShell as Admin for firewall/winget):
    Set-ExecutionPolicy Bypass -Scope Process -Force
    .\run_aidmate_optionA.ps1 -Mode api -Workspace "C:\SourceCode\AidMate" -ApiPort 8000

  Modes:
    - api   : Start Ollama, ensure alias gpt-oss:latest, create/activate .venv, install reqs, run FastAPI (uvicorn), optional open web UI
    - voice : Start Ollama, ensure alias, run test_pipeline.ps1 (mic -> STT -> LLM -> TTS)
#>

[CmdletBinding()]
param(
  [ValidateSet("api","voice")]
  [string]$Mode = "api",

  [int]$ApiPort = 8000,

  [string]$Workspace = "C:\SourceCode\AidMate",

  # Backing small model; change to "llama3.2:3b-instruct" or "qwen2.5:3b-instruct" if preferred
  [string]$BackModel = "phi3:mini",

  # Alias your app will call
  [string]$ModelAlias = "gpt-oss:latest",

  # Optional: open a local HTML after API launch
  [string]$UiFile = "$Workspace\web\index.html"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Paths ---
$AppPy       = Join-Path $Workspace "app.py"
$ReqFile     = Join-Path $Workspace "requirements.txt"
$VenvPath    = Join-Path $Workspace ".venv"
$PyExe       = Join-Path $VenvPath "Scripts\python.exe"
$UvicornArgs = @("-m","uvicorn","app:app","--host","0.0.0.0","--port",$ApiPort.ToString())
$PipelinePS1 = Join-Path $Workspace "test_pipeline.ps1"

function Resolve-OllamaExe {
  $cmd = Get-Command ollama -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $candidates = @(
    "$env:ProgramFiles\Ollama\ollama.exe",
    "$env:ProgramFiles(x86)\Ollama\ollama.exe",
    "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
  )
  foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
  return $null
}

function Ensure-Ollama {
  $ollama = Resolve-OllamaExe
  if (-not $ollama) {
    Write-Host "[INFO] Ollama not found. Installing via winget..." -ForegroundColor Yellow
    winget install --id Ollama.Ollama -e --source winget --silent
    Start-Sleep -Seconds 5
    $ollama = Resolve-OllamaExe
    if (-not $ollama) { throw "Ollama install failed. Install from https://ollama.com/download/windows and re-run." }
  }
  Write-Host "[OK] Using Ollama at: $ollama" -ForegroundColor Green

  try { Start-Service -Name "Ollama" -ErrorAction SilentlyContinue } catch {}
  if (-not (Get-Process -Name "ollama" -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Minimized | Out-Null
    Start-Sleep -Seconds 3
  }

  # CPU-friendly env
  $env:OLLAMA_NUM_PARALLEL = "1"
  $env:OMP_NUM_THREADS = [Environment]::ProcessorCount.ToString()
  setx OLLAMA_NUM_PARALLEL "1" | Out-Null
  setx OMP_NUM_THREADS ([Environment]::ProcessorCount.ToString()) | Out-Null

  return $ollama
}

function Ensure-GptOssAlias {
  param([string]$ollama)

  Write-Host "[INFO] Ensuring backing model: $BackModel" -ForegroundColor Yellow
  & $ollama pull $BackModel

  $modelfile = @"
FROM $BackModel
PARAMETER temperature 0.3
PARAMETER num_ctx 2048
TEMPLATE """{{ .System }}

User: {{ .Prompt }}
Assistant:"""
"@
  $modelfilePath = Join-Path $env:TEMP "Modelfile_gpt_oss.txt"
  $modelfile | Set-Content -Path $modelfilePath -Encoding UTF8

  Write-Host "[INFO] Creating internal model and alias $ModelAlias -> $BackModel" -ForegroundColor Yellow
  & $ollama create "gpt-oss:mini" -f $modelfilePath
  try { & $ollama rm $ModelAlias | Out-Null } catch {}
  & $ollama copy "gpt-oss:mini" $ModelAlias
  Write-Host "[OK] Alias ready: $ModelAlias -> $BackModel" -ForegroundColor Green
}

function Ensure-Venv {
  if (-not (Test-Path $VenvPath)) {
    Write-Host "[INFO] Creating venv at $VenvPath..." -ForegroundColor Yellow
    python -m venv $VenvPath
  }
  $activate = Join-Path $VenvPath "Scripts\Activate.ps1"
  if (-not (Test-Path $activate)) {
    throw "Python venv activation script not found at $activate. Ensure Python is installed and re-run."
  }
  . $activate
  Write-Host "[OK] venv activated." -ForegroundColor Green

  Write-Host "[INFO] Upgrading pip/setuptools/wheel..." -ForegroundColor Yellow
  pip install --upgrade pip wheel setuptools

  if (Test-Path $ReqFile) {
    Write-Host "[INFO] Installing requirements.txt..." -ForegroundColor Yellow
    pip install -r $ReqFile
  } else {
    # Minimal stack to match Option A changes in app.py
    Write-Host "[INFO] Installing minimal dependencies..." -ForegroundColor Yellow
    pip install fastapi "uvicorn[standard]" pydantic requests python-multipart
    pip install langchain langchain-community langchain-ollama
    pip install "faiss-cpu==1.8.0" "sentence-transformers>=2.7.0" "transformers>=4.44.0"
  }
  Write-Host "[OK] Python deps installed." -ForegroundColor Green
}

function Open-Firewall {
  $rule = "AidMate-API-Port-$ApiPort"
  if (-not (Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue)) {
    Write-Host "[INFO] Opening Windows Firewall for TCP $ApiPort..." -ForegroundColor Yellow
    New-NetFirewallRule -DisplayName $rule -Direction Inbound -Profile Any -Action Allow -Protocol TCP -LocalPort $ApiPort | Out-Null
  }
}

function Start-API {
  if (-not (Test-Path $AppPy)) {
    throw "app.py not found at $AppPy. Copy your updated app.py to $Workspace and re-run."
  }
  Open-Firewall
  Push-Location $Workspace
  Write-Host "[INFO] Launching FastAPI on port $ApiPort..." -ForegroundColor Yellow
  $proc = Start-Process -FilePath $PyExe -ArgumentList $UvicornArgs -PassThru -WindowStyle Minimized
  Pop-Location

  Write-Host ""
  Write-Host "[OK] API PID: $($proc.Id)" -ForegroundColor Green
  Write-Host "Docs:  http://localhost:$ApiPort/docs" -ForegroundColor Cyan
  Write-Host "Stop:  taskkill /PID $($proc.Id) /F" -ForegroundColor Cyan

  if (Test-Path $UiFile) {
    Write-Host "[INFO] Opening UI: $UiFile" -ForegroundColor Yellow
    Start-Process $UiFile | Out-Null
  }
}

# --- MAIN ---
if (-not (Test-Path $Workspace)) { throw "Workspace not found: $Workspace" }

$ollama = Ensure-Ollama
Ensure-GptOssAlias -ollama $ollama

switch ($Mode) {
  "api"   {
    Ensure-Venv
    Start-API
  }
  "voice" {
    if (-not (Test-Path $PipelinePS1)) {
      throw "test_pipeline.ps1 not found at $PipelinePS1"
    }
    & powershell -ExecutionPolicy Bypass -File $PipelinePS1
  }
}
