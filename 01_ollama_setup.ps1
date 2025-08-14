<#
  setup_steps1_3.ps1  (ASCII-safe)
  Steps 1-3: Ollama + Whisper.cpp + Piper
  - Auto-detects free port for Ollama (11434+)
  - Sets OLLAMA_HOST
  - Patches build_index.py and app.py to chosen port
  Workspace: C:\SourceCode\AidMate
#>

$ErrorActionPreference = "Stop"
Write-Host "=== AidMate Setup (Steps 1-3, auto-port) ===" -ForegroundColor Cyan

# --------------------------
# Config
# --------------------------
$workspace = "C:\SourceCode\AidMate"
$startPort = 11434
$maxTries  = 20

# --------------------------
# Helpers
# --------------------------
function Assert-Admin {
  $id  = [Security.Principal.WindowsIdentity]::GetCurrent()
  $pri = New-Object Security.Principal.WindowsPrincipal($id)
  if (-not $pri.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Please run this script in an elevated PowerShell (Run as administrator)."
  }
}
function Ensure-Winget {
  try { winget --version | Out-Null }
  catch { throw "'winget' is not available. Install 'App Installer' from Microsoft Store and rerun." }
}
function Get-WingetPackageInstalled {
  param([string]$Id)
  $list = winget list --id $Id --source winget 2>$null
  return ($list -and ($list -match [regex]::Escape($Id)))
}
function Install-WithWinget {
  param([Parameter(Mandatory=$true)][string]$Id,[Parameter(Mandatory=$true)][string]$DisplayName)
  if (Get-WingetPackageInstalled -Id $Id) {
    Write-Host "[OK] $DisplayName already installed ($Id)" -ForegroundColor Green
    return
  }
  Write-Host "[INFO] Installing $DisplayName ($Id)..." -ForegroundColor Yellow
  winget install --id $Id -e --source winget --accept-package-agreements --accept-source-agreements
  Write-Host "[OK] Installed $DisplayName" -ForegroundColor Green
}
function Download-IfMissing {
  param([string]$Url,[string]$Path)
  if (-not (Test-Path $Path)) {
    Write-Host "[INFO] Downloading $(Split-Path $Path -Leaf) ..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $Url -OutFile $Path
    Write-Host "[OK] Downloaded: $Path" -ForegroundColor Green
  } else { Write-Host "[OK] Exists: $Path" -ForegroundColor Green }
}
function Expand-IfMissing {
  param([string]$Zip,[string]$Dest)
  if (-not (Test-Path $Dest)) {
    Write-Host "[INFO] Extracting $Zip ..." -ForegroundColor Yellow
    Expand-Archive $Zip -DestinationPath $Dest -Force
    Write-Host "[OK] Extracted to: $Dest" -ForegroundColor Green
  } else { Write-Host "[OK] Exists: $Dest" -ForegroundColor Green }
}
function Test-PortFree {
  param([int]$Port)
  # Free if nothing is LISTENING on the port
  $net = netstat -ano | Select-String ":$Port\s"
  if (-not $net) { return $true } else { return $false }
}
function Find-FreePort {
  param([int]$Start,[int]$Tries)
  for ($p=$Start; $p -lt ($Start+$Tries); $p++) {
    if (Test-PortFree -Port $p) { return $p }
  }
  throw "Could not find a free port starting at $Start."
}
function Test-OllamaApi {
  param([int]$Port)
  try {
    Invoke-RestMethod -Uri "http://localhost:$Port/api/tags" -Method GET -TimeoutSec 2 | Out-Null
    return $true
  } catch { return $false }
}
function Wait-ForOllama {
  param([int]$Port,[int]$Seconds = 25)
  $deadline = (Get-Date).AddSeconds($Seconds)
  while ((Get-Date) -lt $deadline) {
    if (Test-OllamaApi -Port $Port) { return $true }
    Start-Sleep -Milliseconds 800
  }
  return $false
}
function Start-Ollama {
  param([int]$Port)

  if (Test-OllamaApi -Port $Port) {
    Write-Host "[OK] Ollama service already running on port $Port." -ForegroundColor Green
    return
  }

  # Ensure env var for CLI and service
  $global:env:OLLAMA_HOST = "127.0.0.1:$Port"
  setx OLLAMA_HOST "127.0.0.1:$Port" /M | Out-Null
  Write-Host "[OK] OLLAMA_HOST set to 127.0.0.1:$Port" -ForegroundColor Green

  # Try via PATH
  $cmd = Get-Command "ollama" -ErrorAction SilentlyContinue
  if ($cmd) {
    Write-Host "[INFO] Starting Ollama via PATH: $($cmd.Source) on port $Port ..." -ForegroundColor Yellow
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c start ""Ollama Serve"" `"$exe`" serve --host 127.0.0.1:$Port" | Out-Null
    if (Wait-ForOllama -Port $Port -Seconds 12) { Write-Host "[OK] Ollama started." -ForegroundColor Green; return }
  }

  # Try common install folders
  $candidates = @(
    "$Env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
    "$Env:ProgramFiles\Ollama\ollama.exe"
  )
  foreach ($p in $candidates) {
    if (Test-Path $p) {
      Write-Host "[INFO] Starting Ollama at: $p on port $Port" -ForegroundColor Yellow
      Start-Job -ScriptBlock { param($path,$h) & $path serve --host $h } -ArgumentList $exe,"127.0.0.1:$Port" | Out-Null
      if (Wait-ForOllama -Port $Port -Seconds 12) { Write-Host "[OK] Ollama started." -ForegroundColor Green; return }
    }
  }

  # Visible window to surface firewall prompt
  $exe = if ($cmd) { $cmd.Source } elseif (Test-Path $candidates[0]) { $candidates[0] } else { $candidates[1] }
  if ($exe) {
    Write-Host "[INFO] Starting visible window for firewall prompt..." -ForegroundColor Yellow
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c start ""Ollama Serve"" `"$exe`" serve" | Out-Null
    if (Wait-ForOllama -Port $Port -Seconds 20) { Write-Host "[OK] Ollama started." -ForegroundColor Green; return }
  }

  # Background job fallback
  if ($exe) {
    Write-Host "[INFO] Starting Ollama as background job..." -ForegroundColor Yellow
    Start-Job -ScriptBlock { param($path) & $path serve } -ArgumentList $exe | Out-Null
    if (Wait-ForOllama -Port $Port -Seconds 20) { Write-Host "[OK] Ollama started." -ForegroundColor Green; return }
  }

  throw "Could not start Ollama on port $Port. Try running 'ollama serve' in a separate terminal, then re-run this script."
}
function Patch-File-Port {
  param([string]$Path,[int]$Port)
  if (-not (Test-Path $Path)) { return }

  $backup = "$Path.bak"
  if (-not (Test-Path $backup)) {
    Copy-Item $Path $backup
  }

  $content = Get-Content $Path -Raw

  # Replace common patterns
  $content = $content -replace "http://localhost:\d+/api/generate",  "http://localhost:$Port/api/generate"
  $content = $content -replace "http://localhost:\d+/api/embeddings","http://localhost:$Port/api/embeddings"

  # Also handle exact 11434 occurrences if any remained
  $content = $content -replace "http://localhost:11434/api/generate",  "http://localhost:$Port/api/generate"
  $content = $content -replace "http://localhost:11434/api/embeddings","http://localhost:$Port/api/embeddings"

  Set-Content -Path $Path -Value $content -Encoding UTF8
  Write-Host "[OK] Patched port in $Path (backup at $backup)" -ForegroundColor Green
}

# --------------------------
# Start
# --------------------------
Assert-Admin
Ensure-Winget

# Workspace
if (-not (Test-Path $workspace)) {
  New-Item -ItemType Directory -Path $workspace | Out-Null
  Write-Host "[OK] Created workspace: $workspace" -ForegroundColor Green
} else { Write-Host "[OK] Workspace exists: $workspace" -ForegroundColor Green }
Set-Location $workspace

# Ensure FFmpeg
Install-WithWinget -Id "Gyan.FFmpeg" -DisplayName "FFmpeg"

# Step 1: Ollama (install + choose port + start)
Install-WithWinget -Id "Ollama.Ollama" -DisplayName "Ollama"

# If default port is busy, auto-pick next free
$chosenPort = $startPort
if (-not (Test-PortFree -Port $chosenPort)) {
  Write-Host "[WARN] Port $chosenPort busy. Searching for a free port..." -ForegroundColor Yellow
  $chosenPort = Find-FreePort -Start $startPort -Tries $maxTries
  Write-Host "[OK] Using port $chosenPort for Ollama." -ForegroundColor Green
} else {
  Write-Host "[OK] Port $chosenPort is free for Ollama." -ForegroundColor Green
}

Start-Ollama -Port $chosenPort

Write-Host "[INFO] Pulling Ollama models..." -ForegroundColor Yellow
& ollama pull llama3.1:8b
& ollama pull nomic-embed-text
Write-Host "[OK] Ollama models ready." -ForegroundColor Green

# Patch local files to chosen port (if they exist)
Patch-File-Port -Path (Join-Path $workspace "build_index.py") -Port $chosenPort
Patch-File-Port -Path (Join-Path $workspace "app.py")         -Port $chosenPort

# Step 2: Whisper.cpp
$whisperDir = Join-Path $workspace "whisper"
if (-not (Test-Path $whisperDir)) { New-Item -ItemType Directory -Path $whisperDir | Out-Null }
$whisperZip = Join-Path $workspace "whisper.zip"
$whisperUrl = "https://github.com/ggerganov/whisper.cpp/releases/latest/download/whisper-cpp-win-x64.zip"
Download-IfMissing -Url $whisperUrl -Path $whisperZip
Expand-IfMissing -Zip $whisperZip -Dest $whisperDir
$whisperModel = Join-Path $whisperDir "ggml-base.en.bin"
if (-not (Test-Path $whisperModel)) {
  Write-Host "[INFO] Downloading whisper model..." -ForegroundColor Yellow
  Invoke-WebRequest -Uri "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin" -OutFile $whisperModel
  Write-Host "[OK] Downloaded whisper model." -ForegroundColor Green
} else { Write-Host "[OK] Whisper model exists." -ForegroundColor Green }

# Step 3: Piper
$piperDir = Join-Path $workspace "piper"
if (-not (Test-Path $piperDir)) { New-Item -ItemType Directory -Path $piperDir | Out-Null }
$piperZip = Join-Path $piperDir "piper_windows_amd64.zip"
$piperUrl = "https://github.com/rhasspy/piper/releases/latest/download/piper_windows_amd64.zip"
Download-IfMissing -Url $piperUrl -Path $piperZip
Expand-IfMissing -Zip $piperZip -Dest $piperDir
$voiceModel = Join-Path $piperDir "en_US-amy-low.onnx"
$voiceCfg   = Join-Path $piperDir "en_US-amy-low.onnx.json"
Download-IfMissing -Url "https://github.com/rhasspy/piper/releases/download/2023.11.14/en_US-amy-low.onnx" -Path $voiceModel
Download-IfMissing -Url "https://github.com/rhasspy/piper/releases/download/2023.11.14/en_US-amy-low.onnx.json" -Path $voiceCfg
" AidMate is ready. " | & "$piperDir\piper.exe" -m $voiceModel -c $voiceCfg -f "$workspace\piper_tts_test.wav"
Write-Host "[OK] Piper generated audio." -ForegroundColor Green

Write-Host "[DONE] Steps 1-3 completed on port $chosenPort." -ForegroundColor Cyan
Write-Host "If you change the port later, re-run this script to repatch files." -ForegroundColor Yellow
