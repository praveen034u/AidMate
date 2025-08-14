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
}