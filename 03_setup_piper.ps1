<#
  03_setup_piper.ps1
  Piper (Windows) setup for AidMate
  Workspace: C:\SourceCode\AidMate
#>

$ErrorActionPreference = "Stop"
Write-Host "=== Step 3: Piper TTS setup ===" -ForegroundColor Cyan

# Config
$workspace = "C:\SourceCode\AidMate"
$piperDir  = Join-Path $workspace "piper"
$zipPath   = Join-Path $workspace "piper_windows_amd64.zip"

# Known-good release asset (Windows x64)
$piperZipUrl = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"

# Recommended voice (Amy, en_US, low) from rhasspy/piper-voices
$voiceOnnxUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx"
$voiceCfgUrl  = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx.json"
$voiceOnnx    = Join-Path $piperDir "en_US-amy-low.onnx"
$voiceCfg     = Join-Path $piperDir "en_US-amy-low.onnx.json"

# Ensure folders
if (-not (Test-Path $workspace)) { New-Item -ItemType Directory -Path $workspace | Out-Null }
if (-not (Test-Path $piperDir))  { New-Item -ItemType Directory -Path $piperDir  | Out-Null }

# Download Piper zip if missing/small
function Download-IfMissing {
  param([string]$Url,[string]$Path,[int]$MinBytes=1000000)
  if (-not (Test-Path $Path) -or (Get-Item $Path).Length -lt $MinBytes) {
    Write-Host "[INFO] Downloading $(Split-Path $Path -Leaf) ..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $Url -OutFile $Path
  } else {
    Write-Host "[OK] $(Split-Path $Path -Leaf) already present." -ForegroundColor Green
  }
}

Download-IfMissing -Url $piperZipUrl -Path $zipPath

# Extract to target dir
Write-Host "[INFO] Extracting Piper zip ..." -ForegroundColor Yellow
Expand-Archive -Path $zipPath -DestinationPath $piperDir -Force
Write-Host "[OK] Extracted to $piperDir" -ForegroundColor Green

# Locate piper.exe (some zips nest a folder)
$piperExe = Get-ChildItem -Path $piperDir -Recurse -Filter piper.exe -File -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $piperExe) { throw "piper.exe not found under $piperDir" }
Write-Host "[OK] Found piper.exe at $($piperExe.FullName)" -ForegroundColor Green

# Ensure required runtime DLLs are beside piper.exe (copy from same tree if needed)
# (Most releases already include onnxruntime DLLs in the same folder)
# No-op here; added for completeness.

# Download voice files if missing
if (-not (Test-Path $voiceOnnx) -or (Get-Item $voiceOnnx).Length -lt 1000000) {
  Write-Host "[INFO] Downloading voice model (en_US-amy-low.onnx) ..." -ForegroundColor Yellow
  Invoke-WebRequest -Uri $voiceOnnxUrl -OutFile $voiceOnnx
} else { Write-Host "[OK] Voice model present." -ForegroundColor Green }

if (-not (Test-Path $voiceCfg) -or (Get-Item $voiceCfg).Length -lt 1000) {
  Write-Host "[INFO] Downloading voice config (en_US-amy-low.onnx.json) ..." -ForegroundColor Yellow
  Invoke-WebRequest -Uri $voiceCfgUrl -OutFile $voiceCfg
} else { Write-Host "[OK] Voice config present." -ForegroundColor Green }

# Smoke test: synthesize a short line
$outWav = Join-Path $workspace "piper_tts_test.wav"
$null = " AidMate Piper test. " | & $piperExe.FullName -m $voiceOnnx -c $voiceCfg -f $outWav
if (-not (Test-Path $outWav)) { throw "TTS test failed: no WAV produced at $outWav" }

Write-Host "[OK] Piper TTS ready. Test file: $outWav" -ForegroundColor Green
Write-Host ""
Write-Host "Play it (PowerShell):"
Write-Host '  Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Media; (New-Object System.Media.SoundPlayer("C:\SourceCode\AidMate\piper_tts_test.wav")).PlaySync()'
