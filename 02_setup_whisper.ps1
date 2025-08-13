<#
  02_setup_whisper.ps1
  Whisper.cpp (Windows) setup for AidMate
  Workspace: C:\SourceCode\AidMate
#>

$ErrorActionPreference = "Stop"
Write-Host "=== Step 2: Whisper.cpp setup ===" -ForegroundColor Cyan

# Config
$workspace   = "C:\SourceCode\AidMate"
$zipPath     = Join-Path $workspace "whisper.zip"
$extractTmp  = Join-Path $workspace "whisper_tmp_extract"
$targetDir   = Join-Path $workspace "whisper"
$modelPath   = Join-Path $targetDir "ggml-base.en.bin"
$zipUrl      = "https://github.com/ggerganov/whisper.cpp/releases/latest/download/whisper-cpp-win-x64.zip"
$modelUrl    = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"

# Ensure directories
if (-not (Test-Path $workspace)) { New-Item -ItemType Directory -Path $workspace | Out-Null }
if (-not (Test-Path $targetDir)) { New-Item -ItemType Directory -Path $targetDir | Out-Null }

# Download zip if missing
if (-not (Test-Path $zipPath) -or (Get-Item $zipPath).Length -lt 1000000) {
  Write-Host "[INFO] Downloading Whisper Windows zip..." -ForegroundColor Yellow
  Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath
} else {
  Write-Host "[OK] Existing whisper.zip looks valid." -ForegroundColor Green
}

# Clean extract temp
if (Test-Path $extractTmp) { Remove-Item $extractTmp -Recurse -Force }
New-Item -ItemType Directory -Path $extractTmp | Out-Null

# Extract
Expand-Archive -Path $zipPath -DestinationPath $extractTmp -Force

# Find exe dir (try common layouts)
$exeCandidates = @(
  (Join-Path $extractTmp "Release"),
  (Join-Path $extractTmp "bin"),
  $extractTmp
) | Where-Object { Test-Path $_ }

# Find main/whisper exe anywhere under extracted tree
$exe = Get-ChildItem -Path $exeCandidates -Recurse -Include main.exe, whisper.exe -File -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $exe) {
  throw "Could not find main.exe or whisper.exe in extracted archive."
}
$exeDir = $exe.DirectoryName
Write-Host "[OK] Found Whisper CLI: $($exe.FullName)" -ForegroundColor Green

# Copy exe and dlls to target using -Include (NOT -Filter)
$toCopy = Get-ChildItem -Path $exeDir -Recurse -Include *.exe, *.dll -File -ErrorAction SilentlyContinue
if (-not $toCopy) { throw "No .exe/.dll files found to copy from $exeDir" }

foreach ($f in $toCopy) {
  Copy-Item -Path $f.FullName -Destination (Join-Path $targetDir $f.Name) -Force
}

# Normalize name: prefer whisper.exe if present, else copy main.exe to whisper.exe
$whisperExe = Join-Path $targetDir "whisper.exe"
if (-not (Test-Path $whisperExe)) {
  $mainExe = Join-Path $targetDir "main.exe"
  if (Test-Path $mainExe) {
    Copy-Item $mainExe $whisperExe -Force
  } else {
    # If only one exe exists, use that as whisper.exe
    $anyExe = Get-ChildItem -Path $targetDir -Filter *.exe -File | Select-Object -First 1
    if ($anyExe) { Copy-Item $anyExe.FullName $whisperExe -Force }
  }
}

if (-not (Test-Path $whisperExe)) { throw "whisper.exe not present in $targetDir after copy." }

# Download model if missing
if (-not (Test-Path $modelPath) -or (Get-Item $modelPath).Length -lt 1000000) {
  Write-Host "[INFO] Downloading Whisper model (ggml-base.en.bin)..." -ForegroundColor Yellow
  Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath
} else {
  Write-Host "[OK] Model already present." -ForegroundColor Green
}

# Cleanup extract temp
if (Test-Path $extractTmp) { Remove-Item $extractTmp -Recurse -Force }

Write-Host "[OK] Whisper setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Quick test (record 5s and transcribe):"
Write-Host "  1) List devices:  ffmpeg -list_devices true -f dshow -i dummy"
Write-Host '  2) Record 5s:     $mic = "Microphone (Realtek(R) Audio)"; ffmpeg -y -f dshow -i audio="$mic" -t 5 mic.wav'
Write-Host "  3) Transcribe:    .\whisper\whisper.exe -m .\whisper\ggml-base.en.bin -f .\mic.wav -l en -otxt"
Write-Host "  4) View text:     Get-Content .\mic.wav.txt"
