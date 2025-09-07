Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Resolve Ollama executable path ---
function Resolve-OllamaExe {
  $cmd = Get-Command ollama -ErrorAction SilentlyContinue
  if ($cmd) {
    return $cmd.Source
  }

  $candidates = @(
    "$env:ProgramFiles\Ollama\ollama.exe",
    "$env:ProgramFiles(x86)\Ollama\ollama.exe",
    "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
  )
  foreach ($p in $candidates) {
    if (Test-Path $p) { return $p }
  }
  return $null
}

$ollama = Resolve-OllamaExe

# --- Install Ollama if not found ---
if (-not $ollama) {
  Write-Host "Ollama not found. Installing via winget..." -ForegroundColor Yellow
  winget install --id Ollama.Ollama -e --source winget --silent
  Start-Sleep -Seconds 5
  $ollama = Resolve-OllamaExe
  if (-not $ollama) {
    throw "Ollama install not detected. Install manually from https://ollama.com/download/windows and re-run."
  }
}

Write-Host "Using Ollama at: $ollama" -ForegroundColor Green

# --- Ensure Ollama is running ---
try { Start-Service -Name "Ollama" -ErrorAction SilentlyContinue } catch {}
if (-not (Get-Process -Name "ollama" -ErrorAction SilentlyContinue)) {
  Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Minimized | Out-Null
  Start-Sleep -Seconds 3
}

# --- Settings: backing model + alias ---
$BackModel  = "phi3:mini"          # change to llama3.2:3b-instruct if needed
$ModelAlias = "gpt-oss:latest"

# --- Pull model ---
& $ollama pull $BackModel

# --- Build Modelfile ---
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

# --- Create model + alias ---
& $ollama create "gpt-oss:mini" -f $modelfilePath
try { & $ollama rm $ModelAlias | Out-Null } catch {}
& $ollama copy "gpt-oss:mini" $ModelAlias

Write-Host "Alias created: $ModelAlias -> $BackModel" -ForegroundColor Cyan

# --- Quick test ---
& $ollama run $ModelAlias -p "Say hi in one short sentence."
