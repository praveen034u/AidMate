param(
  [string]$MicName = "",
  [string]$Lang = "en",
  [string]$LLMModel = "llama3.1:8b",
  [int]$RecordSeconds = 5
)

$ErrorActionPreference = "Stop"
Set-Location "C:\SourceCode\AidMate"

if ([string]::IsNullOrWhiteSpace($MicName)) {
  Write-Host "Listing audio devices..." -ForegroundColor Yellow
  ffmpeg -list_devices true -f dshow -i dummy 2>&1 | Select-String "DirectShow audio devices" -Context 0,100
  Write-Host ""
  $MicName = Read-Host "Enter EXACT microphone device name"
}

# 1) Record
ffmpeg -y -f dshow -i audio="$MicName" -t $RecordSeconds mic.wav | Out-Null

# 2) Transcribe (auto-detect whisper.exe or main.exe)
$wdir = "C:\SourceCode\AidMate\whisper"
$wexe = @(
  (Join-Path -Path $wdir -ChildPath "whisper.exe"),
  (Join-Path -Path $wdir -ChildPath "main.exe"),
  (Join-Path -Path $wdir -ChildPath "bin\whisper.exe"),
  (Join-Path -Path $wdir -ChildPath "bin\main.exe")
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $wexe) { throw "Whisper CLI not found in $wdir" }

& $wexe -m "C:\SourceCode\AidMate\whisper\ggml-base.en.bin" -f .\mic.wav -l $Lang -otxt | Out-Null
$q = (Get-Content .\mic.wav.txt) -join "`n"
Write-Host "You said:`n$q`n" -ForegroundColor Green
if ([string]::IsNullOrWhiteSpace($q)) { Write-Warning "No text."; exit 1 }

# 3) LLM (uses OLLAMA_HOST if set)
$port = 11434
if ($env:OLLAMA_HOST -and ($env:OLLAMA_HOST -match ":(\d+)$")) { $port = [int]$Matches[1] }

$system = "You are AidMate, an offline first-aid & crisis assistant.`n- Be calm, concise, and step-by-step.`n- Add a one-line disclaimer at the end: `"Not a medical professional.`""
$prompt = "$system`n`nUser question:`n$q`n`nRespond with:`n- Bullet points`n- Short sentences`n- Numbered steps when actionable"

$body = @{ model = $LLMModel; prompt = $prompt; stream = $false } | ConvertTo-Json -Depth 6
$res = Invoke-RestMethod -Uri ("http://localhost:" + $port + "/api/generate") -Method POST -ContentType "application/json" -Body $body
$answer = $res.response
"`nAidMate:`n$answer`n" | Tee-Object -FilePath .\aidmate_answer.txt | Out-Null

# 4) TTS
$bytes = [System.Text.Encoding]::UTF8.GetBytes($answer)
$outWav = "aidmate_out.wav"
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "C:\SourceCode\AidMate\piper\piper.exe"
$psi.Arguments = "-m `"C:\SourceCode\AidMate\piper\en_US-amy-low.onnx`" -c `"C:\SourceCode\AidMate\piper\en_US-amy-low.onnx.json`" -f `"$outWav`""
$psi.RedirectStandardInput = $true
$psi.UseShellExecute = $false
$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi
$p.Start() | Out-Null
$p.StandardInput.BaseStream.Write($bytes, 0, $bytes.Length)
$p.StandardInput.Close()
$p.WaitForExit()

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Media
(New-Object System.Media.SoundPlayer("C:\SourceCode\AidMate\$outWav")).PlaySync()
