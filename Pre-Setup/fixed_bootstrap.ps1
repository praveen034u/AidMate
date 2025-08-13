<#
  bootstrap.ps1 (fixed, ASCII-only)
  Step 0: Install prerequisites + create workspace for AidMate (Windows 11)
  Installs:
    - Git
    - Python 3.11
    - CMake
    - FFmpeg (Gyan)
  Creates workspace:
    - C:\SourceCode\AidMate  (change $workspace below if desired)
#>

$ErrorActionPreference = "Stop"

Write-Host "=== AidMate Bootstrap (Step 0) ===" -ForegroundColor Cyan

# ---------------------------
# Helpers
# ---------------------------
function Assert-Admin {
  $id  = [Security.Principal.WindowsIdentity]::GetCurrent()
  $pri = New-Object Security.Principal.WindowsPrincipal($id)
  if (-not $pri.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Please run this script in an elevated PowerShell (Run as administrator)."
  }
}

function Ensure-Winget {
  try {
    winget --version | Out-Null
  } catch {
@"
'winget' is not available.
Install 'App Installer' from Microsoft Store and/or update Windows 11, then rerun.
Store link: https://apps.microsoft.com/detail/9NBLGGH4NNS1
"@ | Write-Error
    throw
  }
}

function Get-WingetPackageInstalled {
  param([Parameter(Mandatory=$true)][string]$Id)
  $list = winget list --id $Id --source winget 2>$null
  if ($list) {
    return ($list -match [regex]::Escape($Id))
  }
  return $false
}

function Install-WithWinget {
  param(
    [Parameter(Mandatory=$true)][string]$Id,
    [Parameter(Mandatory=$true)][string]$DisplayName
  )
  if (Get-WingetPackageInstalled -Id $Id) {
    Write-Host ("Already installed: {0} ({1})" -f $DisplayName, $Id) -ForegroundColor Green
    return
  }
  Write-Host ("Installing {0} ({1})..." -f $DisplayName, $Id) -ForegroundColor Yellow
  winget install --id $Id -e --source winget --accept-package-agreements --accept-source-agreements
  Write-Host ("Installed: {0}" -f $DisplayName) -ForegroundColor Green
}

function Try-GetVersion {
  param([Parameter(Mandatory=$true)][string]$Cmd, [Parameter(Mandatory=$true)][string]$Name)
  try {
    $v = & $Cmd
    if ($LASTEXITCODE -ne 0) { throw "$Cmd returned non-zero" }
    return ($v | Select-Object -First 1)
  } catch {
    return ("{0} not on PATH yet (open a new PowerShell window if needed)." -f $Name)
  }
}

# ---------------------------
# Start
# ---------------------------
Assert-Admin
Ensure-Winget

# 1) Install tools
Install-WithWinget -Id "Git.Git"            -DisplayName "Git"
Install-WithWinget -Id "Python.Python.3.11" -DisplayName "Python 3.11"
Install-WithWinget -Id "Kitware.CMake"      -DisplayName "CMake"
Install-WithWinget -Id "Gyan.FFmpeg"        -DisplayName "FFmpeg (Gyan)"

# 2) Create workspace
$workspace = "C:\SourceCode\AidMate"
if (-not (Test-Path -LiteralPath $workspace)) {
  New-Item -ItemType Directory -Path $workspace | Out-Null
  Write-Host ("Created workspace: {0}" -f $workspace) -ForegroundColor Green
} else {
  Write-Host ("Workspace exists: {0}" -f $workspace) -ForegroundColor Green
}

# 3) Print versions / sanity check
Write-Host ""
Write-Host "=== Versions ===" -ForegroundColor Cyan
Write-Host ("Git:     {0}" -f (Try-GetVersion "git --version" "Git"))
Write-Host ("Python:  {0}" -f (Try-GetVersion "python --version" "Python"))
Write-Host ("CMake:   {0}" -f (Try-GetVersion "cmake --version" "CMake"))
Write-Host ("FFmpeg:  {0}" -f (Try-GetVersion "ffmpeg -version" "FFmpeg"))

Write-Host ""
Write-Host "Tips:" -ForegroundColor Cyan
Write-Host 'If any tool shows ''not on PATH'', close and reopen PowerShell (PATH refresh).'
Write-Host 'To list audio devices later:  ffmpeg -list_devices true -f dshow -i dummy'
Write-Host ("Your workspace is here:      {0}" -f $workspace)
Write-Host ""
Write-Host "All set for Step 1 (Ollama) and Step 2 (Whisper/Piper)." -ForegroundColor Green
