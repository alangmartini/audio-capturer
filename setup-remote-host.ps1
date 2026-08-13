<#
.SYNOPSIS
  Configure this machine as the transcription host for remote clients.

.DESCRIPTION
  Clients upload recordings to exposer running here; exposer runs its
  UPLOAD_HOOK for each upload, and that hook is remote_job.py, which
  transcribes + diarizes the file and publishes progress the client polls.

  This script sets the two user-level environment variables exposer needs
  (SHARE_ROOT and UPLOAD_HOOK) and verifies the pieces are in place.  Run it
  once; restart exposer afterwards so it picks the variables up.

.PARAMETER InboxRoot
  Folder exposer shares and drops uploads into.

.PARAMETER Model
  Whisper model remote jobs use when the client does not ask for one.

.PARAMETER NoDiarize
  Disable speaker diarization for remote jobs (it is on by default here).

.EXAMPLE
  .\setup-remote-host.ps1
  .\setup-remote-host.ps1 -InboxRoot D:\MeetingInbox -Model medium
#>
param(
  [string]$InboxRoot = "$env:USERPROFILE\MeetingInbox",
  [string]$Model = "base",
  [switch]$NoDiarize
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot

Write-Host ""
Write-Host "  Configuring this machine as the transcription host" -ForegroundColor Cyan
Write-Host ""

# --- Python: prefer the repo venv, since that is where the models' deps live.
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  $python = (Get-Command python -ErrorAction SilentlyContinue).Source
  if (-not $python) { throw "No Python found. Create the venv first: python -m venv .venv" }
  Write-Host "  ! Using system Python ($python); the .venv was not found." -ForegroundColor Yellow
}

$job = Join-Path $repo "remote_job.py"
if (-not (Test-Path $job)) { throw "remote_job.py not found in $repo" }

if (-not (Test-Path $InboxRoot)) {
  New-Item -ItemType Directory -Path $InboxRoot -Force | Out-Null
  Write-Host "  + Created inbox: $InboxRoot"
}

# The hook is a shell string exposer runs per upload.  It reads the uploaded
# file from UPLOADED_FILE_PATH in the environment, so no argument substitution
# is needed and paths with spaces stay intact.
$hookArgs = "--model $Model"
if ($NoDiarize) { $hookArgs += " --no-diarize" }
$hook = "`"$python`" `"$job`" $hookArgs"

[Environment]::SetEnvironmentVariable("SHARE_ROOT", $InboxRoot, "User")
[Environment]::SetEnvironmentVariable("UPLOAD_HOOK", $hook, "User")

Write-Host "  + SHARE_ROOT  = $InboxRoot"
Write-Host "  + UPLOAD_HOOK = $hook"
Write-Host ""

# --- Checks that catch the usual silent failures -------------------------
$originKey = [Environment]::GetEnvironmentVariable("PROXY_ORIGIN_KEY", "User")
if ($originKey) {
  Write-Host "  + PROXY_ORIGIN_KEY is set (the workers.dev proxy can reach this host)."
} else {
  Write-Host "  ! PROXY_ORIGIN_KEY is not set. Clients on a VPN reach this machine" -ForegroundColor Yellow
  Write-Host "    through the exposer Worker, and server.js rejects proxied requests" -ForegroundColor Yellow
  Write-Host "    without it. See exposer\README.md." -ForegroundColor Yellow
}

Write-Host "  - Checking the transcription stack..."
& $python -c "import faster_whisper, av; print('    + faster-whisper and PyAV OK')"
if (-not $NoDiarize) {
  & $python -c "
try:
    from diarize import is_diarization_available
    print('    + diarization available' if is_diarization_available() else '    ! diarization NOT available - jobs will run without speaker labels')
except Exception as e:
    print(f'    ! diarization check failed: {e}')
"
  $hf = [Environment]::GetEnvironmentVariable("HF_TOKEN", "User")
  if (-not $hf -and -not (Test-Path (Join-Path $repo ".env"))) {
    Write-Host "    ! No HF_TOKEN found; pyannote cannot download its models." -ForegroundColor Yellow
  }
}

Write-Host ""
Write-Host "  Next:" -ForegroundColor Cyan
Write-Host "    1. Restart exposer so it sees the new variables (start-exposer.cmd)."
Write-Host "    2. On the client, point it at this host and send a recording:"
Write-Host "         python capture.py --remote-transcribe <file.wav> \"
Write-Host "           --upload-url https://exposer.<account>.workers.dev \"
Write-Host "           --upload-password <proxy password>"
Write-Host ""
Write-Host "  Per-job logs land next to each upload as <name>.joblog.jsonl."
Write-Host ""
