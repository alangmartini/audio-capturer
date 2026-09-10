param([int]$Port = 5000)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Project Python is missing. Set up .venv first.' }

$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
foreach ($appProcessId in ($listeners | Select-Object -ExpandProperty OwningProcess -Unique)) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $appProcessId"
    if ($process.Name -notmatch '^python(w)?\.exe$' -or $process.CommandLine -notmatch 'app\.py') {
        throw "Port $Port belongs to another application; it was not stopped."
    }
    $recording = Invoke-RestMethod "http://127.0.0.1:$Port/api/recording/status"
    $transcription = Invoke-RestMethod "http://127.0.0.1:$Port/api/transcribe/status"
    if ($recording.recording -or $transcription.active) {
        throw 'The app is recording or tracking a transcription. Wait for it to finish before restarting.'
    }
    Stop-Process -Id $appProcessId
}

$logDir = Join-Path $env:LOCALAPPDATA 'audio_capturer'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Start-Process -FilePath $python -ArgumentList @(('"' + (Join-Path $PSScriptRoot 'app.py') + '"'), '--no-reload', '--port', "$Port") -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'app.stdout.log') -RedirectStandardError (Join-Path $logDir 'app.stderr.log')
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    try {
        $null = Invoke-RestMethod "http://127.0.0.1:$Port/api/recording/status" -TimeoutSec 2
        Write-Host "Audio Capturer restarted: http://127.0.0.1:$Port (refresh the browser with Ctrl+F5)"
        exit 0
    } catch { Start-Sleep -Milliseconds 500 }
}
throw "App did not respond. Check $logDir\app.stderr.log."
