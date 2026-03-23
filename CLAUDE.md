# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Windows system audio capture tool for meeting transcription. Records audio from any application (Teams, Zoom, etc.) via WASAPI loopback, then transcribes locally using OpenAI Whisper.

**Platform:** Windows only (WASAPI loopback). No admin permissions available on this machine.

## Commands

```bash
# Install dependencies
pip install pyaudiowpatch numpy openai-whisper

# Run interactive mode
python capture.py

# Record with auto-stop (minutes)
python capture.py --record 60

# Transcribe an existing file
python capture.py --transcribe path/to/file.wav --model base

# List audio devices
python capture.py --list-devices

# Batch transcribe all pending recordings
python batch_transcribe.py

# Watch folder and auto-transcribe new files
python batch_transcribe.py --watch
```

No test suite, linter, or build system exists — this is a standalone script-based tool.

## Architecture

Two entry points, no shared module between them:

- **`capture.py`** — Main tool. Contains `SystemAudioRecorder` class (WASAPI loopback capture via `pyaudiowpatch`), `transcribe_file()` function, and an interactive CLI with live RMS level display. Supports both CLI flags (`--record`, `--transcribe`, `--list-devices`) and interactive mode (keyboard-driven menu).
- **`batch_transcribe.py`** — Standalone batch processor. Finds WAV files without a matching `.txt` and transcribes them. Has a `--watch` mode that polls for new files. Imports `whisper` at module level (unlike `capture.py` which defers the import).

## Development Rules

- **Frontend parity:** Every feature implemented in the CLI (`capture.py`) must also be implemented in the web frontend (`app.py` + `templates/index.html`). The two interfaces should offer equivalent functionality.

Key details:
- Audio callback runs on a separate thread (`_audio_callback`), protected by `threading.Lock`
- Config persisted at `~/.audio_capture_config.json`; recordings default to `~/MeetingRecordings/`
- `format_srt_time()` is duplicated in both files
- Transcription outputs: `.txt` (plain text), `.srt` (subtitles), `.json` (structured — batch only; capture.py only writes txt/srt)
- Non-blocking key detection uses `msvcrt` on Windows, `termios`/`select` on Unix
