# System Audio Capture for Meeting Transcription

Capture audio from **any** application (Microsoft Teams, Zoom, Google Meet, Discord, browser playback, …) on Windows using WASAPI loopback, optionally mix in your microphone, and transcribe locally with **faster-whisper** — with optional speaker diarization.

```
Teams / Zoom / browser audio ─┐
                              ├──► mixed WAV ──► faster-whisper ──► .txt / .srt / .json
              microphone ─────┘                       │
                                                      └──► pyannote diarization ──► speaker-labeled transcript
```

Everything runs locally. No cloud calls, no virtual audio drivers, no admin rights required.

---

## Features

- **Loopback recording** of any app's audio via WASAPI (`pyaudiowpatch`)
- **Optional mic mixing** — record system audio + your voice into a single file
- **Local transcription** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, ~4× faster than `openai-whisper` on CPU)
- **Speaker diarization** with [pyannote.audio](https://github.com/pyannote/pyannote-audio) (optional)
- **Speaker enrollment** — name your colleagues once, get labeled transcripts forever
- **Two interfaces**: terminal CLI (`capture.py`) and web UI (`app.py`)
- **Batch processor** with watch mode (`batch_transcribe.py`)
- Outputs `.wav`, `.txt`, `.srt`, `.json` per recording, organized into per-recording subfolders

---

## Installation

```bash
pip install -r requirements.txt
```

Core requirements: `PyAudioWPATCH`, `numpy`, `faster-whisper`, `flask`. Diarization adds `pyannote.audio`.

> **Diarization (optional)** requires a free HuggingFace token and accepting the model terms:
> 1. Create a token at <https://huggingface.co/settings/tokens>
> 2. Accept the terms at <https://huggingface.co/pyannote/speaker-diarization-3.1>
> 3. Set `HF_TOKEN=<your_token>` in your environment (or a `.env` file)

`ffmpeg` must be on your `PATH` for partial-segment transcription.

---

## Quick start

### Terminal (CLI)

```bash
python capture.py
```

Launches an interactive menu:

| Key | Action |
|-----|--------|
| `r` | Start recording |
| `t` | Transcribe an existing recording |
| `n` | Rename a recording (renames `.wav` / `.txt` / `.srt` / `.json` together) |
| `l` | Listen to a transcription |
| `d` | List audio devices |
| `c` | Configure settings (model, output dir, mic, auto-transcribe) |
| `o` | Open recordings folder |
| `q` | Quit |

While recording, press `s` to stop, `q` to abort.

### Web UI

```bash
python app.py                 # http://127.0.0.1:5000
python app.py --port 8080     # custom port
python app.py --host 0.0.0.0  # expose on LAN (use with care)
```

The web UI mirrors the CLI: record, browse recordings (with subfolders), transcribe (full or segment), diarize, enroll/identify speakers, rename, delete, download, and live device-level previews.

---

## CLI flags (`capture.py`)

```bash
# Record for 60 minutes then auto-stop
python capture.py --record 60

# Record + mix microphone in (with optional volume 0.0–1.0)
python capture.py --record 60 --mic --mic-volume 0.7

# Choose specific devices
python capture.py --device 12 --mic-device 5 --record 30

# Transcribe an existing file
python capture.py --transcribe path/to/recording.wav --model medium

# Transcribe only a segment (seconds, or MM:SS via the interactive menu)
python capture.py --transcribe recording.wav --start-time 60 --end-time 300

# List available audio devices (loopback + input)
python capture.py --list-devices

# Custom name / output directory
python capture.py --record 30 --name "Sprint planning" --output-dir D:\Meetings
```

Useful flags:

| Flag | Description |
|------|-------------|
| `--record MIN` | Record for N minutes then auto-stop (omit for indefinite) |
| `--transcribe PATH` | Transcribe an existing `.wav` and exit |
| `--model NAME` | Whisper model (`tiny` / `base` / `small` / `medium` / `large-v3`) |
| `--mic` | Mix the microphone into the recording |
| `--mic-device N` | Specific mic device index |
| `--mic-volume F` | Mic gain, 0.0–1.0 (default 1.0) |
| `--allow-mic-conflict` | Don't bail when the mic might already be in use by Teams/Zoom |
| `--device N` | Specific loopback device index |
| `--name NAME` | Custom recording filename |
| `--output-dir DIR` | Override output directory |
| `--start-time / --end-time` | Trim a segment for partial transcription (seconds) |
| `--list-devices` | Print loopback + input devices |

---

## Batch transcription

Process all `.wav` files in the recordings folder that don't yet have a `.txt`:

```bash
# One-shot: transcribe everything pending
python batch_transcribe.py

# Watch mode: auto-transcribe new recordings as they appear
python batch_transcribe.py --watch

# Pick a specific model
python batch_transcribe.py --model medium
```

---

## Output layout

Each recording lives in its own subfolder for tidiness:

```
~/MeetingRecordings/
  recording_2026-05-02_10-15-00/
    recording_2026-05-02_10-15-00.wav   # raw audio
    recording_2026-05-02_10-15-00.txt   # plain text (with timestamps if diarized)
    recording_2026-05-02_10-15-00.srt   # subtitles
    recording_2026-05-02_10-15-00.json  # structured segments + language (batch only)
```

---

## Whisper models

| Model | RAM/VRAM | Speed | Accuracy | Best for |
|-------|----------|-------|----------|----------|
| `tiny` | ~1 GB | Fastest | Basic | Quick drafts |
| `base` | ~1 GB | Fast | Good | **Default** — daily meetings |
| `small` | ~2 GB | Medium | Better | Most meetings |
| `medium` | ~5 GB | Slow | Great | Important meetings |
| `large-v3` | ~10 GB | Slowest | Best | Critical / multilingual |

CPU users get a quality boost from `int16` compute type (already enabled). For GPU, install CUDA-enabled `ctranslate2` and the model loader will pick it up automatically.

---

## Speaker diarization & enrollment

When diarization is enabled, transcription and diarization run in parallel and the segments are merged into a speaker-labeled transcript with timestamps:

```
[00:00:03] SPEAKER_00: Morning everyone, let's get started.
[00:00:06] SPEAKER_01: Quick update from my side — …
```

Use the web UI to **enroll** a speaker by selecting a clip where only that person talks; future recordings will replace `SPEAKER_xx` with their name.

---

## Configuration

Settings are persisted to `~/.audio_capture_config.json`. Edit via the CLI `c` menu, the web UI Settings panel, or directly:

```json
{
  "output_dir": "C:\\Users\\you\\MeetingRecordings",
  "auto_transcribe": true,
  "whisper_model": "base",
  "device_index": null,
  "mic_enabled": false,
  "mic_device_index": null,
  "mic_volume": 1.0
}
```

---

## Tips

- **Start recording before the meeting starts** — you won't miss intros.
- **`base` is the sweet spot** on CPU. Step up to `small`/`medium` only when accuracy matters.
- **Mute notifications and music** — loopback captures *everything* the speakers play.
- **Mic mixing**: the mic is opened *before* the loopback stream for a reason — opening loopback first silently kills mic callbacks under WASAPI.
- The recordings folder uses **per-recording subfolders** so renames keep all artifacts together.

---

## Troubleshooting

**"No loopback device found"**
→ `python capture.py --list-devices` and look for entries marked `LOOPBACK`. Update audio drivers if none appear.

**Recording is silent**
→ Confirm audio is actually playing through the device you captured. Some headsets expose multiple endpoints — pick the active one.

**Mic stream fails to open**
→ The device's advertised channel count may not match WASAPI shared mode. The recorder retries at the advertised count, then stereo, then mono. If Teams/Zoom is already holding the mic exclusively, pass `--allow-mic-conflict` to bypass the safety check.

**Diarization says "model not authorized"**
→ Accept the terms on the HuggingFace pyannote page and set `HF_TOKEN`.

**Transcription is slow**
→ Use a smaller model, or move to a GPU build of `ctranslate2`.

---

## Platform notes

Built for **Windows** (WASAPI loopback). On other platforms you'll need to swap the capture backend:

- **Linux**: PulseAudio monitor sources (`alsa_output.*.monitor`) via plain `pyaudio`
- **macOS**: a virtual audio device such as [BlackHole](https://github.com/ExistentialAudio/BlackHole)

Transcription, diarization, batch processing, and the web UI are platform-agnostic.
