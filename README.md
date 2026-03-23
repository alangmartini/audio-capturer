# System Audio Capture for Meeting Transcription

Capture audio from **any** application (Microsoft Teams, Zoom, Google Meet, etc.) on Windows using WASAPI loopback, then transcribe it with OpenAI Whisper.

## How It Works

Windows WASAPI loopback lets you record whatever audio is playing through your speakers/headphones — no virtual audio cables or drivers needed. This tool taps into that to record meeting audio, then uses Whisper to generate transcripts.

```
Teams/Zoom/etc → Speakers (WASAPI loopback) → WAV file → Whisper → .txt / .srt
```

## Quick Start

### 1. Install dependencies

```bash
pip install pyaudiowpatch numpy openai-whisper
```

### 2. Run the recorder

```bash
python capture.py
```

This launches interactive mode. Press **r** to record, **s** to stop, **t** to transcribe.

### 3. Or use CLI flags

```bash
# Record for 60 minutes then auto-stop
python capture.py --record 60

# Transcribe an existing recording
python capture.py --transcribe ~/MeetingRecordings/recording_2026-03-20_14-00-00.wav

# Use a larger Whisper model for better accuracy
python capture.py --transcribe recording.wav --model medium

# List available audio devices
python capture.py --list-devices

# Use a specific audio device
python capture.py --device 12 --record 30
```

## Batch Transcription

Process all un-transcribed recordings at once, or watch for new files:

```bash
# Transcribe all pending recordings
python batch_transcribe.py

# Watch folder and auto-transcribe new recordings
python batch_transcribe.py --watch

# Use a better model
python batch_transcribe.py --model medium
```

## Output Files

For each recording, the transcriber generates:

| File | Contents |
|------|----------|
| `recording_*.wav` | Raw audio |
| `recording_*.txt` | Plain text transcript |
| `recording_*.srt` | Subtitle file with timestamps |
| `recording_*.json` | Structured data (segments, timestamps, language) |

Default output directory: `~/MeetingRecordings/`

## Whisper Model Sizes

| Model | VRAM | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| `tiny` | ~1 GB | Fastest | Basic | Quick drafts, testing |
| `base` | ~1 GB | Fast | Good | **Daily use (default)** |
| `small` | ~2 GB | Medium | Better | Most meetings |
| `medium` | ~5 GB | Slow | Great | Important meetings |
| `large` | ~10 GB | Slowest | Best | Critical / multilingual |

## Configuration

Settings are saved to `~/.audio_capture_config.json`. Configure via the interactive menu (press **c**) or edit directly:

```json
{
  "output_dir": "C:\\Users\\you\\MeetingRecordings",
  "auto_transcribe": true,
  "whisper_model": "base",
  "device_index": null
}
```

## Tips

- **Start recording before the meeting begins** — you won't miss anything.
- **Use `base` model** for a good speed/accuracy balance on CPU.
- **Use `medium` or `large`** if you have an NVIDIA GPU and need high accuracy.
- For **GPU acceleration**, install `faster-whisper` instead of `openai-whisper`.
- The recorder captures **all system audio**, so mute notifications / music during meetings.
- Works with **any app** that plays audio through your speakers — not limited to Teams.

## Linux / macOS

This tool is built for **Windows** (WASAPI loopback). On other platforms:

- **Linux**: Use PulseAudio monitor sources. Replace `pyaudiowpatch` with `pyaudio` and select a monitor device (e.g., `alsa_output.*.monitor`).
- **macOS**: Install [BlackHole](https://github.com/ExistentialAudio/BlackHole) as a virtual audio device, then record from it.

## Troubleshooting

**"No loopback device found"**
→ Run `python capture.py --list-devices` and check for devices marked `LOOPBACK`. If none exist, update your audio drivers.

**Recording is silent**
→ Make sure audio is actually playing through the device you're capturing. Check `--list-devices` to confirm the correct device.

**Whisper is slow**
→ Use a smaller model (`tiny` or `base`), or install `faster-whisper` with CUDA support for GPU acceleration.
