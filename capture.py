#!/usr/bin/env python3
"""
System Audio Capture for Meeting Transcription
================================================
Captures audio output from any application (Teams, Zoom, etc.)
via WASAPI loopback on Windows, then optionally transcribes with Whisper.

Requirements:
    pip install pyaudiowpatch numpy wave
    pip install openai-whisper  (optional, for local transcription)

Usage:
    python capture.py                  # Interactive mode
    python capture.py --record 60      # Record 60 minutes then stop
    python capture.py --transcribe recording.wav  # Transcribe existing file
    python capture.py --list-devices   # List available audio devices
"""

import argparse
import datetime
import json
import os
import signal
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("=" * 60)
    print("ERROR: pyaudiowpatch is required.")
    print("Install it with:  pip install pyaudiowpatch")
    print("=" * 60)
    sys.exit(1)


# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path.home() / "MeetingRecordings"
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 0.001  # RMS threshold for silence detection
CONFIG_FILE = Path.home() / ".audio_capture_config.json"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_config():
    """Load saved configuration or return defaults."""
    defaults = {
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "auto_transcribe": False,
        "whisper_model": "base",
        "file_format": "wav",
        "device_index": None,
        "diarization_enabled": False,
        "hf_token": None,
        "diarization_max_speakers": None,
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
            defaults.update(saved)
        except Exception:
            pass
    return defaults


def save_config(config):
    """Persist configuration to disk."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_loopback_device(p: pyaudio.PyAudio, device_index=None):
    """
    Find a WASAPI loopback device.
    If device_index is given, validate it. Otherwise auto-detect the default.
    """
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        print("[ERROR] WASAPI not available. This tool requires Windows with WASAPI support.")
        return None

    if device_index is not None:
        # Validate the chosen device
        try:
            device = p.get_device_info_by_index(device_index)
            if device.get("isLoopbackDevice"):
                return device
            else:
                print(f"[WARN] Device {device_index} is not a loopback device. Auto-detecting...")
        except Exception:
            print(f"[WARN] Device {device_index} not found. Auto-detecting...")

    # Auto-detect: find the default speakers' loopback
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    print(f"[INFO] Default output device: {default_speakers['name']}")

    # Search for the loopback version of the default output
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get("isLoopbackDevice") and dev["name"].startswith(default_speakers["name"]):
            return dev

    # Fallback: any loopback device
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get("isLoopbackDevice"):
            print(f"[WARN] Using fallback loopback: {dev['name']}")
            return dev

    print("[ERROR] No loopback device found.")
    return None


def list_devices():
    """Print all audio devices, highlighting loopback ones."""
    p = pyaudio.PyAudio()
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║                   AUDIO DEVICES                            ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        is_loop = dev.get("isLoopbackDevice", False)
        marker = " ◄── LOOPBACK" if is_loop else ""
        ch_in = dev["maxInputChannels"]
        ch_out = dev["maxOutputChannels"]
        rate = int(dev["defaultSampleRate"])
        name = dev["name"]
        print(f"║  [{i:3d}] {name[:42]:<42} {rate}Hz  in:{ch_in} out:{ch_out}{marker}")

    print("╚══════════════════════════════════════════════════════════════╝")
    p.terminate()


def format_duration(seconds):
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def rms(data):
    """Calculate root-mean-square of audio data."""
    if len(data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))


# ─── Recorder Class ──────────────────────────────────────────────────────────

class SystemAudioRecorder:
    """Captures system audio via WASAPI loopback and writes to WAV."""

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.p = pyaudio.PyAudio()
        self.stream = None
        self.wav_file = None
        self.is_recording = False
        self.frames_written = 0
        self.start_time = None
        self.peak_rms = 0.0
        self.current_rms = 0.0
        self._lock = threading.Lock()

    def _generate_filename(self, name=None):
        """Generate a filename. Uses custom name if provided, otherwise timestamped."""
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if name:
            # Sanitize: keep alphanumeric, spaces, hyphens, underscores
            safe = "".join(c for c in name if c.isalnum() or c in " -_").strip()
            if safe:
                return self.output_dir / f"{safe}_{ts}.wav"
        return self.output_dir / f"recording_{ts}.wav"

    def start(self, max_duration_minutes=None, name=None):
        """Begin recording system audio."""
        device = get_loopback_device(self.p, self.config.get("device_index"))
        if device is None:
            return False

        self.device = device
        self.channels = device["maxInputChannels"]
        self.sample_rate = int(device["defaultSampleRate"])
        self.filepath = self._generate_filename(name)

        print(f"\n  ● Recording from: {device['name']}")
        print(f"  ● Channels: {self.channels}  Sample rate: {self.sample_rate} Hz")
        print(f"  ● Saving to: {self.filepath}")
        if max_duration_minutes:
            print(f"  ● Auto-stop after: {max_duration_minutes} minutes")
        print()

        # Open WAV file
        self.wav_file = wave.open(str(self.filepath), "wb")
        self.wav_file.setnchannels(self.channels)
        self.wav_file.setsampwidth(2)  # 16-bit
        self.wav_file.setframerate(self.sample_rate)

        # Open audio stream
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._audio_callback,
        )

        self.is_recording = True
        self.frames_written = 0
        self.start_time = time.time()
        self.peak_rms = 0.0
        self.stream.start_stream()

        # Auto-stop timer
        if max_duration_minutes:
            self._timer = threading.Timer(
                max_duration_minutes * 60, self.stop
            )
            self._timer.daemon = True
            self._timer.start()

        return True

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Called by PyAudio for each chunk of audio."""
        if not self.is_recording:
            return (None, pyaudio.paComplete)

        with self._lock:
            self.wav_file.writeframes(in_data)
            self.frames_written += frame_count

            # Compute level for the status display
            arr = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.current_rms = rms(arr)
            self.peak_rms = max(self.peak_rms, self.current_rms)

        return (None, pyaudio.paContinue)

    def stop(self):
        """Stop recording and finalize the WAV file."""
        if not self.is_recording:
            return None

        self.is_recording = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        with self._lock:
            if self.wav_file:
                self.wav_file.close()
                self.wav_file = None

        duration = time.time() - self.start_time if self.start_time else 0
        size_mb = self.filepath.stat().st_size / (1024 * 1024)

        print(f"\n  ■ Recording stopped.")
        print(f"  ■ Duration: {format_duration(duration)}")
        print(f"  ■ File size: {size_mb:.1f} MB")
        print(f"  ■ Saved: {self.filepath}")

        return self.filepath

    def get_status(self):
        """Return current recording status dict."""
        if not self.is_recording:
            return {"recording": False}
        elapsed = time.time() - self.start_time
        return {
            "recording": True,
            "elapsed": format_duration(elapsed),
            "rms": self.current_rms,
            "peak_rms": self.peak_rms,
            "frames": self.frames_written,
            "file": str(self.filepath),
        }

    def cleanup(self):
        """Release PyAudio resources."""
        if self.is_recording:
            self.stop()
        self.p.terminate()


# ─── Recording Setup ──────────────────────────────────────────────────────────

def test_microphone(p):
    """Monitor microphone input with live level meter and playback."""
    try:
        default_input = p.get_default_input_device_info()
    except OSError:
        print("  [ERROR] No microphone found.\n")
        return

    print(f"\n  Microphone: {default_input['name']}")
    rate = int(default_input["defaultSampleRate"])
    channels = max(1, default_input["maxInputChannels"])

    in_stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=default_input["index"],
        frames_per_buffer=CHUNK_SIZE,
    )

    # Try to open output for playback
    try:
        out_stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        print("  Playback: ON (use headphones to avoid feedback)")
    except Exception:
        out_stream = None
        print("  Playback: OFF (could not open output device)")

    print("  Press any key to stop.\n")

    try:
        while not _kbhit():
            data = in_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            level = rms(arr)
            bar_len = min(int(level * 200), 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"\r  MIC [{bar}] {level:.4f}  ", end="", flush=True)
            if out_stream:
                out_stream.write(data)
    finally:
        if _kbhit():
            _getch()  # consume the keypress
        in_stream.stop_stream()
        in_stream.close()
        if out_stream:
            out_stream.stop_stream()
            out_stream.close()
    print("\n")


def preview_loopback(p, device):
    """Preview system audio from loopback device with level meter and playback."""
    if not device:
        print("  [ERROR] No loopback device available.\n")
        return

    print(f"\n  Loopback: {device['name']}")
    rate = int(device["defaultSampleRate"])
    channels = device["maxInputChannels"]

    in_stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=device["index"],
        frames_per_buffer=CHUNK_SIZE,
    )

    # Try to open output for playback
    try:
        out_stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        print("  Playback: ON (use headphones to avoid echo)")
    except Exception:
        out_stream = None
        print("  Playback: OFF (showing levels only)")

    print("  Play audio in any app to see levels. Press any key to stop.\n")

    try:
        while not _kbhit():
            data = in_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            level = rms(arr)
            bar_len = min(int(level * 200), 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"\r  AUDIO [{bar}] {level:.4f}  ", end="", flush=True)
            if out_stream:
                out_stream.write(data)
    finally:
        if _kbhit():
            _getch()  # consume the keypress
        in_stream.stop_stream()
        in_stream.close()
        if out_stream:
            out_stream.stop_stream()
            out_stream.close()
    print("\n")


def recording_setup(config, recorder):
    """Sub-menu for testing devices before starting a recording."""
    p = recorder.p
    device = get_loopback_device(p, config.get("device_index"))

    while True:
        print("\n  ┌─────────────────────────────────────┐")
        print("  │         RECORDING SETUP              │")
        print("  └─────────────────────────────────────┘")
        if device:
            print(f"  Capture device: {device['name']}")
        else:
            print("  Capture device: NOT FOUND")
        print()
        print("    [d] Show connected devices")
        print("    [m] Test microphone")
        print("    [p] Preview system audio (loopback)")
        print("    [r] Start recording")
        print("    [b] Back")
        print()

        choice = input("  > ").strip().lower()

        if choice == "d":
            list_devices()
        elif choice == "m":
            test_microphone(p)
        elif choice == "p":
            preview_loopback(p, device)
        elif choice == "r":
            return True
        elif choice == "b":
            return False


# ─── Transcription ────────────────────────────────────────────────────────────

def transcribe_file(filepath, model_name="base"):
    """Transcribe a WAV file using OpenAI Whisper (local)."""
    try:
        import whisper
    except ImportError:
        print("\n[ERROR] Whisper not installed.")
        print("Install with:  pip install openai-whisper")
        print("Or use:        pip install faster-whisper  (for GPU-accelerated)")
        return None

    filepath = Path(filepath)
    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        return None

    print(f"\n  ⟳ Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    print(f"  ⟳ Transcribing {filepath.name}...")
    start = time.time()
    result = model.transcribe(str(filepath), verbose=False)
    elapsed = time.time() - start

    # Speaker diarization (if enabled)
    config = load_config()
    diarized = False
    segments = result["segments"]

    if config.get("diarization_enabled"):
        from diarize import is_diarization_available
        if not is_diarization_available():
            print("  [WARN] Diarization enabled but pyannote.audio not installed. Skipping.")
        else:
            print(f"  ⟳ Identifying speakers...")
            try:
                from diarize import (
                    diarize_audio,
                    merge_transcription_with_diarization,
                    normalize_speaker_labels,
                    get_speaker_list,
                    format_txt_with_speakers,
                    format_srt_with_speakers,
                )
                turns = diarize_audio(
                    str(filepath),
                    hf_token=config.get("hf_token"),
                    max_speakers=config.get("diarization_max_speakers"),
                )
                segments = merge_transcription_with_diarization(segments, turns)
                segments = normalize_speaker_labels(segments)
                speakers = get_speaker_list(segments)
                diarized = True
                print(f"  ✓ Identified {len(speakers)} speaker(s): {', '.join(speakers)}")
            except Exception as e:
                print(f"  [WARN] Diarization failed: {e}. Saving without speaker labels.")

    # Save transcript
    txt_path = filepath.with_suffix(".txt")
    srt_path = filepath.with_suffix(".srt")

    # Plain text
    with open(txt_path, "w", encoding="utf-8") as f:
        if diarized:
            f.write(format_txt_with_speakers(segments))
        else:
            f.write(result["text"].strip())

    # SRT subtitles
    with open(srt_path, "w", encoding="utf-8") as f:
        if diarized:
            f.write(format_srt_with_speakers(segments, format_srt_time))
        else:
            for i, seg in enumerate(segments, 1):
                start_ts = format_srt_time(seg["start"])
                end_ts = format_srt_time(seg["end"])
                text = seg["text"].strip()
                f.write(f"{i}\n{start_ts} --> {end_ts}\n{text}\n\n")

    print(f"  ✓ Transcription complete in {elapsed:.1f}s")
    print(f"  ✓ Text saved: {txt_path}")
    print(f"  ✓ SRT saved:  {srt_path}")
    print(f"  ✓ Language detected: {result.get('language', 'unknown')}")

    return result


def format_srt_time(seconds):
    """Format seconds to SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def rename_recording(filepath, new_name):
    """Rename a recording and all associated files (.wav, .txt, .srt, .json)."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"  [ERROR] File not found: {filepath}")
        return None
    # Sanitize new name
    safe = "".join(c for c in new_name if c.isalnum() or c in " -_").strip()
    if not safe:
        print("  [ERROR] Invalid name.")
        return None
    new_stem = safe
    parent = filepath.parent
    # Check for collision
    if (parent / f"{new_stem}.wav").exists():
        print(f"  [ERROR] A recording named '{new_stem}.wav' already exists.")
        return None
    renamed = []
    for ext in (".wav", ".txt", ".srt", ".json"):
        old = filepath.with_suffix(ext)
        if old.exists():
            new_path = parent / f"{new_stem}{ext}"
            old.rename(new_path)
            renamed.append((old.name, new_path.name))
    if renamed:
        print(f"  Renamed {len(renamed)} file(s):")
        for old_name, nn in renamed:
            print(f"    {old_name} -> {nn}")
    return parent / f"{new_stem}.wav"


# ─── Interactive CLI ──────────────────────────────────────────────────────────

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓   ║
║   ┃   SYSTEM AUDIO CAPTURE                              ┃   ║
║   ┃   Meeting Transcription Recorder                    ┃   ║
║   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   ║
║                                                              ║
║   Captures audio from Teams / Zoom / any app via WASAPI     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def interactive_mode():
    """Run the interactive terminal UI."""
    print_banner()
    config = load_config()
    recorder = SystemAudioRecorder(config)

    def signal_handler(sig, frame):
        print("\n\n  Shutting down...")
        recorder.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    while True:
        status = recorder.get_status()
        if status["recording"]:
            # Show live recording status
            level = min(int(status["rms"] * 200), 30)
            bar = "█" * level + "░" * (30 - level)
            print(
                f"\r  ● REC {status['elapsed']}  [{bar}]  ",
                end="",
                flush=True,
            )
            time.sleep(0.2)
            # Check for keypress (non-blocking)
            if _kbhit():
                key = _getch().lower()
                if key == "s":
                    filepath = recorder.stop()
                    if filepath and config.get("auto_transcribe"):
                        transcribe_file(filepath, config.get("whisper_model", "base"))
                elif key == "q":
                    recorder.stop()
                    recorder.cleanup()
                    print("\n  Goodbye!\n")
                    return
        else:
            print("\n  Commands:")
            print("    [r] Start recording")
            print("    [t] Transcribe a recording")
            print("    [n] Rename a recording")
            print("    [l] Listen to a transcription")
            print("    [d] List audio devices")
            print("    [c] Configure settings")
            print("    [o] Open recordings folder")
            print("    [q] Quit")
            print()

            choice = input("  > ").strip().lower()

            if choice == "r":
                if not recording_setup(config, recorder):
                    continue
                rec_name = input("  Recording name (Enter for auto): ").strip() or None
                dur = input("  Max duration in minutes (Enter for unlimited): ").strip()
                dur = int(dur) if dur.isdigit() else None
                if recorder.start(max_duration_minutes=dur, name=rec_name):
                    print("\n  Recording! Press [s] to stop, [q] to quit.\n")
                else:
                    print("  [ERROR] Could not start recording.\n")

            elif choice == "t":
                # List available recordings
                recordings = sorted(Path(config["output_dir"]).glob("*.wav"))
                if not recordings:
                    print("  No recordings found.\n")
                    continue
                print("\n  Available recordings:")
                for i, r in enumerate(recordings[-10:], 1):
                    size = r.stat().st_size / (1024 * 1024)
                    print(f"    [{i}] {r.name}  ({size:.1f} MB)")
                idx = input("\n  Select number (or path): ").strip()
                if idx.isdigit() and 1 <= int(idx) <= len(recordings[-10:]):
                    target = recordings[-10:][int(idx) - 1]
                else:
                    target = Path(idx)
                model = input(f"  Whisper model [{config.get('whisper_model', 'base')}]: ").strip()
                model = model or config.get("whisper_model", "base")
                transcribe_file(target, model)

            elif choice == "n":
                # Rename a recording
                recordings = sorted(Path(config["output_dir"]).glob("*.wav"))
                if not recordings:
                    print("  No recordings found.\n")
                    continue
                print("\n  Available recordings:")
                for i, r in enumerate(recordings[-10:], 1):
                    size = r.stat().st_size / (1024 * 1024)
                    print(f"    [{i}] {r.name}  ({size:.1f} MB)")
                idx = input("\n  Select number: ").strip()
                if idx.isdigit() and 1 <= int(idx) <= len(recordings[-10:]):
                    target = recordings[-10:][int(idx) - 1]
                    new_name = input("  New name: ").strip()
                    if new_name:
                        rename_recording(target, new_name)
                    else:
                        print("  No name entered.\n")
                else:
                    print("  Invalid selection.\n")

            elif choice == "l":
                # List recordings that have a transcription
                out_dir = Path(config["output_dir"])
                transcripts = sorted(out_dir.glob("*.txt"))
                if not transcripts:
                    print("  No transcriptions found.\n")
                    continue
                print("\n  Available transcriptions:")
                shown = transcripts[-10:]
                for i, t in enumerate(shown, 1):
                    print(f"    [{i}] {t.stem}")
                idx = input("\n  Select number: ").strip()
                if idx.isdigit() and 1 <= int(idx) <= len(shown):
                    target = shown[int(idx) - 1]
                    with open(target, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if not text:
                        print("  Transcription is empty.\n")
                        continue
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        print(f"\n  Reading aloud: {target.name}")
                        print("  (close the console or press Ctrl+C to stop)\n")
                        engine.say(text)
                        engine.runAndWait()
                        engine.stop()
                        print("  Done.\n")
                    except ImportError:
                        print("  [ERROR] pyttsx3 not installed.")
                        print("  Install with:  pip install pyttsx3\n")
                    except Exception as e:
                        print(f"  [ERROR] TTS failed: {e}\n")
                else:
                    print("  Invalid selection.\n")

            elif choice == "d":
                list_devices()

            elif choice == "c":
                print(f"\n  Current config:")
                print(f"    Output dir:      {config['output_dir']}")
                print(f"    Auto-transcribe: {config['auto_transcribe']}")
                print(f"    Whisper model:   {config['whisper_model']}")
                print(f"    Device index:    {config.get('device_index', 'auto')}")
                print()
                new_dir = input(f"  Output directory [{config['output_dir']}]: ").strip()
                if new_dir:
                    config["output_dir"] = new_dir
                auto = input(f"  Auto-transcribe after recording? (y/n) [{config['auto_transcribe']}]: ").strip()
                if auto.lower() in ("y", "yes"):
                    config["auto_transcribe"] = True
                elif auto.lower() in ("n", "no"):
                    config["auto_transcribe"] = False
                wmodel = input(f"  Whisper model (tiny/base/small/medium/large) [{config['whisper_model']}]: ").strip()
                if wmodel:
                    config["whisper_model"] = wmodel
                dev = input(f"  Device index (number or 'auto') [{config.get('device_index', 'auto')}]: ").strip()
                if dev == "auto":
                    config["device_index"] = None
                elif dev.isdigit():
                    config["device_index"] = int(dev)
                save_config(config)
                recorder = SystemAudioRecorder(config)
                print("  ✓ Config saved.\n")

            elif choice == "o":
                out = Path(config["output_dir"])
                out.mkdir(parents=True, exist_ok=True)
                if sys.platform == "win32":
                    os.startfile(str(out))
                elif sys.platform == "darwin":
                    os.system(f'open "{out}"')
                else:
                    os.system(f'xdg-open "{out}"')

            elif choice == "q":
                recorder.cleanup()
                print("  Goodbye!\n")
                return


# ─── Cross-platform key detection ────────────────────────────────────────────

if sys.platform == "win32":
    import msvcrt

    def _kbhit():
        return msvcrt.kbhit()

    def _getch():
        return msvcrt.getch().decode("utf-8", errors="ignore")
else:
    import select
    import tty
    import termios

    def _kbhit():
        return select.select([sys.stdin], [], [], 0)[0] != []

    def _getch():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Capture system audio for meeting transcription"
    )
    parser.add_argument(
        "--record",
        type=int,
        metavar="MINUTES",
        help="Record for N minutes then stop",
    )
    parser.add_argument(
        "--transcribe",
        type=str,
        metavar="FILE",
        help="Transcribe an existing WAV file",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all audio devices",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Whisper model to use (tiny/base/small/medium/large)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save recordings",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio device index to use",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom name for the recording",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.transcribe:
        config = load_config()
        model = args.model or config.get("whisper_model", "base")
        transcribe_file(args.transcribe, model)
        return

    if args.record is not None:
        config = load_config()
        if args.output_dir:
            config["output_dir"] = args.output_dir
        if args.device is not None:
            config["device_index"] = args.device
        recorder = SystemAudioRecorder(config)

        def on_sigint(sig, frame):
            recorder.stop()
            recorder.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, on_sigint)

        print_banner()
        if recorder.start(max_duration_minutes=args.record, name=args.name):
            print("  Recording... Press Ctrl+C to stop early.\n")
            while recorder.is_recording:
                status = recorder.get_status()
                level = min(int(status["rms"] * 200), 30)
                bar = "█" * level + "░" * (30 - level)
                print(
                    f"\r  ● REC {status['elapsed']}  [{bar}]  ",
                    end="",
                    flush=True,
                )
                time.sleep(0.3)
            filepath = recorder.stop()
            if filepath and config.get("auto_transcribe"):
                model = args.model or config.get("whisper_model", "base")
                transcribe_file(filepath, model)
        recorder.cleanup()
        return

    # Default: interactive mode
    interactive_mode()


if __name__ == "__main__":
    main()
