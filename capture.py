#!/usr/bin/env python3
"""
System Audio Capture for Meeting Transcription
================================================
Captures audio output from any application (Teams, Zoom, etc.)
via WASAPI loopback on Windows, then optionally transcribes with Whisper.

Requirements:
    pip install pyaudiowpatch numpy wave
    pip install faster-whisper  (optional, for local transcription)

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


def resolve_wav(out_dir, filename):
    """Resolve a WAV filename to its actual path.

    Checks subfolder first (out_dir/stem/filename),
    then flat (out_dir/filename) for backward compatibility.
    """
    stem = Path(filename).stem
    subfolder_path = Path(out_dir) / stem / filename
    if subfolder_path.exists():
        return subfolder_path
    return Path(out_dir) / filename


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
        "language": None,
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
        """Generate a filename inside a per-recording subfolder."""
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if name:
            safe = "".join(c for c in name if c.isalnum() or c in " -_").strip()
            stem = f"{safe}_{ts}" if safe else f"recording_{ts}"
        else:
            stem = f"recording_{ts}"
        folder = self.output_dir / stem
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{stem}.wav"

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

def transcribe_file(filepath, model_name="base", start_time=None, end_time=None):
    """Transcribe a WAV file using faster-whisper (CTranslate2)."""
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        print("\n[ERROR] faster-whisper not installed.")
        print("Install with:  pip install faster-whisper")
        return None

    filepath = Path(filepath)
    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        return None

    config = load_config()
    total_start = time.time()
    steps = []
    diarization_on = config.get("diarization_enabled", False)
    print(f"\n  ── Transcription starting ──")
    print(f"  File:         {filepath.name}")
    print(f"  Whisper model: {model_name}")
    lang_setting = config.get("language")
    print(f"  Language:     {lang_setting.upper() if lang_setting else 'auto-detect'}")
    print(f"  Diarization:  {'ON (speakers will be identified)' if diarization_on else 'OFF'}")
    if start_time is not None or end_time is not None:
        total_dur = get_wav_duration(filepath)
        actual_start = start_time if start_time is not None else 0
        actual_end = end_time if end_time is not None else total_dur
        print(f"  Range:        {format_duration(actual_start)} – {format_duration(actual_end)} (of {format_duration(total_dur)})")
    print()

    # --- Step: Load Whisper model ---
    print(f"  ⟳ Loading Whisper model '{model_name}' (downloading on first use, may take a few minutes)...")
    from whisper_loader import load_whisper_model, transcribe_audio

    def _print_load_progress(pct):
        bar_len = 30
        filled = int(bar_len * pct)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r    [{bar}] {pct*100:5.1f}% loading model weights  ", end="", flush=True)

    t = time.time()
    model = load_whisper_model(model_name, progress_callback=_print_load_progress)
    steps.append({"name": "Load Whisper model", "seconds": round(time.time() - t, 1)})
    print(f"\r  ✓ Whisper model '{model_name}' ready.{' ' * 40}")

    # --- Step: Trim audio (if partial) ---
    audio_path = filepath
    temp_path = None
    time_offset = 0
    if start_time is not None or end_time is not None:
        t = time.time()
        total_dur = get_wav_duration(filepath)
        actual_start = start_time if start_time is not None else 0
        actual_end = end_time if end_time is not None else total_dur
        print(f"  ⟳ Trimming audio to {format_duration(actual_start)} – {format_duration(actual_end)}...")
        temp_path = trim_wav_to_temp(filepath, actual_start, actual_end)
        audio_path = temp_path
        time_offset = actual_start
        steps.append({"name": "Trim audio", "seconds": round(time.time() - t, 1)})

    # Build transcribe kwargs
    transcribe_kwargs = {}
    vocab = config.get("vocabulary_terms", "")
    if vocab and vocab.strip():
        terms = vocab.strip()
        print(f"  ✓ Using custom vocabulary hints ({len(terms.split(','))} terms)")
        transcribe_kwargs["hotwords"] = terms
        prompt = f"Terms: {terms}."
        if len(prompt) > 200:
            prompt = prompt[:200]
        transcribe_kwargs["initial_prompt"] = prompt

    lang = config.get("language")
    if lang:
        transcribe_kwargs["language"] = lang

    # --- Step: Load diarization model (if needed) ---
    diarize_available = False
    if diarization_on:
        from diarize import is_diarization_available
        if not is_diarization_available():
            print("  [WARN] Speaker diarization is enabled but pyannote.audio is not installed.")
            print("         Install it with: pip install pyannote.audio")
            print("         Skipping diarization — transcription will not have speaker labels.")
        else:
            from diarize import preload_pipeline
            print(f"  ⟳ Preloading diarization model...")
            t = time.time()
            preload_pipeline(
                hf_token=config.get("hf_token"),
                status_callback=lambda msg: print(f"    ⟳ {msg}"),
            )
            steps.append({"name": "Load diarization model", "seconds": round(time.time() - t, 1)})
            diarize_available = True

    # --- Parallel execution: transcription + diarization ---
    if diarize_available:
        from concurrent.futures import ThreadPoolExecutor
        from diarize import (
            diarize_audio,
            merge_transcription_with_diarization,
            normalize_speaker_labels,
            normalize_speaker_labels_with_profiles,
            get_speaker_list,
            format_txt_with_speakers,
            format_srt_with_speakers,
            load_profiles,
        )

        print(f"  ⟳ Running transcription + diarization in parallel...")
        _print_lock = threading.Lock()

        def _whisper_progress(pct):
            elapsed = time.time() - _parallel_start
            eta = (elapsed / pct - elapsed) if pct > 0.01 else 0
            bar_len = 30
            filled = int(bar_len * pct)
            bar = '█' * filled + '░' * (bar_len - filled)
            el_m, el_s = int(elapsed // 60), int(elapsed % 60)
            eta_m, eta_s = int(eta // 60), int(eta % 60)
            with _print_lock:
                print(f"\r  [Whisper ] [{bar}] {pct*100:5.1f}% | {el_m}:{el_s:02d} elapsed | ETA: {eta_m}:{eta_s:02d}  ", end="", flush=True)

        _diarize_has_bar = [False]

        def _diarize_status(msg):
            with _print_lock:
                if _diarize_has_bar[0]:
                    print()
                    _diarize_has_bar[0] = False
                print(f"  [Diarize ] {msg}")

        def _diarize_progress(pct, step):
            with _print_lock:
                filled = int(pct / 100 * 20)
                bar = '█' * filled + '░' * (20 - filled)
                print(f"\r  [Diarize ] [{bar}] {pct:5.1f}% — {step}  ", end="", flush=True)
                _diarize_has_bar[0] = True

        def _run_whisper():
            return transcribe_audio(model, audio_path, progress_callback=_whisper_progress, **transcribe_kwargs)

        def _run_diarize():
            return diarize_audio(
                str(audio_path),
                hf_token=config.get("hf_token"),
                max_speakers=config.get("diarization_max_speakers"),
                status_callback=_diarize_status,
                progress_callback=_diarize_progress,
            )

        # --- Steps: Transcription + Diarization (parallel) ---
        _parallel_start = time.time()
        _whisper_start = time.time()
        _diarize_start = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            whisper_future = executor.submit(_run_whisper)
            diarize_future = executor.submit(_run_diarize)

            result = whisper_future.result()
            _whisper_elapsed = round(time.time() - _whisper_start, 1)
            steps.append({"name": "Transcription", "seconds": _whisper_elapsed})
            with _print_lock:
                print(f"\n  ✓ Transcription finished in {_whisper_elapsed}s")

            try:
                turns = diarize_future.result()
                _diarize_elapsed = round(time.time() - _diarize_start, 1)
                steps.append({"name": "Diarization", "seconds": _diarize_elapsed})
                with _print_lock:
                    if _diarize_has_bar[0]:
                        print()
                    print(f"  ✓ Diarization finished in {_diarize_elapsed}s")
            except Exception as e:
                with _print_lock:
                    if _diarize_has_bar[0]:
                        print()
                    print(f"  [WARN] Speaker diarization failed: {e}")
                    print(f"         Saving transcription without speaker labels.")
                turns = None

        diarized = False
        segments = result["segments"]

        if turns is not None:
            # --- Step: Merge speaker labels ---
            print(f"  ⟳ Merging speaker labels with transcription...")
            t = time.time()
            segments = merge_transcription_with_diarization(segments, turns)

            profiles = load_profiles()
            if profiles:
                print(f"  ⟳ Matching speakers against {len(profiles)} enrolled profile(s)...")
                segments, speakers, _ = normalize_speaker_labels_with_profiles(
                    segments, turns, str(audio_path),
                    hf_token=config.get("hf_token"),
                    status_callback=lambda msg: print(f"    ⟳ {msg}"),
                )
            else:
                segments = normalize_speaker_labels(segments)
                speakers = get_speaker_list(segments)
            steps.append({"name": "Merge speaker labels", "seconds": round(time.time() - t, 1)})
            diarized = True
            print(f"  ✓ Diarization complete — identified {len(speakers)} speaker(s): {', '.join(speakers)}")
    else:
        # --- Sequential: transcription only ---
        print(f"  ⟳ Transcribing {filepath.name}...")

        def _whisper_progress_seq(pct):
            elapsed = time.time() - _seq_start
            eta = (elapsed / pct - elapsed) if pct > 0.01 else 0
            bar_len = 30
            filled = int(bar_len * pct)
            bar = '█' * filled + '░' * (bar_len - filled)
            el_m, el_s = int(elapsed // 60), int(elapsed % 60)
            eta_m, eta_s = int(eta // 60), int(eta % 60)
            print(f"\r  [{bar}] {pct*100:5.1f}% | {el_m}:{el_s:02d} elapsed | ETA: {eta_m}:{eta_s:02d}  ", end="", flush=True)

        _seq_start = time.time()
        result = transcribe_audio(model, audio_path, progress_callback=_whisper_progress_seq, **transcribe_kwargs)
        steps.append({"name": "Transcription", "seconds": round(time.time() - _seq_start, 1)})
        print()  # newline after progress bar

        diarized = False
        segments = result["segments"]

    # Offset timestamps for partial transcription
    if time_offset > 0:
        for seg in segments:
            seg["start"] += time_offset
            seg["end"] += time_offset

    # Clean up temp file
    if temp_path is not None and temp_path.exists():
        temp_path.unlink()

    # --- Step: Save outputs ---
    t = time.time()
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

    # JSON (structured output with timing)
    total_seconds = round(time.time() - total_start, 1)
    json_path = filepath.with_suffix(".json")
    timing = {
        "total_seconds": total_seconds,
        "model": model_name,
        "steps": steps,
    }
    json_export = {
        "text": result["text"].strip(),
        "language": result.get("language", "unknown"),
        "timing": timing,
        "segments": [
            {
                "id": s.get("id", i),
                "start": s["start"],
                "end": s["end"],
                "text": s["text"].strip(),
                **({"speaker": s["speaker"]} if diarized else {}),
            }
            for i, s in enumerate(segments)
        ],
    }
    if diarized:
        json_export["speakers"] = [s["speaker"] for s in segments if "speaker" in s]
        seen = set()
        json_export["speakers"] = [x for x in json_export["speakers"] if not (x in seen or seen.add(x))]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2, ensure_ascii=False)
    steps.append({"name": "Save outputs", "seconds": round(time.time() - t, 1)})

    print(f"  ✓ Transcription complete in {total_seconds}s")
    print(f"  ✓ Text saved: {txt_path}")
    print(f"  ✓ SRT saved:  {srt_path}")
    print(f"  ✓ JSON saved: {json_path}")
    print(f"  ✓ Language detected: {result.get('language', 'unknown')}")

    return result


def format_srt_time(seconds):
    """Format seconds to SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def get_wav_duration(filepath):
    """Get duration of a WAV file in seconds."""
    filepath = Path(filepath)
    with wave.open(str(filepath), 'rb') as wf:
        return wf.getnframes() / wf.getframerate()


def trim_wav_to_temp(filepath, start_time, end_time):
    """Create a temporary trimmed WAV file. Returns the temp file path."""
    import tempfile
    filepath = Path(filepath)
    with wave.open(str(filepath), 'rb') as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        total_frames = wf.getnframes()
        start_frame = max(0, int(start_time * rate))
        end_frame = min(total_frames, int(end_time * rate))
        wf.setpos(start_frame)
        frames = wf.readframes(end_frame - start_frame)

    fd, tmp_name = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    tmp_path = Path(tmp_name)

    with wave.open(str(tmp_path), 'wb') as out:
        out.setnchannels(channels)
        out.setsampwidth(sampwidth)
        out.setframerate(rate)
        out.writeframes(frames)

    return tmp_path


def parse_time_input(s):
    """Parse a time string (SS, MM:SS, or HH:MM:SS) to seconds."""
    parts = s.strip().split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    return None


def rename_recording(filepath, new_name):
    """Rename a recording, its associated files, and subfolder if applicable."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"  [ERROR] File not found: {filepath}")
        return None
    safe = "".join(c for c in new_name if c.isalnum() or c in " -_").strip()
    if not safe:
        print("  [ERROR] Invalid name.")
        return None

    old_stem = filepath.stem
    old_parent = filepath.parent
    # Determine recordings root (one level up if in subfolder, same dir if flat)
    is_subfolder = old_parent.name == old_stem
    rec_root = old_parent.parent if is_subfolder else old_parent

    new_folder = rec_root / safe
    if new_folder.exists():
        print(f"  [ERROR] A recording named '{safe}' already exists.")
        return None

    renamed = []
    if is_subfolder:
        old_parent.rename(new_folder)
        for ext in (".wav", ".txt", ".srt", ".json"):
            old_file = new_folder / f"{old_stem}{ext}"
            if old_file.exists():
                new_file = new_folder / f"{safe}{ext}"
                old_file.rename(new_file)
                renamed.append((f"{old_stem}{ext}", f"{safe}{ext}"))
    else:
        new_folder.mkdir(parents=True, exist_ok=True)
        for ext in (".wav", ".txt", ".srt", ".json"):
            old_file = old_parent / f"{old_stem}{ext}"
            if old_file.exists():
                new_file = new_folder / f"{safe}{ext}"
                old_file.rename(new_file)
                renamed.append((old_file.name, new_file.name))

    if renamed:
        print(f"  Renamed {len(renamed)} file(s):")
        for old_n, new_n in renamed:
            print(f"    {old_n} -> {new_n}")
    return new_folder / f"{safe}.wav"


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
                recordings = sorted(Path(config["output_dir"]).glob("**/*.wav"))
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
                # Partial transcription option
                st, et = None, None
                try:
                    dur = get_wav_duration(target)
                    print(f"\n  Duration: {format_duration(dur)}")
                    partial = input("  Transcribe [f]ull or [s]egment? (f): ").strip().lower()
                    if partial == "s":
                        s_in = input(f"  Start time (MM:SS or seconds, default 0): ").strip()
                        if s_in:
                            st = parse_time_input(s_in)
                            if st is None:
                                print("  Invalid time format.")
                        e_in = input(f"  End time (MM:SS or seconds, default {format_duration(dur)}): ").strip()
                        if e_in:
                            et = parse_time_input(e_in)
                            if et is None:
                                print("  Invalid time format.")
                except Exception:
                    pass
                transcribe_file(target, model, start_time=st, end_time=et)

            elif choice == "n":
                # Rename a recording
                recordings = sorted(Path(config["output_dir"]).glob("**/*.wav"))
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
                transcripts = sorted(out_dir.glob("**/*.txt"))
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
                wmodel = input(f"  Whisper model (tiny/base/small/medium/large/turbo) [{config['whisper_model']}]:").strip()
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
        help="Whisper model to use (tiny/base/small/medium/large/turbo)",
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
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Start time in seconds for partial transcription",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="End time in seconds for partial transcription",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.transcribe:
        config = load_config()
        model = args.model or config.get("whisper_model", "base")
        transcribe_file(args.transcribe, model, start_time=args.start_time, end_time=args.end_time)
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
