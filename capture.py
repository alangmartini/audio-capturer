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
import queue
import re
import signal
import sys
import threading
import time
import traceback
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
        "mic_enabled": False,
        "mic_device_index": None,
        "mic_volume": 1.0,
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
    default_speakers = None
    try:
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        print(f"[INFO] Default output device: {default_speakers['name']}")
    except Exception as e:
        print(f"[WARN] Could not query default output device (index={wasapi_info.get('defaultOutputDevice')}): {e}")

    # Enumerate once — used for both matching and diagnostics.
    n_devices = p.get_device_count()
    loopbacks = []
    for i in range(n_devices):
        try:
            dev = p.get_device_info_by_index(i)
        except Exception:
            continue
        if dev.get("isLoopbackDevice"):
            loopbacks.append(dev)

    # Prefer loopback matching the default output device
    if default_speakers:
        for dev in loopbacks:
            if dev["name"].startswith(default_speakers["name"]):
                return dev

    # Fallback: any loopback device
    if loopbacks:
        print(f"[WARN] Using fallback loopback: {loopbacks[0]['name']}")
        return loopbacks[0]

    # Diagnostics: dump what we actually saw so the user (or us) can tell whether
    # pyaudiowpatch is enumerating loopback devices at all. If this list shows
    # only regular input/output devices with no "[Loopback]" suffix, the
    # pyaudiowpatch install is broken or has been shadowed by vanilla pyaudio.
    print(f"[ERROR] No loopback device found among {n_devices} enumerated devices.")
    print(f"[DIAG]  pyaudio module: {pyaudio.__name__}  paWASAPI={pyaudio.paWASAPI}")
    for i in range(n_devices):
        try:
            dev = p.get_device_info_by_index(i)
        except Exception:
            continue
        print(f"[DIAG]   [{i:3d}] {dev['name']}  in={dev['maxInputChannels']} "
              f"out={dev['maxOutputChannels']} hostApi={dev.get('hostApi')} "
              f"loopback={bool(dev.get('isLoopbackDevice'))}")
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


def resample_linear(data_int16, from_rate, to_rate):
    """Resample int16 audio using linear interpolation (numpy only)."""
    if from_rate == to_rate:
        return data_int16
    ratio = to_rate / from_rate
    in_len = len(data_int16)
    out_len = int(in_len * ratio)
    if out_len == 0:
        return np.array([], dtype=np.int16)
    in_times = np.arange(in_len, dtype=np.float64) / from_rate
    out_times = np.arange(out_len, dtype=np.float64) / to_rate
    return np.interp(out_times, in_times, data_int16.astype(np.float64)).astype(np.int16)


def mono_to_stereo(data_int16):
    """Duplicate mono samples to stereo (interleaved L/R)."""
    return np.column_stack((data_int16, data_int16)).flatten()


def get_mic_device(p, mic_device_index=None):
    """Find a microphone (non-loopback input) device."""
    if mic_device_index is not None:
        try:
            dev = p.get_device_info_by_index(mic_device_index)
            if dev["maxInputChannels"] > 0 and not dev.get("isLoopbackDevice", False):
                return dev
            else:
                print(f"[WARN] Device {mic_device_index} is not a valid microphone. Using default...")
        except Exception:
            print(f"[WARN] Mic device {mic_device_index} not found. Using default...")

    try:
        return p.get_default_input_device_info()
    except OSError:
        print("[WARN] No default microphone found.")
        return None


def _extract_device_identity(name):
    """Extract a distinctive hardware identifier from a Windows audio device name.

    Windows typically names devices like "Speakers (Brand Model)" / "Microphone (Brand Model)".
    The parenthesized part identifies the physical device. WASAPI loopback entries may
    append a suffix like " [Loopback]" which we strip. Some drivers produce nested parens
    (e.g. "Microphone Array (Realtek(R) Audio)"), so we take the span between the first '('
    and the last ')' to get the full outermost identifier rather than an inner fragment.
    """
    if not name:
        return ""
    s = re.sub(r"\s*\[[^\]]*\]\s*$", "", name).strip()
    first = s.find("(")
    last = s.rfind(")")
    if first != -1 and last > first:
        return s[first + 1:last].strip().lower()
    return s.strip().lower()


def detect_device_conflict(loopback_device, mic_device):
    """Return True if the loopback output and mic input share a physical device.

    Typical case: a Bluetooth or USB headset exposes both a speaker and a mic with the
    same hardware name. Opening the mic forces Windows to switch Bluetooth profile from
    A2DP (stereo playback) to HFP (mono call mode), which degrades/kills playback and
    invalidates the WASAPI loopback stream already capturing it.
    """
    if not loopback_device or not mic_device:
        return False
    lb_id = _extract_device_identity(loopback_device.get("name", ""))
    mic_id = _extract_device_identity(mic_device.get("name", ""))
    if not lb_id or not mic_id:
        return False
    if lb_id == mic_id:
        return True
    # Substring match with a minimum length to avoid matching generic words like "audio"
    if len(lb_id) >= 6 and (lb_id in mic_id or mic_id in lb_id):
        return True
    return False


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

        # Microphone mixing
        self.mic_stream = None
        self.mic_enabled = bool(config.get("mic_enabled", False))
        self.mic_device_index = config.get("mic_device_index")
        self.mic_volume = float(config.get("mic_volume", 1.0))
        self._loopback_queue = None
        self._mic_queue = None
        self._mixer_thread = None
        self._mixer_stop = threading.Event()
        self.current_rms_mic = 0.0
        self.mic_sample_rate = None
        self.mic_channels = None

        # Structured error from the last failed start() — consumed by the web API
        # to surface actionable messages (e.g. Bluetooth headset conflict).
        self.last_error = None

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

    def _reinit_pyaudio(self):
        """Recreate the PyAudio instance.

        Windows WASAPI can refuse to reopen a loopback device on a PyAudio
        instance that already used it, returning "Unanticipated host error"
        (-9999). A fresh instance sidesteps that by forcing PortAudio to
        re-initialize COM and re-enumerate the host API.
        """
        if self.p is not None:
            try:
                self.p.terminate()
            except Exception:
                pass
        self.p = pyaudio.PyAudio()

    def start(self, max_duration_minutes=None, name=None, allow_mic_conflict=False):
        """Begin recording system audio (optionally mixed with microphone).

        allow_mic_conflict: when True, skip the Bluetooth-headset safety check
        (caller has acknowledged that opening the mic will likely degrade playback).
        """
        self.last_error = None

        # Fresh PyAudio per start() prevents Windows from refusing a reopen with
        # -9999 after a prior recording on the same instance.
        self._reinit_pyaudio()

        device = get_loopback_device(self.p, self.config.get("device_index"))
        if device is None:
            self.last_error = {
                "code": "no_loopback",
                "message": (
                    "No WASAPI loopback device found. This usually means pyaudiowpatch "
                    "isn't loading loopback support correctly — try fully restarting the "
                    "app (not just auto-reload). Check the terminal for the [DIAG] lines "
                    "showing which devices were enumerated."
                ),
            }
            return False

        self.device = device
        self.channels = device["maxInputChannels"]
        self.sample_rate = int(device["defaultSampleRate"])

        # Resolve mic device up front so we can detect conflicts BEFORE touching
        # any file or audio-stream resources (nothing to clean up if we abort here).
        mic_device = None
        if self.mic_enabled:
            mic_device = get_mic_device(self.p, self.mic_device_index)
            if mic_device is None:
                print("  ⚠ Microphone not found — recording system audio only")
            elif not allow_mic_conflict and detect_device_conflict(device, mic_device):
                self.last_error = {
                    "code": "mic_device_conflict",
                    "message": (
                        f"The selected microphone and the playback device appear to be the same "
                        f"physical hardware (likely a Bluetooth or USB headset). Opening the mic "
                        f"will force Windows to switch the headset into call mode, killing audio "
                        f"playback and the loopback capture. Pick a different microphone (built-in "
                        f"laptop mic, USB mic) or confirm to proceed anyway."
                    ),
                    "loopback": device["name"],
                    "mic": mic_device["name"],
                }
                print(f"  ✖ Aborting: mic/playback device conflict — {mic_device['name']}")
                return False

        self.filepath = self._generate_filename(name)

        print(f"\n  ● Recording from: {device['name']}")
        print(f"  ● Channels: {self.channels}  Sample rate: {self.sample_rate} Hz")
        print(f"  ● Saving to: {self.filepath}")
        if max_duration_minutes:
            print(f"  ● Auto-stop after: {max_duration_minutes} minutes")

        # Open WAV file
        self.wav_file = wave.open(str(self.filepath), "wb")
        self.wav_file.setnchannels(self.channels)
        self.wav_file.setsampwidth(2)  # 16-bit
        self.wav_file.setframerate(self.sample_rate)

        if mic_device:
            # Dual-stream mode: loopback + mic via mixer thread
            self.mic_sample_rate = int(mic_device["defaultSampleRate"])
            max_mic_ch = int(mic_device["maxInputChannels"])
            print(f"  ● Microphone: {mic_device['name']}")

            self._loopback_queue = queue.Queue(maxsize=200)
            self._mic_queue = queue.Queue(maxsize=200)
            self._mixer_stop.clear()

            # Open loopback stream (queued callback)
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device["index"],
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._loopback_callback_queued,
            )

            # Open mic stream. `maxInputChannels` is what the device advertises
            # but WASAPI shared mode refuses anything that doesn't match the
            # device's current mix format — array mics that report 4/8 channels
            # typically only open at 1 or 2. Try the advertised count first,
            # then fall back to stereo and mono before giving up.
            candidates = []
            for ch in (max_mic_ch, 2, 1):
                if ch >= 1 and ch not in candidates:
                    candidates.append(ch)

            mic_last_error = None
            self.mic_channels = None
            for ch in candidates:
                try:
                    self.mic_stream = self.p.open(
                        format=pyaudio.paInt16,
                        channels=ch,
                        rate=self.mic_sample_rate,
                        input=True,
                        input_device_index=mic_device["index"],
                        frames_per_buffer=CHUNK_SIZE,
                        stream_callback=self._mic_callback,
                    )
                    self.mic_channels = ch
                    if ch != max_mic_ch:
                        print(f"  ⚠ Mic opened at {ch} ch (device reported {max_mic_ch} — driver refused)")
                    break
                except OSError as e:
                    mic_last_error = e
                    continue

            if self.mic_channels is None:
                self._abort_start_cleanup()
                self.last_error = {
                    "code": "mic_open_failed",
                    "message": (
                        f"Could not open microphone '{mic_device['name']}' at any of "
                        f"{candidates} channels: {mic_last_error}. This often happens with "
                        f"Bluetooth headsets when the device is already in use by another app, "
                        f"or when Windows refuses to switch audio profiles."
                    ),
                    "mic": mic_device["name"],
                }
                print(f"  ✖ Mic stream failed to open: {mic_last_error}")
                return False

            print(f"  ● Mic rate: {self.mic_sample_rate} Hz  channels: {self.mic_channels}  volume: {self.mic_volume:.0%}")
        else:
            # Single-stream mode: loopback only (original path)
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device["index"],
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_callback,
            )

        print()

        self.is_recording = True
        self.frames_written = 0
        self.start_time = time.time()
        self.peak_rms = 0.0
        self.current_rms_mic = 0.0
        self._loopback_callback_seen = False
        self._mic_callback_seen = False
        self.stream.start_stream()

        if mic_device and self.mic_stream:
            self.mic_stream.start_stream()
            self._mixer_thread = threading.Thread(target=self._mixer_loop, daemon=True)
            self._mixer_thread.start()

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

    def _loopback_callback_queued(self, in_data, frame_count, time_info, status):
        """Loopback callback that pushes to queue (used when mic mixing is active)."""
        if not self.is_recording:
            return (None, pyaudio.paComplete)

        if not self._loopback_callback_seen:
            self._loopback_callback_seen = True
            print(f"[LOOPBACK] first callback: {len(in_data)} bytes / {frame_count} frames  status={status}")

        arr = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.current_rms = rms(arr)
        self.peak_rms = max(self.peak_rms, self.current_rms)

        try:
            self._loopback_queue.put_nowait(in_data)
        except queue.Full:
            pass  # Drop frame rather than block audio thread

        return (None, pyaudio.paContinue)

    def _mic_callback(self, in_data, frame_count, time_info, status):
        """Microphone callback that pushes to queue."""
        if not self.is_recording:
            return (None, pyaudio.paComplete)

        if not self._mic_callback_seen:
            self._mic_callback_seen = True
            print(f"[MIC] first callback: {len(in_data)} bytes / {frame_count} frames  status={status}")

        arr = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.current_rms_mic = rms(arr)

        try:
            self._mic_queue.put_nowait(in_data)
        except queue.Full:
            pass  # Drop frame rather than block audio thread

        return (None, pyaudio.paContinue)

    def _mixer_loop(self):
        """Mixer thread, mic-driven.

        Mic is the time reference because WASAPI loopback only delivers callbacks
        while the OS audio engine is actively rendering — during silence (no app
        playing audio) loopback fires nothing at all, not even zeros. If we used
        loopback as the clock, mic data would be discarded during those gaps.
        Mic-driven means: every mic chunk → write a chunk to disk, mixing in
        whatever loopback we have buffered (and silence if we have none).
        """
        print(f"[MIXER] thread started (lb={self.sample_rate}Hz/{self.channels}ch  "
              f"mic={self.mic_sample_rate}Hz/{self.mic_channels}ch)  mic-driven")
        lb_buffer = bytearray()
        bytes_per_lb_frame = 2 * self.channels
        first_write_logged = False
        first_lb_logged = False
        empty_mic_iters = 0
        chunks_with_lb = 0
        chunks_silent = 0

        try:
            while not self._mixer_stop.is_set():
                # Get next mic chunk (mic always fires steady callbacks)
                try:
                    mic_data = self._mic_queue.get(timeout=0.1)
                except queue.Empty:
                    empty_mic_iters += 1
                    if empty_mic_iters in (10, 50, 100):
                        print(f"[MIXER] WARNING: no mic data after ~{empty_mic_iters/10:.0f}s — "
                              f"mic callback may not be firing.")
                    continue
                empty_mic_iters = 0

                # Drain all available loopback data into rolling buffer
                while True:
                    try:
                        lb_chunk = self._loopback_queue.get_nowait()
                        lb_buffer.extend(lb_chunk)
                        if not first_lb_logged:
                            print("[MIXER] loopback queue began delivering")
                            first_lb_logged = True
                    except queue.Empty:
                        break

                # Process mic chunk: bytes → int16 array
                mic_arr = np.frombuffer(mic_data, dtype=np.int16)

                # Mic channel conversion to match output (loopback) channels
                if self.mic_channels == 1 and self.channels == 2:
                    mic_arr = mono_to_stereo(mic_arr)
                elif self.mic_channels == 2 and self.channels == 1:
                    mic_arr = mic_arr[::2]  # take left channel only
                elif self.mic_channels != self.channels:
                    # Multi-channel mic → take first channel, then upmix if needed
                    mic_arr = mic_arr[::self.mic_channels]
                    if self.channels == 2:
                        mic_arr = mono_to_stereo(mic_arr)

                # Resample mic to loopback rate (output WAV is at loopback rate)
                if self.mic_sample_rate != self.sample_rate:
                    if self.channels == 2:
                        left = resample_linear(mic_arr[0::2], self.mic_sample_rate, self.sample_rate)
                        right = resample_linear(mic_arr[1::2], self.mic_sample_rate, self.sample_rate)
                        mic_arr = np.column_stack((left, right)).flatten()
                    else:
                        mic_arr = resample_linear(mic_arr, self.mic_sample_rate, self.sample_rate)

                # Now mic_arr is in loopback-rate, loopback-channel format.
                # Pull the same number of bytes from the loopback buffer, or pad
                # with silence if loopback hasn't kept up (or isn't firing at all).
                needed_lb_bytes = mic_arr.nbytes
                if len(lb_buffer) >= needed_lb_bytes:
                    lb_raw = bytes(lb_buffer[:needed_lb_bytes])
                    del lb_buffer[:needed_lb_bytes]
                    chunks_with_lb += 1
                else:
                    # Use whatever loopback we have; zero-pad the rest
                    lb_raw = bytes(lb_buffer) + b'\x00' * (needed_lb_bytes - len(lb_buffer))
                    lb_buffer.clear()
                    chunks_silent += 1

                lb_arr = np.frombuffer(lb_raw, dtype=np.int16)

                # Mix in float space, clip, back to int16
                mic_f = mic_arr.astype(np.float32) * self.mic_volume
                lb_f = lb_arr.astype(np.float32)
                mixed = np.clip(lb_f + mic_f, -32768, 32767).astype(np.int16)

                with self._lock:
                    if self.wav_file:
                        self.wav_file.writeframes(mixed.tobytes())
                        self.frames_written += len(mixed) // self.channels
                        if not first_write_logged:
                            print(f"[MIXER] first frame written ({len(mixed) // self.channels} frames)")
                            first_write_logged = True

                # Don't let lb_buffer grow without bound if loopback fires faster
                # than mic (rare, but possible). Cap at ~2s of audio.
                max_lb_buffer = bytes_per_lb_frame * self.sample_rate * 2
                if len(lb_buffer) > max_lb_buffer:
                    overflow = len(lb_buffer) - max_lb_buffer
                    del lb_buffer[:overflow]
        except Exception as exc:
            print(f"[MIXER] FATAL exception — thread is dying:\n{traceback.format_exc()}")
            self.last_error = {
                "code": "mixer_crashed",
                "message": f"Mixer thread crashed: {exc}",
            }
            return

        print(f"[MIXER] thread exiting (frames_written={self.frames_written}, "
              f"chunks_with_loopback={chunks_with_lb}, chunks_silent={chunks_silent})")

    def _abort_start_cleanup(self):
        """Undo partial state from a failed start() call.

        Closes any already-opened loopback stream and wav file, then removes the
        empty wav file and (if empty) the recording subfolder so a failed attempt
        doesn't leave orphan files behind.
        """
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.wav_file is not None:
            try:
                self.wav_file.close()
            except Exception:
                pass
            self.wav_file = None

        try:
            fp = getattr(self, "filepath", None)
            if fp and fp.exists():
                fp.unlink()
                # Remove the per-recording subfolder if it's now empty
                parent = fp.parent
                if parent != self.output_dir and parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
        except Exception:
            pass

    def stop(self):
        """Stop recording and finalize the WAV file."""
        if not self.is_recording:
            return None

        self.is_recording = False

        # Stop mixer thread first (so it drains remaining data)
        if self._mixer_thread and self._mixer_thread.is_alive():
            self._mixer_stop.set()
            self._mixer_thread.join(timeout=2)
            self._mixer_thread = None

        # Stop mic stream
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()
            self.mic_stream = None

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
            "mic_enabled": self.mic_enabled and self.mic_stream is not None,
            "rms_mic": self.current_rms_mic,
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

    # Hotwords: only a small set of high-value terms that bias decoding.
    # Large hotword lists cause severe hallucination (the model "hears" them
    # even on clear speech).  Keep this tiny.
    hotwords = config.get("hotwords", "Task")
    if hotwords and hotwords.strip():
        transcribe_kwargs["hotwords"] = hotwords.strip()

    # Vocabulary terms go into initial_prompt only — gentle contextual
    # conditioning, not aggressive token biasing like hotwords.
    vocab = config.get("vocabulary_terms", "")
    if vocab and vocab.strip():
        terms = vocab.strip()
        term_count = len(terms.split(","))
        print(f"  ✓ Using vocabulary context ({term_count} terms)")
        prompt = f"Technical meeting transcript. Terms: {terms}."
        if len(prompt) > 500:
            prompt = prompt[:500]
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
            if status.get("mic_enabled"):
                lb_level = min(int(status["rms"] * 200), 15)
                mic_level = min(int(status.get("rms_mic", 0) * 200), 15)
                lb_bar = "█" * lb_level + "░" * (15 - lb_level)
                mic_bar = "█" * mic_level + "░" * (15 - mic_level)
                print(
                    f"\r  ● REC {status['elapsed']}  SYS[{lb_bar}] MIC[{mic_bar}]  ",
                    end="", flush=True,
                )
            else:
                level = min(int(status["rms"] * 200), 30)
                bar = "█" * level + "░" * (30 - level)
                print(
                    f"\r  ● REC {status['elapsed']}  [{bar}]  ",
                    end="", flush=True,
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
                mic_status = "on" if config.get("mic_enabled") else "off"
                print(f"\n  Current config:")
                print(f"    Output dir:      {config['output_dir']}")
                print(f"    Auto-transcribe: {config['auto_transcribe']}")
                print(f"    Whisper model:   {config['whisper_model']}")
                print(f"    Device index:    {config.get('device_index', 'auto')}")
                print(f"    Mic capture:     {mic_status}")
                print(f"    Mic device:      {config.get('mic_device_index', 'auto')}")
                print(f"    Mic volume:      {config.get('mic_volume', 1.0):.0%}")
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
                mic = input(f"  Mic capture (on/off) [{mic_status}]: ").strip().lower()
                if mic in ("on", "yes", "y"):
                    config["mic_enabled"] = True
                elif mic in ("off", "no", "n"):
                    config["mic_enabled"] = False
                mic_dev = input(f"  Mic device index (number or 'auto') [{config.get('mic_device_index', 'auto')}]: ").strip()
                if mic_dev == "auto":
                    config["mic_device_index"] = None
                elif mic_dev.isdigit():
                    config["mic_device_index"] = int(mic_dev)
                mic_vol = input(f"  Mic volume 0.0-2.0 [{config.get('mic_volume', 1.0)}]: ").strip()
                if mic_vol:
                    try:
                        vol = float(mic_vol)
                        config["mic_volume"] = max(0.0, min(2.0, vol))
                    except ValueError:
                        pass
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
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Enable microphone capture (mixed with system audio)",
    )
    parser.add_argument(
        "--mic-device",
        type=int,
        default=None,
        help="Microphone device index (default: system default)",
    )
    parser.add_argument(
        "--mic-volume",
        type=float,
        default=None,
        help="Microphone volume multiplier 0.0-2.0 (default: 1.0)",
    )
    parser.add_argument(
        "--allow-mic-conflict",
        action="store_true",
        help="Proceed even if the mic and playback device share hardware (Bluetooth/USB headset). "
             "Warning: this typically kills playback by forcing the headset into call mode.",
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
        if args.mic:
            config["mic_enabled"] = True
        if args.mic_device is not None:
            config["mic_device_index"] = args.mic_device
        if args.mic_volume is not None:
            config["mic_volume"] = max(0.0, min(2.0, args.mic_volume))
        recorder = SystemAudioRecorder(config)

        def on_sigint(sig, frame):
            recorder.stop()
            recorder.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, on_sigint)

        print_banner()
        if recorder.start(
            max_duration_minutes=args.record,
            name=args.name,
            allow_mic_conflict=args.allow_mic_conflict,
        ):
            print("  Recording... Press Ctrl+C to stop early.\n")
            while recorder.is_recording:
                status = recorder.get_status()
                if status.get("mic_enabled"):
                    lb_level = min(int(status["rms"] * 200), 15)
                    mic_level = min(int(status.get("rms_mic", 0) * 200), 15)
                    lb_bar = "█" * lb_level + "░" * (15 - lb_level)
                    mic_bar = "█" * mic_level + "░" * (15 - mic_level)
                    print(
                        f"\r  ● REC {status['elapsed']}  SYS[{lb_bar}] MIC[{mic_bar}]  ",
                        end="", flush=True,
                    )
                else:
                    level = min(int(status["rms"] * 200), 30)
                    bar = "█" * level + "░" * (30 - level)
                    print(
                        f"\r  ● REC {status['elapsed']}  [{bar}]  ",
                        end="", flush=True,
                    )
                time.sleep(0.3)
            filepath = recorder.stop()
            if filepath and config.get("auto_transcribe"):
                model = args.model or config.get("whisper_model", "base")
                transcribe_file(filepath, model)
        else:
            err = recorder.last_error or {}
            msg = err.get("message", "Could not start recording. Check audio device.")
            print(f"\n  [ERROR] {msg}")
            if err.get("code") == "mic_device_conflict":
                print("  Re-run with --allow-mic-conflict to proceed anyway.\n")
        recorder.cleanup()
        return

    # Default: interactive mode
    interactive_mode()


if __name__ == "__main__":
    main()
