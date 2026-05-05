#!/usr/bin/env python3
"""
Web UI for System Audio Capture
================================
Flask server that wraps capture.py functionality in a browser interface.

Usage:
    python app.py
    # Opens http://127.0.0.1:5000
"""

import json
import math
import os
import struct
import sys
import threading
import time
import traceback
from pathlib import Path

# Windows consoles default to cp1252 which can't encode the unicode glyphs
# (●, ✖, ⚠) used in our log output. Switch stdio to UTF-8 so prints from
# capture.py don't raise UnicodeEncodeError mid-recording.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, render_template, request, send_file

from capture import (
    SystemAudioRecorder,
    format_srt_time,
    get_wav_duration,
    load_config,
    resolve_wav,
    save_config,
    trim_wav_to_temp,
    maybe_upload_recording,
)

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    pyaudio = None

app = Flask(__name__)

# ─── Global state ─────────────────────────────────────────────────────────────

_recorder = None
_recorder_lock = threading.Lock()
_recording_upload_override = None

_transcription = {
    "active": False,
    "stage": "idle",       # idle / loading_model / transcribing / diarizing / diarize_merging / saving / done / error
    "filename": None,
    "model": None,
    "error": None,
    "progress": None,
    "diarize_detail": None,
    "diarize_progress": None,
    "diarization_enabled": False,
    "started_at": None,
}
_transcription_lock = threading.Lock()


def _get_recorder():
    """Lazy-init recorder singleton."""
    global _recorder
    with _recorder_lock:
        if _recorder is None:
            config = load_config()
            _recorder = SystemAudioRecorder(config)
        return _recorder


def _reset_recorder():
    """Recreate recorder (e.g. after config change)."""
    global _recorder
    with _recorder_lock:
        if _recorder is not None:
            if not _recorder.is_recording:
                _recorder.cleanup()
            else:
                return False
        _recorder = None
    return True


def _get_output_dir():
    config = load_config()
    return Path(config["output_dir"])


# ─── Preview state ────────────────────────────────────────────────────────────

_preview = {
    "active": False,
    "type": None,       # "loopback" or "mic"
    "rms": 0.0,
    "peak_rms": 0.0,
    "device_name": None,
}
_preview_lock = threading.Lock()
_preview_p = None
_preview_stream = None


def _preview_callback(in_data, frame_count, time_info, status):
    """Compute RMS for preview level meters."""
    n_samples = len(in_data) // 2
    if n_samples == 0:
        return (None, pyaudio.paContinue)
    samples = struct.unpack(f'<{n_samples}h', in_data)
    sum_sq = sum(s * s for s in samples)
    level = math.sqrt(sum_sq / n_samples) / 32768.0
    with _preview_lock:
        _preview["rms"] = level
        _preview["peak_rms"] = max(_preview["peak_rms"], level)
    return (None, pyaudio.paContinue)


def _start_audio_preview(device_type="loopback"):
    """Start a preview stream for mic or loopback. Returns (ok, error)."""
    global _preview_p, _preview_stream

    _stop_audio_preview()

    if pyaudio is None:
        return False, "pyaudiowpatch not installed"

    _preview_p = pyaudio.PyAudio()

    if device_type == "loopback":
        from capture import get_loopback_device
        config = load_config()
        device = get_loopback_device(_preview_p, config.get("device_index"))
        if not device:
            _preview_p.terminate()
            _preview_p = None
            return False, "No loopback device found"
    else:
        try:
            device = _preview_p.get_default_input_device_info()
        except OSError:
            _preview_p.terminate()
            _preview_p = None
            return False, "No microphone found"

    rate = int(device["defaultSampleRate"])
    channels = max(1, device["maxInputChannels"])

    try:
        _preview_stream = _preview_p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=1024,
            stream_callback=_preview_callback,
        )
        _preview_stream.start_stream()
    except Exception as e:
        _preview_p.terminate()
        _preview_p = None
        return False, str(e)

    with _preview_lock:
        _preview.update({
            "active": True,
            "type": device_type,
            "rms": 0.0,
            "peak_rms": 0.0,
            "device_name": device["name"],
        })

    return True, None


def _stop_audio_preview():
    """Stop the current preview stream."""
    global _preview_p, _preview_stream

    if _preview_stream:
        try:
            _preview_stream.stop_stream()
            _preview_stream.close()
        except Exception:
            pass
        _preview_stream = None

    if _preview_p:
        try:
            _preview_p.terminate()
        except Exception:
            pass
        _preview_p = None

    with _preview_lock:
        _preview.update({
            "active": False,
            "type": None,
            "rms": 0.0,
            "peak_rms": 0.0,
            "device_name": None,
        })


# ─── Error handling ───────────────────────────────────────────────────────────

@app.errorhandler(Exception)
def handle_error(e):
    tb = traceback.format_exc()
    return jsonify({"ok": False, "error": str(e), "traceback": tb}), 500


# ─── Pages ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─── Recording endpoints ─────────────────────────────────────────────────────

@app.route("/api/recording/start", methods=["POST"])
def recording_start():
    global _recording_upload_override

    # Stop any active preview before recording
    _stop_audio_preview()

    recorder = _get_recorder()
    if recorder.is_recording:
        return jsonify({"ok": False, "error": "Already recording"}), 409

    data = request.get_json(silent=True) or {}
    max_dur = data.get("max_duration_minutes")
    name = data.get("name")
    allow_mic_conflict = bool(data.get("allow_mic_conflict", False))
    upload_keys = {"remote_upload_enabled", "remote_upload_url", "remote_upload_path"}
    _recording_upload_override = None
    if any(key in data for key in upload_keys):
        _recording_upload_override = {
            "remote_upload_enabled": bool(data.get("remote_upload_enabled", False)),
            "remote_upload_url": data.get("remote_upload_url"),
            "remote_upload_path": data.get("remote_upload_path") or "audio-inbox",
        }

    ok = recorder.start(
        max_duration_minutes=max_dur,
        name=name,
        allow_mic_conflict=allow_mic_conflict,
    )
    if not ok:
        _recording_upload_override = None
        err = recorder.last_error or {}
        payload = {
            "ok": False,
            "error": err.get("message") or "Could not start recording. Check audio device.",
            "error_code": err.get("code"),
        }
        # Pass through extra detail fields the UI needs (e.g. device names for the confirm dialog)
        for k in ("loopback", "mic"):
            if k in err:
                payload[k] = err[k]
        # After a failed start, discard the recorder. Windows WASAPI can leave a
        # PyAudio instance in a half-cleaned state after a failed mic/stream open
        # — subsequent attempts then surface as -9999 "Unanticipated host error"
        # or even "No loopback device found" until the PyAudio instance is
        # recreated. Force a fresh one on the next attempt.
        _reset_recorder()
        # 409 for actionable conflicts the user can resolve; 500 for hard failures
        status = 409 if err.get("code") == "mic_device_conflict" else 500
        return jsonify(payload), status

    return jsonify({"ok": True, "file": str(recorder.filepath)})


@app.route("/api/recording/stop", methods=["POST"])
def recording_stop():
    global _recording_upload_override

    with _recorder_lock:
        if _recorder is None:
            return jsonify({"ok": False, "error": "Not recording"}), 409
    recorder = _recorder
    if not recorder.is_recording:
        return jsonify({"ok": False, "error": "Not recording"}), 409

    filepath = recorder.stop()
    result = {"ok": True, "file": str(filepath) if filepath else None}

    # Auto-transcribe if enabled
    config = load_config()
    if _recording_upload_override is not None:
        config.update(_recording_upload_override)
        _recording_upload_override = None
    if filepath and config.get("auto_transcribe"):
        _start_transcription(filepath.name, config.get("whisper_model", "base"))
        result["auto_transcribe_started"] = True

    if filepath and config.get("remote_upload_enabled"):
        def _upload_worker(path, upload_config):
            maybe_upload_recording(path, upload_config)

        threading.Thread(target=_upload_worker, args=(filepath, config.copy()), daemon=True).start()
        result["remote_upload_started"] = True

    return jsonify(result)


@app.route("/api/recording/status")
def recording_status():
    with _recorder_lock:
        if _recorder is None:
            return jsonify({"recording": False})
    status = _recorder.get_status()
    return jsonify(status)


# ─── Recordings list ─────────────────────────────────────────────────────────

@app.route("/api/recordings")
def list_recordings():
    out_dir = _get_output_dir()
    if not out_dir.exists():
        return jsonify({"recordings": []})

    recordings = []
    for wav in sorted(out_dir.glob("**/*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = wav.stat()
        entry = {
            "name": wav.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 1),
            "modified": stat.st_mtime,
            "transcripts": {},
        }
        for ext in (".txt", ".srt", ".json"):
            t = wav.with_suffix(ext)
            if t.exists():
                entry["transcripts"][ext.lstrip(".")] = str(t)
        # Read timing from JSON export if available
        json_path = wav.with_suffix(".json")
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    jdata = json.load(f)
                if "timing" in jdata:
                    entry["timing"] = jdata["timing"]
            except Exception:
                pass
        recordings.append(entry)

    return jsonify({"recordings": recordings})


@app.route("/api/recordings/open-folder", methods=["POST"])
def open_recordings_folder():
    out_dir = _get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.startfile(str(out_dir))
    return jsonify({"ok": True, "path": str(out_dir)})


@app.route("/api/recordings/<name>/transcript")
def get_transcript(name):
    fmt = request.args.get("format", "txt")
    if fmt not in ("txt", "srt", "json"):
        return jsonify({"ok": False, "error": f"Invalid format: {fmt}"}), 400

    out_dir = _get_output_dir()
    wav_path = resolve_wav(out_dir, name)
    transcript_path = wav_path.with_suffix(f".{fmt}")

    if not transcript_path.exists():
        return jsonify({"ok": False, "error": f"Transcript not found: {transcript_path.name}"}), 404

    content = transcript_path.read_text(encoding="utf-8")
    if fmt == "json":
        return jsonify({"ok": True, "content": json.loads(content), "format": fmt})
    return jsonify({"ok": True, "content": content, "format": fmt})


@app.route("/api/recordings/<name>/download")
def download_recording(name):
    fmt = request.args.get("format", "wav")
    if fmt not in ("wav", "txt", "srt", "json"):
        return jsonify({"ok": False, "error": f"Invalid format: {fmt}"}), 400

    out_dir = _get_output_dir()
    wav_path = resolve_wav(out_dir, name)

    if fmt == "wav":
        target = wav_path
    else:
        target = wav_path.with_suffix(f".{fmt}")

    if not target.exists():
        return jsonify({"ok": False, "error": f"File not found: {target.name}"}), 404

    return send_file(str(target), as_attachment=True)


@app.route("/api/recordings/<name>/duration")
def get_duration(name):
    out_dir = _get_output_dir()
    wav_path = resolve_wav(out_dir, name)
    if not wav_path.exists():
        return jsonify({"ok": False, "error": f"File not found: {name}"}), 404
    duration = get_wav_duration(wav_path)
    return jsonify({"ok": True, "duration": duration})


@app.route("/api/recordings/<name>", methods=["DELETE"])
def delete_recording(name):
    out_dir = _get_output_dir()
    wav_path = resolve_wav(out_dir, name)

    # Prevent path traversal
    if not wav_path.resolve().is_relative_to(out_dir.resolve()):
        return jsonify({"ok": False, "error": "Invalid filename"}), 400

    deleted = []
    for ext in (".wav", ".txt", ".srt", ".json"):
        p = wav_path.with_suffix(ext) if ext != ".wav" else wav_path
        if p.exists():
            p.unlink()
            deleted.append(p.name)

    if not deleted:
        return jsonify({"ok": False, "error": "No files found to delete"}), 404

    # Remove empty subfolder if applicable
    parent = wav_path.parent
    if parent != out_dir and parent.exists() and not any(parent.iterdir()):
        parent.rmdir()

    return jsonify({"ok": True, "deleted": deleted})


@app.route("/api/recordings/<name>/rename", methods=["POST"])
def rename_recording(name):
    data = request.get_json(silent=True) or {}
    new_name = data.get("new_name", "").strip()
    if not new_name:
        return jsonify({"ok": False, "error": "Missing 'new_name'"}), 400

    # Sanitize
    safe = "".join(c for c in new_name if c.isalnum() or c in " -_").strip()
    if not safe:
        return jsonify({"ok": False, "error": "Invalid name (use alphanumeric, spaces, hyphens, underscores)"}), 400

    out_dir = _get_output_dir()
    wav_path = resolve_wav(out_dir, name)

    # Prevent path traversal
    if not wav_path.resolve().is_relative_to(out_dir.resolve()):
        return jsonify({"ok": False, "error": "Invalid filename"}), 400

    if not wav_path.exists():
        return jsonify({"ok": False, "error": f"Recording not found: {name}"}), 404

    new_folder = out_dir / safe
    if new_folder.exists():
        return jsonify({"ok": False, "error": f"A recording named '{safe}' already exists"}), 409

    old_stem = wav_path.stem
    old_parent = wav_path.parent
    is_subfolder = (old_parent.name == old_stem and old_parent.parent.resolve() == out_dir.resolve())

    renamed = []
    if is_subfolder:
        old_parent.rename(new_folder)
        for ext in (".wav", ".txt", ".srt", ".json"):
            old_file = new_folder / f"{old_stem}{ext}"
            if old_file.exists():
                new_file = new_folder / f"{safe}{ext}"
                old_file.rename(new_file)
                renamed.append({"old": f"{old_stem}{ext}", "new": f"{safe}{ext}"})
    else:
        new_folder.mkdir(parents=True, exist_ok=True)
        for ext in (".wav", ".txt", ".srt", ".json"):
            old_file = old_parent / f"{old_stem}{ext}"
            if old_file.exists():
                new_file = new_folder / f"{safe}{ext}"
                old_file.rename(new_file)
                renamed.append({"old": old_file.name, "new": new_file.name})

    return jsonify({"ok": True, "renamed": renamed, "new_name": f"{safe}.wav"})


# ─── Transcription ────────────────────────────────────────────────────────────

def _start_transcription(filename, model_name, start_time=None, end_time=None):
    """Launch background transcription thread."""
    config = load_config()
    with _transcription_lock:
        if _transcription["active"]:
            return False
        _transcription.update({
            "active": True,
            "stage": "loading_model",
            "filename": filename,
            "model": model_name,
            "error": None,
            "progress": None,
            "diarize_detail": None,
            "diarization_enabled": config.get("diarization_enabled", False),
        })

    t = threading.Thread(target=_transcribe_worker, args=(filename, model_name, start_time, end_time), daemon=True)
    t.start()
    return True


def _ensure_ffmpeg():
    """Make sure ffmpeg is on PATH. Returns True if available, False otherwise."""
    import shutil
    if shutil.which("ffmpeg"):
        return True
    # Try imageio-ffmpeg (pip install imageio-ffmpeg)
    try:
        import imageio_ffmpeg
        ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        return True
    except (ImportError, Exception):
        pass
    # Try static-ffmpeg (pip install static-ffmpeg)
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        return True
    except (ImportError, Exception):
        pass
    return False


def _transcribe_worker(filename, model_name, start_time=None, end_time=None):
    """Background transcription worker."""
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        with _transcription_lock:
            _transcription.update({
                "active": False,
                "stage": "error",
                "error": "faster-whisper not installed. Run: pip install faster-whisper",
            })
        return

    if not _ensure_ffmpeg():
        with _transcription_lock:
            _transcription.update({
                "active": False,
                "stage": "error",
                "error": "ffmpeg not found. Install it with: pip install imageio-ffmpeg",
            })
        return

    out_dir = _get_output_dir()
    filepath = resolve_wav(out_dir, filename)

    if not filepath.exists():
        with _transcription_lock:
            _transcription.update({
                "active": False,
                "stage": "error",
                "error": f"File not found: {filename}",
            })
        return

    try:
        config = load_config()
        total_start = time.time()
        steps = []

        # --- Step: Load Whisper model ---
        with _transcription_lock:
            _transcription["stage"] = "loading_model"
            _transcription["model_load_progress"] = {"percent": 0}

        from whisper_loader import load_whisper_model, transcribe_audio

        def _model_load_progress(pct):
            with _transcription_lock:
                _transcription["model_load_progress"] = {"percent": round(pct * 100, 1)}

        t = time.time()
        model = load_whisper_model(model_name, progress_callback=_model_load_progress)
        steps.append({"name": "Load Whisper model", "seconds": round(time.time() - t, 1)})

        # --- Step: Trim audio (if partial) ---
        audio_path = filepath
        temp_path = None
        time_offset = 0
        if start_time is not None or end_time is not None:
            t = time.time()
            total_dur = get_wav_duration(filepath)
            actual_start = start_time if start_time is not None else 0
            actual_end = end_time if end_time is not None else total_dur
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
            prompt = f"Technical meeting transcript. Terms: {terms}."
            if len(prompt) > 500:
                prompt = prompt[:500]
            transcribe_kwargs["initial_prompt"] = prompt

        lang = config.get("language")
        if lang:
            transcribe_kwargs["language"] = lang

        # Check diarization availability before starting parallel work
        diarize_available = False
        if config.get("diarization_enabled"):
            from diarize import is_diarization_available
            if not is_diarization_available():
                with _transcription_lock:
                    _transcription.update({
                        "active": False,
                        "stage": "error",
                        "error": "Diarization is enabled but pyannote.audio is not installed. "
                                 "Run: pip install pyannote.audio",
                    })
                return

            # --- Step: Load diarization model ---
            with _transcription_lock:
                _transcription["stage"] = "loading_diarize_model"
                _transcription["diarize_detail"] = "Loading diarization model..."

            from diarize import (
                preload_pipeline,
                diarize_audio,
                merge_transcription_with_diarization,
                normalize_speaker_labels,
                normalize_speaker_labels_with_profiles,
                get_speaker_list,
                format_txt_with_speakers,
                format_srt_with_speakers,
                load_profiles,
            )

            def _preload_status(msg):
                with _transcription_lock:
                    _transcription["diarize_detail"] = msg

            t = time.time()
            preload_pipeline(
                hf_token=config.get("hf_token"),
                status_callback=_preload_status,
            )
            steps.append({"name": "Load diarization model", "seconds": round(time.time() - t, 1)})
            diarize_available = True

        # --- Progress callbacks ---
        _parallel_start = time.time()
        with _transcription_lock:
            _transcription["started_at"] = _parallel_start

        def _whisper_progress(pct):
            elapsed = time.time() - _parallel_start
            eta = (elapsed / pct - elapsed) if pct > 0.01 else None
            with _transcription_lock:
                _transcription["progress"] = {
                    "percent": round(pct * 100, 1),
                    "elapsed": round(elapsed),
                    "eta": round(eta) if eta is not None else None,
                }

        if diarize_available:
            # --- Parallel execution: transcription + diarization ---
            from concurrent.futures import ThreadPoolExecutor

            with _transcription_lock:
                _transcription["stage"] = "processing"
                _transcription["progress"] = {"percent": 0, "elapsed": 0, "eta": None}
                _transcription["diarize_detail"] = "Starting speaker analysis..."
                _transcription["diarize_progress"] = {"percent": 0}

            def _diarize_status(msg):
                with _transcription_lock:
                    _transcription["diarize_detail"] = msg

            def _diarize_progress(percent, step_label):
                with _transcription_lock:
                    _transcription["diarize_progress"] = {"percent": round(percent, 1)}
                    _transcription["diarize_detail"] = f"Speaker diarization: {step_label}..."

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
            _whisper_start = time.time()
            _diarize_start = time.time()
            with ThreadPoolExecutor(max_workers=2) as executor:
                whisper_future = executor.submit(_run_whisper)
                diarize_future = executor.submit(_run_diarize)

                result = whisper_future.result()
                steps.append({"name": "Transcription", "seconds": round(time.time() - _whisper_start, 1)})
                turns = diarize_future.result()
                steps.append({"name": "Diarization", "seconds": round(time.time() - _diarize_start, 1)})

            diarized = False
            segments = result["segments"]
            speakers = []

            # --- Step: Merge speaker labels ---
            with _transcription_lock:
                _transcription["stage"] = "diarize_merging"
            t = time.time()
            segments = merge_transcription_with_diarization(segments, turns)

            # Use speaker profile matching if profiles exist
            profiles = load_profiles()
            if profiles:
                with _transcription_lock:
                    _transcription["diarize_detail"] = "Matching speakers against enrolled profiles..."

                def _profile_status(msg):
                    with _transcription_lock:
                        _transcription["diarize_detail"] = msg

                segments, speakers, _ = normalize_speaker_labels_with_profiles(
                    segments, turns, str(audio_path),
                    hf_token=config.get("hf_token"),
                    status_callback=_profile_status,
                )
                steps.append({"name": "Merge speaker labels", "seconds": round(time.time() - t, 1)})
                # Profile matching time is included in merge step above
            else:
                segments = normalize_speaker_labels(segments)
                speakers = get_speaker_list(segments)
                steps.append({"name": "Merge speaker labels", "seconds": round(time.time() - t, 1)})
            diarized = True
        else:
            # --- Sequential: transcription only ---
            with _transcription_lock:
                _transcription["stage"] = "transcribing"
                _transcription["progress"] = {"percent": 0, "elapsed": 0, "eta": None}

            t = time.time()
            result = transcribe_audio(model, audio_path, progress_callback=_whisper_progress, **transcribe_kwargs)
            steps.append({"name": "Transcription", "seconds": round(time.time() - t, 1)})
            diarized = False
            segments = result["segments"]
            speakers = []

        # Offset timestamps for partial transcription
        if time_offset > 0:
            for seg in segments:
                seg["start"] += time_offset
                seg["end"] += time_offset

        # Clean up temp file
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

        # --- Step: Save outputs ---
        with _transcription_lock:
            _transcription["stage"] = "saving"

        t = time.time()

        # Save .txt
        txt_path = filepath.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            if diarized:
                f.write(format_txt_with_speakers(segments))
            else:
                f.write(result["text"].strip())

        # Save .srt
        srt_path = filepath.with_suffix(".srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            if diarized:
                f.write(format_srt_with_speakers(segments, format_srt_time))
            else:
                for i, seg in enumerate(segments, 1):
                    start_ts = format_srt_time(seg["start"])
                    end_ts = format_srt_time(seg["end"])
                    text = seg["text"].strip()
                    f.write(f"{i}\n{start_ts} --> {end_ts}\n{text}\n\n")

        # Save .json
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
            json_export["speakers"] = speakers

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_export, f, indent=2, ensure_ascii=False)

        steps.append({"name": "Save outputs", "seconds": round(time.time() - t, 1)})

        with _transcription_lock:
            _transcription.update({
                "active": False,
                "stage": "done",
                "progress": {
                    "language": result.get("language", "unknown"),
                    "segments": len(segments),
                    **({"speakers": speakers} if diarized else {}),
                },
            })

    except Exception as e:
        with _transcription_lock:
            _transcription.update({
                "active": False,
                "stage": "error",
                "error": f"{type(e).__name__}: {e}",
            })


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"ok": False, "error": "Missing 'filename'"}), 400

    config = load_config()
    model = data.get("model") or config.get("whisper_model", "base")
    start_time = data.get("start_time")
    end_time = data.get("end_time")

    with _transcription_lock:
        if _transcription["active"]:
            return jsonify({"ok": False, "error": "Transcription already in progress"}), 409

    ok = _start_transcription(filename, model, start_time, end_time)
    if not ok:
        return jsonify({"ok": False, "error": "Could not start transcription"}), 500

    return jsonify({"ok": True, "filename": filename, "model": model})


@app.route("/api/transcribe/status")
def transcribe_status():
    with _transcription_lock:
        snapshot = dict(_transcription)
        # Auto-reset terminal states so they don't re-display on next poll
        if _transcription["stage"] in ("error", "done"):
            _transcription.update({
                "stage": "idle",
                "error": None,
                "filename": None,
                "progress": None,
                "model_load_progress": None,
                "diarize_detail": None,
                "diarize_progress": None,
                "diarization_enabled": False,
                "started_at": None,
            })
    return jsonify(snapshot)


# ─── Audio Preview ────────────────────────────────────────────────────────────

@app.route("/api/preview/start", methods=["POST"])
def preview_start():
    data = request.get_json(silent=True) or {}
    device_type = data.get("type", "loopback")
    if device_type not in ("loopback", "mic"):
        return jsonify({"ok": False, "error": "type must be 'loopback' or 'mic'"}), 400

    # Don't allow preview while recording
    with _recorder_lock:
        if _recorder is not None and _recorder.is_recording:
            return jsonify({"ok": False, "error": "Cannot preview while recording"}), 409

    ok, error = _start_audio_preview(device_type)
    if not ok:
        return jsonify({"ok": False, "error": error}), 500

    with _preview_lock:
        device_name = _preview["device_name"]
    return jsonify({"ok": True, "type": device_type, "device": device_name})


@app.route("/api/preview/stop", methods=["POST"])
def preview_stop():
    _stop_audio_preview()
    return jsonify({"ok": True})


@app.route("/api/preview/status")
def preview_status():
    with _preview_lock:
        return jsonify(dict(_preview))


# ─── Devices ──────────────────────────────────────────────────────────────────

@app.route("/api/devices")
def list_devices():
    if pyaudio is None:
        return jsonify({"ok": False, "error": "pyaudiowpatch not installed"}), 500

    p = None
    try:
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            is_loopback = bool(dev.get("isLoopbackDevice", False))
            devices.append({
                "index": i,
                "name": dev["name"],
                "is_loopback": is_loopback,
                "is_input": dev["maxInputChannels"] > 0 and not is_loopback,
                "max_input_channels": dev["maxInputChannels"],
                "max_output_channels": dev["maxOutputChannels"],
                "sample_rate": int(dev["defaultSampleRate"]),
            })
        return jsonify({"ok": True, "devices": devices})
    finally:
        if p is not None:
            p.terminate()


# ─── Settings ─────────────────────────────────────────────────────────────────

@app.route("/api/settings")
def get_settings():
    config = load_config()
    return jsonify({"ok": True, "settings": config})


@app.route("/api/settings", methods=["PUT"])
def update_settings():
    data = request.get_json(silent=True) or {}
    config = load_config()

    allowed_keys = {
        "output_dir", "auto_transcribe", "whisper_model", "device_index",
        "diarization_enabled", "hf_token", "diarization_max_speakers",
        "vocabulary_terms", "hotwords", "language",
        "mic_enabled", "mic_device_index", "mic_volume",
        "remote_upload_enabled", "remote_upload_url", "remote_upload_path",
    }
    for key in allowed_keys:
        if key in data:
            config[key] = data[key]

    save_config(config)
    _reset_recorder()

    return jsonify({"ok": True, "settings": config})


# ─── Diarization status ───────────────────────────────────────────────────

@app.route("/api/diarization/status")
def diarization_status():
    from diarize import is_diarization_available
    config = load_config()
    return jsonify({
        "ok": True,
        "available": is_diarization_available(),
        "enabled": config.get("diarization_enabled", False),
        "has_token": bool(config.get("hf_token")),
    })


# ─── Speaker Profiles ─────────────────────────────────────────────────────

_speaker_task = {
    "active": False,
    "stage": "idle",  # idle / loading / diarizing / extracting / matching / enrolling / done / error
    "detail": None,
    "diarize_progress": None,
    "error": None,
    "result": None,
}
_speaker_task_lock = threading.Lock()


@app.route("/api/speakers")
def list_speakers():
    """List all enrolled speaker profiles."""
    from diarize import load_profiles
    profiles = load_profiles()
    # Strip embeddings from response (they're large)
    summary = {}
    for name, data in profiles.items():
        summary[name] = {
            "enrolled_date": data.get("enrolled_date", ""),
            "enrolled_from": data.get("enrolled_from", ""),
        }
    return jsonify({"ok": True, "profiles": summary, "count": len(summary)})


@app.route("/api/speakers/enroll", methods=["POST"])
def enroll_speaker_endpoint():
    """Enroll speakers from a recording.
    Body: {"filename": "recording.wav", "assignments": {"SPEAKER_00": "John", "SPEAKER_01": "Maria"},
           "start_time": 0, "end_time": 120}
    """
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    assignments = data.get("assignments", {})
    start_time = data.get("start_time")
    end_time = data.get("end_time")

    if not filename:
        return jsonify({"ok": False, "error": "Missing 'filename'"}), 400
    if not assignments:
        return jsonify({"ok": False, "error": "Missing 'assignments'"}), 400

    with _speaker_task_lock:
        if _speaker_task["active"]:
            return jsonify({"ok": False, "error": "A speaker task is already running"}), 409
        _speaker_task.update({
            "active": True,
            "stage": "loading",
            "detail": f"Starting enrollment from {filename}...",
            "error": None,
            "result": None,
        })

    t = threading.Thread(
        target=_enroll_worker, args=(filename, assignments, start_time, end_time), daemon=True
    )
    t.start()
    return jsonify({"ok": True})


def _enroll_worker(filename, assignments, start_time=None, end_time=None):
    """Background worker for speaker enrollment."""
    try:
        from diarize import (
            is_diarization_available,
            diarize_audio,
            extract_speaker_embeddings,
            enroll_speaker,
        )

        if not is_diarization_available():
            with _speaker_task_lock:
                _speaker_task.update({
                    "active": False,
                    "stage": "error",
                    "error": "pyannote.audio is not installed",
                })
            return

        config = load_config()
        out_dir = _get_output_dir()
        filepath = resolve_wav(out_dir, filename)

        if not filepath.exists():
            with _speaker_task_lock:
                _speaker_task.update({
                    "active": False,
                    "stage": "error",
                    "error": f"File not found: {filename}",
                })
            return

        def _status(msg):
            with _speaker_task_lock:
                _speaker_task["detail"] = msg

        def _diarize_prog(percent, step_label):
            with _speaker_task_lock:
                _speaker_task["diarize_progress"] = {"percent": round(percent, 1)}
                _speaker_task["detail"] = f"Speaker diarization: {step_label}..."

        # Run diarization
        with _speaker_task_lock:
            _speaker_task["stage"] = "diarizing"
            _speaker_task["detail"] = "Running speaker diarization..."

        turns = diarize_audio(
            str(filepath),
            hf_token=config.get("hf_token"),
            max_speakers=config.get("diarization_max_speakers"),
            status_callback=_status,
            progress_callback=_diarize_prog,
            start_time=start_time,
            end_time=end_time,
        )

        # Extract embeddings
        with _speaker_task_lock:
            _speaker_task["stage"] = "extracting"
            _speaker_task["detail"] = "Extracting speaker embeddings..."
            _speaker_task["diarize_progress"] = None

        embeddings = extract_speaker_embeddings(
            str(filepath), turns,
            hf_token=config.get("hf_token"),
            status_callback=_status,
        )

        # Enroll the assigned speakers
        with _speaker_task_lock:
            _speaker_task["stage"] = "enrolling"

        enrolled = []
        for raw_id, name in assignments.items():
            name = name.strip()
            if not name:
                continue
            if raw_id in embeddings:
                enroll_speaker(name, embeddings[raw_id]["embedding"], filename)
                enrolled.append(name)
                with _speaker_task_lock:
                    _speaker_task["detail"] = f"Enrolled {name}"

        with _speaker_task_lock:
            _speaker_task.update({
                "active": False,
                "stage": "done",
                "detail": None,
                "result": {"enrolled": enrolled},
            })

    except Exception as e:
        with _speaker_task_lock:
            _speaker_task.update({
                "active": False,
                "stage": "error",
                "error": f"{type(e).__name__}: {e}",
            })


@app.route("/api/speakers/<name>", methods=["DELETE"])
def delete_speaker_endpoint(name):
    """Delete a speaker profile."""
    from diarize import delete_profile
    ok = delete_profile(name)
    if ok:
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": f"Profile '{name}' not found"}), 404


@app.route("/api/speakers/test", methods=["POST"])
def test_speaker_endpoint():
    """Test speaker recognition on a file.
    Body: {"filename": "recording.wav"}
    """
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"ok": False, "error": "Missing 'filename'"}), 400

    with _speaker_task_lock:
        if _speaker_task["active"]:
            return jsonify({"ok": False, "error": "A speaker task is already running"}), 409
        _speaker_task.update({
            "active": True,
            "stage": "loading",
            "detail": f"Starting recognition test on {filename}...",
            "error": None,
            "result": None,
        })

    t = threading.Thread(
        target=_test_recognition_worker, args=(filename,), daemon=True
    )
    t.start()
    return jsonify({"ok": True})


def _test_recognition_worker(filename):
    """Background worker for testing speaker recognition."""
    try:
        from diarize import (
            is_diarization_available,
            diarize_audio,
            extract_speaker_embeddings,
            match_speakers,
            load_profiles,
        )

        if not is_diarization_available():
            with _speaker_task_lock:
                _speaker_task.update({
                    "active": False,
                    "stage": "error",
                    "error": "pyannote.audio is not installed",
                })
            return

        profiles = load_profiles()
        if not profiles:
            with _speaker_task_lock:
                _speaker_task.update({
                    "active": False,
                    "stage": "error",
                    "error": "No speaker profiles enrolled yet",
                })
            return

        config = load_config()
        out_dir = _get_output_dir()
        filepath = resolve_wav(out_dir, filename)

        if not filepath.exists():
            with _speaker_task_lock:
                _speaker_task.update({
                    "active": False,
                    "stage": "error",
                    "error": f"File not found: {filename}",
                })
            return

        def _status(msg):
            with _speaker_task_lock:
                _speaker_task["detail"] = msg

        def _diarize_prog(percent, step_label):
            with _speaker_task_lock:
                _speaker_task["diarize_progress"] = {"percent": round(percent, 1)}
                _speaker_task["detail"] = f"Speaker diarization: {step_label}..."

        # Run diarization
        with _speaker_task_lock:
            _speaker_task["stage"] = "diarizing"
            _speaker_task["detail"] = "Running speaker diarization..."

        turns = diarize_audio(
            str(filepath),
            hf_token=config.get("hf_token"),
            max_speakers=config.get("diarization_max_speakers"),
            status_callback=_status,
            progress_callback=_diarize_prog,
        )

        # Extract embeddings
        with _speaker_task_lock:
            _speaker_task["stage"] = "extracting"
            _speaker_task["detail"] = "Extracting speaker embeddings..."
            _speaker_task["diarize_progress"] = None

        embeddings = extract_speaker_embeddings(
            str(filepath), turns,
            hf_token=config.get("hf_token"),
            status_callback=_status,
        )

        # Match against profiles
        with _speaker_task_lock:
            _speaker_task["stage"] = "matching"
            _speaker_task["detail"] = "Matching speakers against profiles..."

        matches = match_speakers(embeddings)

        # Build result
        results = {}
        for spk, spk_data in embeddings.items():
            match = matches.get(spk)
            results[spk] = {
                "duration": spk_data["duration"],
                "match": match,
            }

        with _speaker_task_lock:
            _speaker_task.update({
                "active": False,
                "stage": "done",
                "detail": None,
                "result": {"speakers": results},
            })

    except Exception as e:
        with _speaker_task_lock:
            _speaker_task.update({
                "active": False,
                "stage": "error",
                "error": f"{type(e).__name__}: {e}",
            })


@app.route("/api/speakers/identify", methods=["POST"])
def identify_speakers_endpoint():
    """Identify speakers in a recording (diarize without enrolling).
    Body: {"filename": "recording.wav", "start_time": 0, "end_time": 120}
    Returns raw speaker segments and any matches against profiles.
    """
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    start_time = data.get("start_time")
    end_time = data.get("end_time")
    if not filename:
        return jsonify({"ok": False, "error": "Missing 'filename'"}), 400

    with _speaker_task_lock:
        if _speaker_task["active"]:
            return jsonify({"ok": False, "error": "A speaker task is already running"}), 409
        _speaker_task.update({
            "active": True,
            "stage": "loading",
            "detail": f"Identifying speakers in {filename}...",
            "error": None,
            "result": None,
        })

    t = threading.Thread(
        target=_identify_worker, args=(filename, start_time, end_time), daemon=True
    )
    t.start()
    return jsonify({"ok": True})


def _identify_worker(filename, start_time=None, end_time=None):
    """Background worker to identify speakers in a recording."""
    try:
        from diarize import (
            is_diarization_available,
            diarize_audio,
            extract_speaker_embeddings,
            match_speakers,
            load_profiles,
        )

        if not is_diarization_available():
            with _speaker_task_lock:
                _speaker_task.update({
                    "active": False,
                    "stage": "error",
                    "error": "pyannote.audio is not installed",
                })
            return

        config = load_config()
        out_dir = _get_output_dir()
        filepath = resolve_wav(out_dir, filename)

        if not filepath.exists():
            with _speaker_task_lock:
                _speaker_task.update({
                    "active": False,
                    "stage": "error",
                    "error": f"File not found: {filename}",
                })
            return

        def _status(msg):
            with _speaker_task_lock:
                _speaker_task["detail"] = msg

        def _diarize_prog(percent, step_label):
            with _speaker_task_lock:
                _speaker_task["diarize_progress"] = {"percent": round(percent, 1)}
                _speaker_task["detail"] = f"Speaker diarization: {step_label}..."

        # Run diarization
        with _speaker_task_lock:
            _speaker_task["stage"] = "diarizing"
            _speaker_task["detail"] = "Running speaker diarization..."

        turns = diarize_audio(
            str(filepath),
            hf_token=config.get("hf_token"),
            max_speakers=config.get("diarization_max_speakers"),
            status_callback=_status,
            progress_callback=_diarize_prog,
            start_time=start_time,
            end_time=end_time,
        )

        # Extract embeddings
        with _speaker_task_lock:
            _speaker_task["stage"] = "extracting"
            _speaker_task["detail"] = "Extracting speaker embeddings..."
            _speaker_task["diarize_progress"] = None

        embeddings = extract_speaker_embeddings(
            str(filepath), turns,
            hf_token=config.get("hf_token"),
            status_callback=_status,
        )

        # Match against profiles if any exist
        profiles = load_profiles()
        matches = {}
        if profiles:
            with _speaker_task_lock:
                _speaker_task["stage"] = "matching"
                _speaker_task["detail"] = "Matching speakers against profiles..."
            matches = match_speakers(embeddings)

        # Build result with speaker info (include turns for audio preview)
        speakers_info = {}
        for spk, spk_data in embeddings.items():
            match = matches.get(spk)
            spk_turns = [t for t in turns if t["speaker"] == spk]
            speakers_info[spk] = {
                "duration": spk_data["duration"],
                "segments": len(spk_turns),
                "match": match,
                "turns": [{"start": t["start"], "end": t["end"]} for t in spk_turns],
            }

        with _speaker_task_lock:
            _speaker_task.update({
                "active": False,
                "stage": "done",
                "detail": None,
                "result": {
                    "filename": filename,
                    "speakers": speakers_info,
                    "total_speakers": len(speakers_info),
                },
            })

    except Exception as e:
        with _speaker_task_lock:
            _speaker_task.update({
                "active": False,
                "stage": "error",
                "error": f"{type(e).__name__}: {e}",
            })


@app.route("/api/speakers/task/status")
def speaker_task_status():
    """Get current speaker task status (similar to transcribe_status)."""
    with _speaker_task_lock:
        snapshot = dict(_speaker_task)
        # Auto-reset terminal states
        if _speaker_task["stage"] in ("error", "done"):
            _speaker_task.update({
                "stage": "idle",
                "detail": None,
                "diarize_progress": None,
                "error": None,
                "result": None,
            })
    return jsonify(snapshot)


# ─── Speaker Audio Preview ────────────────────────────────────────────────

@app.route("/api/speakers/preview")
def speaker_preview():
    """Serve an audio clip for a speaker given time segments.
    Query params:
        filename: recording WAV filename
        segments: JSON array of {start, end} objects — time ranges to extract
    Returns a WAV file with the concatenated speaker segments.
    """
    import io
    import wave as wave_mod

    filename = request.args.get("filename")
    segments_json = request.args.get("segments")

    if not filename or not segments_json:
        return jsonify({"ok": False, "error": "Missing filename or segments"}), 400

    try:
        segments = json.loads(segments_json)
    except (json.JSONDecodeError, TypeError):
        return jsonify({"ok": False, "error": "Invalid segments JSON"}), 400

    if not segments:
        return jsonify({"ok": False, "error": "No segments provided"}), 400

    out_dir = _get_output_dir()
    filepath = resolve_wav(out_dir, filename)

    if not filepath.exists():
        return jsonify({"ok": False, "error": f"File not found: {filename}"}), 404

    # Prevent path traversal
    if not filepath.resolve().is_relative_to(out_dir.resolve()):
        return jsonify({"ok": False, "error": "Invalid filename"}), 400

    # Read the source WAV and extract speaker segments
    try:
        with wave_mod.open(str(filepath), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            all_frames = wf.readframes(n_frames)

        # Sort segments by start time, limit to first 30s of audio to keep preview reasonable
        segs = sorted(segments, key=lambda s: s["start"])
        extracted = bytearray()
        total_preview = 0.0
        max_preview_seconds = 30.0
        bytes_per_second = framerate * n_channels * sampwidth

        for seg in segs:
            if total_preview >= max_preview_seconds:
                break
            start_byte = int(seg["start"] * bytes_per_second)
            end_byte = int(seg["end"] * bytes_per_second)
            # Align to frame boundaries
            frame_size = n_channels * sampwidth
            start_byte = (start_byte // frame_size) * frame_size
            end_byte = (end_byte // frame_size) * frame_size
            # Clamp
            start_byte = max(0, min(start_byte, len(all_frames)))
            end_byte = max(start_byte, min(end_byte, len(all_frames)))

            remaining = max_preview_seconds - total_preview
            max_bytes = int(remaining * bytes_per_second)
            max_bytes = (max_bytes // frame_size) * frame_size
            chunk = all_frames[start_byte:end_byte]
            if len(chunk) > max_bytes:
                chunk = chunk[:max_bytes]

            extracted.extend(chunk)
            total_preview += len(chunk) / bytes_per_second

        # Write to in-memory WAV
        buf = io.BytesIO()
        with wave_mod.open(buf, "wb") as out_wf:
            out_wf.setnchannels(n_channels)
            out_wf.setsampwidth(sampwidth)
            out_wf.setframerate(framerate)
            out_wf.writeframes(bytes(extracted))

        buf.seek(0)
        return send_file(buf, mimetype="audio/wav", download_name=f"preview_{filename}")

    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to extract preview: {e}"}), 500


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Audio Capturer Web UI")
    _parser.add_argument("--port", type=int, default=5000, help="Port to run on (default: 5000)")
    _parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    _args = _parser.parse_args()

    print("=" * 60)
    print(f"  Audio Capturer Web UI")
    print(f"  Open http://{_args.host}:{_args.port} in your browser")
    print("=" * 60)
    app.run(host=_args.host, port=_args.port, debug=True, use_reloader=True)
