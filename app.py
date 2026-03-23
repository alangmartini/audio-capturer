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
import threading
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from capture import (
    SystemAudioRecorder,
    format_srt_time,
    load_config,
    save_config,
)

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    pyaudio = None

app = Flask(__name__)

# ─── Global state ─────────────────────────────────────────────────────────────

_recorder = None
_recorder_lock = threading.Lock()

_transcription = {
    "active": False,
    "stage": "idle",       # idle / loading_model / transcribing / diarizing / saving / done / error
    "filename": None,
    "model": None,
    "error": None,
    "progress": None,
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
    # Stop any active preview before recording
    _stop_audio_preview()

    recorder = _get_recorder()
    if recorder.is_recording:
        return jsonify({"ok": False, "error": "Already recording"}), 409

    data = request.get_json(silent=True) or {}
    max_dur = data.get("max_duration_minutes")
    name = data.get("name")

    ok = recorder.start(max_duration_minutes=max_dur, name=name)
    if not ok:
        return jsonify({"ok": False, "error": "Could not start recording. Check audio device."}), 500

    return jsonify({"ok": True, "file": str(recorder.filepath)})


@app.route("/api/recording/stop", methods=["POST"])
def recording_stop():
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
    if filepath and config.get("auto_transcribe"):
        _start_transcription(filepath.name, config.get("whisper_model", "base"))
        result["auto_transcribe_started"] = True

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
    for wav in sorted(out_dir.glob("*.wav"), reverse=True):
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
                entry["transcripts"][ext.lstrip(".")] = True
        recordings.append(entry)

    return jsonify({"recordings": recordings})


@app.route("/api/recordings/<name>/transcript")
def get_transcript(name):
    fmt = request.args.get("format", "txt")
    if fmt not in ("txt", "srt", "json"):
        return jsonify({"ok": False, "error": f"Invalid format: {fmt}"}), 400

    out_dir = _get_output_dir()
    wav_path = out_dir / name
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
    wav_path = out_dir / name

    if fmt == "wav":
        target = wav_path
    else:
        target = wav_path.with_suffix(f".{fmt}")

    if not target.exists():
        return jsonify({"ok": False, "error": f"File not found: {target.name}"}), 404

    return send_file(str(target), as_attachment=True)


@app.route("/api/recordings/<name>", methods=["DELETE"])
def delete_recording(name):
    out_dir = _get_output_dir()
    wav_path = out_dir / name

    # Prevent path traversal
    if not wav_path.resolve().parent == out_dir.resolve():
        return jsonify({"ok": False, "error": "Invalid filename"}), 400

    deleted = []
    for ext in (".wav", ".txt", ".srt", ".json"):
        p = wav_path.with_suffix(ext) if ext != ".wav" else wav_path
        if p.exists():
            p.unlink()
            deleted.append(p.name)

    if not deleted:
        return jsonify({"ok": False, "error": "No files found to delete"}), 404

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
    wav_path = out_dir / name

    # Prevent path traversal
    if not wav_path.resolve().parent == out_dir.resolve():
        return jsonify({"ok": False, "error": "Invalid filename"}), 400

    if not wav_path.exists():
        return jsonify({"ok": False, "error": f"Recording not found: {name}"}), 404

    new_wav = out_dir / f"{safe}.wav"
    if new_wav.exists():
        return jsonify({"ok": False, "error": f"A recording named '{safe}.wav' already exists"}), 409

    renamed = []
    for ext in (".wav", ".txt", ".srt", ".json"):
        old = wav_path.with_suffix(ext)
        if old.exists():
            new_path = out_dir / f"{safe}{ext}"
            old.rename(new_path)
            renamed.append({"old": old.name, "new": new_path.name})

    return jsonify({"ok": True, "renamed": renamed, "new_name": f"{safe}.wav"})


# ─── Transcription ────────────────────────────────────────────────────────────

def _start_transcription(filename, model_name):
    """Launch background transcription thread."""
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
        })

    t = threading.Thread(target=_transcribe_worker, args=(filename, model_name), daemon=True)
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


def _transcribe_worker(filename, model_name):
    """Background transcription worker."""
    try:
        import whisper
    except ImportError:
        with _transcription_lock:
            _transcription.update({
                "active": False,
                "stage": "error",
                "error": "Whisper not installed. Run: pip install openai-whisper",
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
    filepath = out_dir / filename

    if not filepath.exists():
        with _transcription_lock:
            _transcription.update({
                "active": False,
                "stage": "error",
                "error": f"File not found: {filename}",
            })
        return

    try:
        with _transcription_lock:
            _transcription["stage"] = "loading_model"
        model = whisper.load_model(model_name)

        with _transcription_lock:
            _transcription["stage"] = "transcribing"
        result = model.transcribe(str(filepath), verbose=False)

        # Speaker diarization (if enabled)
        config = load_config()
        diarized = False
        segments = result["segments"]
        speakers = []

        if config.get("diarization_enabled"):
            from diarize import is_diarization_available
            if is_diarization_available():
                with _transcription_lock:
                    _transcription["stage"] = "diarizing"
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
                except Exception as e:
                    # Diarization failed; continue without speaker labels
                    pass

        with _transcription_lock:
            _transcription["stage"] = "saving"

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
        json_path = filepath.with_suffix(".json")
        json_export = {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
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

    with _transcription_lock:
        if _transcription["active"]:
            return jsonify({"ok": False, "error": "Transcription already in progress"}), 409

    ok = _start_transcription(filename, model)
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
            devices.append({
                "index": i,
                "name": dev["name"],
                "is_loopback": bool(dev.get("isLoopbackDevice", False)),
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
