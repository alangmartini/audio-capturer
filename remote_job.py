#!/usr/bin/env python3
"""
Host-side runner for remotely submitted recordings.

This is what exposer's UPLOAD_HOOK launches on the beelink whenever a client
finishes uploading audio.  It transcribes (and optionally diarizes) the file
using the local models and writes a status document beside it after every
stage, which the client downloads over the same exposer endpoint it uploaded
through.  That status file is the entire "show me progress on the other
machine" mechanism — there is no second channel and nothing to keep in sync.

Install it as the hook (see setup-remote-host.ps1):

    $env:UPLOAD_HOOK='python "...\\remote_job.py" "%UPLOADED_FILE_PATH%"'

Run it by hand against any audio file to debug the same path:

    python remote_job.py "C:\\...\\MeetingInbox\\audio-inbox\\LAPTOP\\rec\\rec.flac"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from remote_common import (
    TERMINAL_STAGES,
    log_event,
    new_request_id,
    new_status,
    set_log_sink,
    write_status_atomic,
)

# Audio the hook should act on.  Anything else landing in the inbox (status
# documents the client uploaded, stray text files) is ignored rather than fed
# to Whisper.
AUDIO_SUFFIXES = {".flac", ".wav", ".mp3", ".m4a", ".ogg", ".opus", ".webm"}

# Only one transcription at a time.  Two Whisper models plus two pyannote
# pipelines will not fit comfortably in memory on this box, and exposer spawns
# one hook process per upload — back-to-back meetings would otherwise collide.
LOCK_NAME = ".transcribe.lock"
LOCK_STALE_SECONDS = 6 * 60 * 60


def _lock_path():
    return Path(os.environ.get("TEMP", ".")) / LOCK_NAME


class JobLock:
    """
    Cross-process lock built on exclusive file creation.

    A stale lock (a hook killed mid-job, a reboot) would otherwise wedge every
    later upload forever, so a lock older than LOCK_STALE_SECONDS is broken.
    """

    def __init__(self, on_wait=None, poll=5):
        self.path = _lock_path()
        self.on_wait = on_wait
        self.poll = poll
        self.fd = None

    def acquire(self):
        announced = False
        while True:
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode())
                return
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except OSError:
                    continue
                if age > LOCK_STALE_SECONDS:
                    log_event("job_lock_stale_broken", level="warn", age_seconds=round(age))
                    try:
                        self.path.unlink()
                    except OSError:
                        pass
                    continue
                if not announced and self.on_wait:
                    self.on_wait()
                    announced = True
                time.sleep(self.poll)

    def release(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        try:
            self.path.unlink()
        except OSError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class StatusWriter:
    """
    Owns the job's status document.

    Rewrites are throttled: transcription reports progress many times a second
    and the client polls every couple of seconds, so writing every single
    update would be pure disk churn for information nobody reads.  Stage
    changes and terminal states always flush immediately.
    """

    def __init__(self, path, job_id, min_interval=1.0):
        self.path = Path(path)
        self.job_id = job_id
        self.min_interval = min_interval
        self.started_at = time.time()
        self.last_write = 0.0
        self.last_stage = None
        self.doc = {}

    def update(self, stage, message=None, progress=None, force=False, **extra):
        try:
            stage_changed = stage != self.last_stage
            now = time.time()
            self.doc = new_status(
                self.job_id, stage, message=message, progress=progress,
                started_at=self.started_at,
                elapsed_seconds=round(now - self.started_at, 1),
                **extra,
            )
            if force or stage_changed or (now - self.last_write) >= self.min_interval:
                # The client stops polling on a terminal stage, so that one has
                # to land; intermediate progress can be dropped harmlessly.
                write_status_atomic(
                    self.path, self.doc, required=stage in TERMINAL_STAGES,
                )
                self.last_write = now
            if stage_changed:
                log_event("job_stage", stage=stage, message=message, progress=progress)
                self.last_stage = stage
        except Exception as exc:
            # Reporting progress must never take down the job it reports on.
            log_event("status_update_failed", level="warn", stage=stage,
                      error=f"{type(exc).__name__}: {exc}")


def _resolve_config_overrides(args):
    """Per-job overrides the client asked for, applied to the host's config."""
    overrides = {}
    if args.model:
        overrides["whisper_model"] = args.model
    if args.diarize is not None:
        overrides["diarization_enabled"] = args.diarize
    if args.language:
        overrides["language"] = args.language
    return overrides


def _read_manifest(audio_path):
    """
    Optional per-job request written by the client next to the audio.

    Lets the client pick the model / diarization / language for its own job
    without anyone touching the host's saved settings.  Absent or unreadable
    manifests just mean "use the host defaults".
    """
    manifest = audio_path.with_suffix(".request.json")
    if not manifest.exists():
        return {}
    try:
        with open(manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log_event("job_manifest_unreadable", level="warn", error=str(exc))
        return {}


def run_job(audio_path, args):
    audio_path = Path(audio_path).resolve()
    job_id = f"{audio_path.parent.name}/{audio_path.stem}"
    req_id = new_request_id()

    set_log_sink(audio_path.with_suffix(".joblog.jsonl"), job_id=job_id, request_id=req_id)
    status = StatusWriter(audio_path.with_suffix(".status.json"), job_id)

    log_event("job_received", path=str(audio_path),
              bytes=audio_path.stat().st_size if audio_path.exists() else None)
    status.update("received", "Host received the recording", 0.0, force=True)

    manifest = _read_manifest(audio_path)
    overrides = _resolve_config_overrides(args)
    for key in ("whisper_model", "diarization_enabled", "language"):
        if key in manifest and key not in overrides:
            overrides[key] = manifest[key]

    # capture.transcribe_file reads settings from the saved config, so per-job
    # choices are applied by patching load_config for this process only.  The
    # host's config file on disk is never modified by a remote job.
    import capture

    if overrides:
        original_load_config = capture.load_config

        def load_config_with_overrides():
            cfg = original_load_config()
            cfg.update(overrides)
            return cfg

        capture.load_config = load_config_with_overrides
        log_event("job_overrides_applied", **overrides)

    model_name = overrides.get("whisper_model") or capture.load_config().get("whisper_model", "base")

    def on_wait():
        status.update("queued", "Another transcription is running on the host — queued",
                      None, force=True)
        log_event("job_queued")

    with JobLock(on_wait=on_wait):
        status.update("preparing", "Starting transcription on the host", 0.0, force=True)
        started = time.time()
        result = capture.transcribe_file(
            audio_path,
            model_name=model_name,
            status_callback=lambda stage, message=None, progress=None, **extra:
                status.update(stage, message, progress, **extra),
        )

    if result is None:
        raise RuntimeError(
            "Transcription produced no result — see the host log for the cause "
            "(most often a missing model or an unreadable audio file)."
        )

    outputs = [p.name for p in (
        audio_path.with_suffix(".txt"),
        audio_path.with_suffix(".srt"),
        audio_path.with_suffix(".json"),
    ) if p.exists()]

    elapsed = round(time.time() - started, 1)
    status.update(
        "done", f"Transcript ready ({elapsed}s on the host)", 1.0, force=True,
        outputs=outputs,
        language=result.get("language"),
        model=model_name,
        duration_seconds=elapsed,
    )
    log_event("job_done", elapsed_seconds=elapsed, outputs=outputs,
              language=result.get("language"))
    return 0


def _force_utf8_streams():
    """
    Make stdout/stderr accept the characters the transcription pipeline prints.

    exposer spawns this hook with piped stdio, and on Windows a piped stream
    defaults to the ANSI code page (cp1252) — which cannot encode the box and
    progress characters capture.py writes, so the whole job would die with a
    UnicodeEncodeError before transcribing anything.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    _force_utf8_streams()
    parser = argparse.ArgumentParser(
        description="Transcribe an uploaded recording and publish progress for the sender."
    )
    parser.add_argument("path", nargs="?", help="Audio file to process (from UPLOAD_HOOK)")
    parser.add_argument("--model", help="Whisper model override for this job")
    parser.add_argument("--language", help="Force a language instead of auto-detecting")
    parser.add_argument("--diarize", dest="diarize", action="store_true", default=None,
                        help="Force speaker diarization on for this job")
    parser.add_argument("--no-diarize", dest="diarize", action="store_false",
                        help="Force speaker diarization off for this job")
    args = parser.parse_args()

    # exposer passes the uploaded file both as an argument and in the
    # environment; accept either so the hook string can be written whichever
    # way is convenient.
    raw_path = args.path or os.environ.get("UPLOADED_FILE_PATH")
    if not raw_path:
        print("No file given. Pass a path or set UPLOADED_FILE_PATH.", file=sys.stderr)
        return 2

    audio_path = Path(raw_path)
    if audio_path.suffix.lower() not in AUDIO_SUFFIXES:
        # Not an error: the client also uploads a manifest into the same
        # folder, and exposer fires the hook for every uploaded file.
        log_event("job_skipped_non_audio", path=str(audio_path), suffix=audio_path.suffix)
        return 0
    if not audio_path.exists():
        log_event("job_file_missing", level="error", path=str(audio_path))
        return 1

    try:
        return run_job(audio_path, args)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        log_event("job_failed", level="error", error=detail)
        # The client is polling and would otherwise wait forever on a crashed
        # host job, so a failure has to be published, not just logged.
        try:
            write_status_atomic(
                audio_path.with_suffix(".status.json"),
                new_status(
                    f"{audio_path.parent.name}/{audio_path.stem}", "error",
                    message=f"Transcription failed on the host: {detail}",
                    error=detail,
                ),
                required=True,
            )
        except Exception:
            pass
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
