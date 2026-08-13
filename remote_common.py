"""
Shared conventions for the remote-transcription flow.

The client (a laptop that records audio but has no GPU/models) hands a
recording to the host (this beelink, which owns the Whisper + pyannote
models) and watches it happen.  Both sides need to agree on three things,
and all three live here so they cannot drift:

  1. Where a job's files live on the exposer share (`job_paths`).
  2. What a status document looks like (`STAGES`, `new_status`).
  3. How events are logged (`log_event`).

Transport lives in `remote_upload.py`; the host runner in `remote_job.py`;
the client orchestration in `remote_client.py`.
"""

import json
import os
import socket
import sys
import time
import uuid
from pathlib import Path

# Bumped when the status document or path layout changes incompatibly.  The
# client warns rather than misreading a document it doesn't understand.
PROTOCOL_VERSION = 1

# Ordered pipeline stages.  The same names are shown by the CLI and the web
# UI on both machines, so a user reading progress on the client sees exactly
# the stage the host is in.  `queued` .. `saving` are transient; `done` and
# `error` are terminal.
STAGES = [
    "uploading",      # client-side only (host never writes this)
    "queued",         # host received it, waiting for the job lock
    "received",       # host picked the job up
    "preparing",      # decoding/normalising audio
    "loading_model",  # Whisper (and diarization) model load
    "transcribing",
    "diarizing",
    "merging",        # merging speaker turns into the transcript
    "saving",
    "done",
    "error",
]

TERMINAL_STAGES = {"done", "error"}

# Human-readable labels, so client and host never invent different wording.
STAGE_LABELS = {
    "uploading": "Uploading audio to the host",
    "queued": "Queued on the host (another job is running)",
    "received": "Host received the recording",
    "preparing": "Preparing audio",
    "loading_model": "Loading models on the host",
    "transcribing": "Transcribing",
    "diarizing": "Identifying speakers",
    "merging": "Merging speaker labels",
    "saving": "Saving transcript",
    "done": "Done",
    "error": "Failed",
}


def client_hostname():
    """Stable, filesystem-safe identifier for the machine sending the job."""
    host = socket.gethostname() or os.environ.get("COMPUTERNAME") or "remote"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in host)


def job_paths(stem, host=None, remote_dir="audio-inbox", audio_ext=".flac"):
    """
    Every remote path for one job, as POSIX paths relative to exposer's
    SHARE_ROOT.  Both sides derive these from the recording stem alone, so
    neither has to tell the other where anything is.
    """
    host = host or client_hostname()
    base = "/".join(p for p in [str(remote_dir).strip("/\\"), host, stem] if p)
    return {
        "job_id": f"{host}/{stem}",
        "dir": base,
        "audio": f"{base}/{stem}{audio_ext}",
        "status": f"{base}/{stem}.status.json",
        "log": f"{base}/{stem}.joblog.jsonl",
        "txt": f"{base}/{stem}.txt",
        "srt": f"{base}/{stem}.srt",
        "json": f"{base}/{stem}.json",
    }


def new_status(job_id, stage, message=None, progress=None, **extra):
    """Build a status document.  `progress` is 0..1 for the current stage."""
    doc = {
        "version": PROTOCOL_VERSION,
        "job_id": job_id,
        "stage": stage,
        "message": message or STAGE_LABELS.get(stage, stage),
        "progress": progress,
        "updated_at": time.time(),
    }
    doc.update(extra)
    return doc


def write_status_atomic(path, doc):
    """
    Write a status document so a concurrent reader never sees a torn file.

    The client polls this file over HTTP while the host rewrites it several
    times a second; without the temp-file + replace dance it would regularly
    download half a JSON document.  os.replace is atomic on Windows and POSIX.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    os.replace(tmp, path)


# ─── Structured logging ───────────────────────────────────────────────────────
#
# Events are JSON lines with a stable `event` name and a `job_id` correlation
# field, so a whole job can be reconstructed from interleaved output.  They go
# to stderr (captured by whatever launched the process) and, when a sink path
# is set, appended to a file inside the job folder — which the client can
# download over the same channel it uses for everything else.  That is the
# "reachable from the other machine" part; stderr on the host alone would not
# help someone debugging from the client.

_log_sink = None
_log_context = {}


def set_log_sink(path, **context):
    """Send subsequent events to `path` (JSON lines) as well as stderr."""
    global _log_sink, _log_context
    _log_sink = Path(path) if path else None
    _log_context = {k: v for k, v in context.items() if v is not None}
    if _log_sink:
        _log_sink.parent.mkdir(parents=True, exist_ok=True)


# Fields that must never reach a log line.  Credentials are passed around as
# config values, and a careless `log_event("x", **config)` would publish them
# into a file the client downloads.
_REDACT_KEYS = {"password", "hf_token", "token", "key", "authorization", "secret"}


def _redact(fields):
    clean = {}
    for k, v in fields.items():
        if any(marker in k.lower() for marker in _REDACT_KEYS):
            clean[k] = "<redacted>" if v else None
        else:
            clean[k] = v
    return clean


def log_event(event, level="info", **fields):
    """Emit one structured event.  Never raises — logging must not break a job."""
    try:
        line = {
            "ts": round(time.time(), 3),
            "event": event,
            "level": level,
        }
        line.update(_log_context)
        line.update(_redact(fields))
        encoded = json.dumps(line, ensure_ascii=False, default=str)
        print(encoded, file=sys.stderr, flush=True)
        if _log_sink:
            with open(_log_sink, "a", encoding="utf-8") as f:
                f.write(encoded + "\n")
    except Exception:
        pass


def new_request_id():
    return uuid.uuid4().hex[:12]
