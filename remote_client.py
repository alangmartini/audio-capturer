"""
Client-side remote transcription: hand a recording to the host and wait.

One call to `remote_transcribe()` does the whole round trip — prepare, upload,
watch the host work, pull the transcript back into the recording's own folder —
and reports every step through a callback so the CLI and the web UI can show
the same progress the host is producing.  When it returns, the recording folder
looks exactly as it would have if the machine had transcribed it locally.

The host side is `remote_job.py`; the shared path/status conventions are in
`remote_common.py`.
"""

import json
import time
from pathlib import Path

from remote_common import (
    STAGE_LABELS,
    TERMINAL_STAGES,
    client_hostname,
    job_paths,
    log_event,
    new_request_id,
)
from remote_upload import RemoteEndpoint

# How often to ask the host for a status update.  Fast enough that the UI feels
# live, slow enough that a 90-minute job is a few thousand tiny requests rather
# than a denial-of-service against your own tunnel.
POLL_INTERVAL = 2.0

# Give up if the host's status document stops changing for this long.  A job
# can legitimately sit on one stage for many minutes (model download, a long
# diarization pass), but silence past this point means the hook died without
# being able to publish an error.
STALL_TIMEOUT = 20 * 60

# Absolute ceiling, scaled by audio length: transcription on CPU runs at worst
# a few times slower than realtime, so this only trips on something genuinely
# stuck.
MIN_JOB_TIMEOUT = 45 * 60
JOB_TIMEOUT_PER_AUDIO_SECOND = 8

# Cloudflare caps the request body a Worker will proxy at 100 MB on Free and
# Pro plans (200 MB Business), and answers anything larger with a bare 413.
# 16 kHz mono FLAC runs about 55 MB/hour, so this only bites past roughly two
# hours of audio — but when it does, the user deserves to hear why before
# spending the upload rather than after.
WORKER_BODY_LIMIT = 100 * 1024 * 1024
WORKER_BODY_MARGIN = 2 * 1024 * 1024


class RemoteTranscribeError(RuntimeError):
    """Raised when the round trip fails; the message is user-facing."""


def _noop(event):
    pass


def _audio_duration(path):
    try:
        import wave
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return None


# ─── Audio preparation ────────────────────────────────────────────────────

def prepare_audio_for_upload(wav_path, dest_dir=None, progress_callback=None):
    """
    Re-encode a recording to 16 kHz mono FLAC for the trip over the network.

    Whisper and pyannote both resample to 16 kHz mono internally, so this
    discards nothing they would have used, while turning an hour of 48 kHz
    stereo PCM (~660 MB) into roughly 55 MB.  That matters twice: uploads over
    a VPN are slow, and Cloudflare caps the request body a Worker will proxy.

    Falls back to sending the original file untouched if PyAV isn't available,
    which is correct but much slower over the wire.
    """
    wav_path = Path(wav_path)
    try:
        import av
    except ImportError:
        log_event("remote_prepare_skipped", level="warn", reason="pyav_not_installed")
        return wav_path, False

    dest_dir = Path(dest_dir) if dest_dir else wav_path.parent
    dest = dest_dir / f"{wav_path.stem}.flac"

    try:
        container = av.open(str(wav_path))
        out = av.open(str(dest), mode="w")
        try:
            stream = out.add_stream("flac", rate=16000)
            stream.format = "s16"
            try:
                stream.layout = "mono"
            except Exception:
                # Older PyAV names this differently; the resampler below is
                # what actually guarantees mono, so this is best-effort.
                pass

            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            total = float(container.duration or 0) / 1_000_000 if container.duration else None

            def _emit_frames(frames):
                for frame in frames:
                    # Timestamps come from the resampler; letting the encoder
                    # assign its own avoids "non monotonically increasing" errors.
                    frame.pts = None
                    for packet in stream.encode(frame):
                        out.mux(packet)

            for frame in container.decode(audio=0):
                resampled = resampler.resample(frame)
                # PyAV <9 returns a single frame (or None), >=9 returns a list.
                if resampled is None:
                    continue
                if not isinstance(resampled, list):
                    resampled = [resampled]
                _emit_frames(resampled)
                if progress_callback and total and frame.time is not None:
                    progress_callback(min(frame.time / total, 1.0))

            flushed = resampler.resample(None)
            if flushed:
                _emit_frames(flushed if isinstance(flushed, list) else [flushed])
            for packet in stream.encode(None):
                out.mux(packet)
        finally:
            out.close()
            container.close()
    except Exception as exc:
        log_event("remote_prepare_failed", level="warn", error=f"{type(exc).__name__}: {exc}")
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return wav_path, False

    original = wav_path.stat().st_size
    compressed = dest.stat().st_size
    log_event("remote_prepare_complete", original_bytes=original, encoded_bytes=compressed,
              ratio=round(compressed / original, 3) if original else None)
    return dest, True


# ─── The round trip ───────────────────────────────────────────────────────

def remote_transcribe(
    wav_path,
    server_url,
    *,
    user=None,
    password=None,
    remote_dir="audio-inbox",
    model=None,
    language=None,
    diarize=None,
    on_event=None,
    poll_interval=POLL_INTERVAL,
    keep_encoded=False,
    cancel_check=None,
):
    """
    Send `wav_path` to the host, wait for the transcript, save it locally.

    `on_event(event)` receives dicts shaped like the host's status documents:
        {"stage", "message", "progress", "label", "elapsed_seconds", ...}
    with client-side stages ("preparing", "uploading") reported the same way,
    so a caller can render one progress display for the entire round trip.

    Returns a summary dict with the local paths that were written.
    """
    on_event = on_event or _noop
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise RemoteTranscribeError(f"Recording not found: {wav_path}")

    req_id = new_request_id()
    stem = wav_path.stem
    duration = _audio_duration(wav_path)
    started = time.time()

    def emit(stage, message=None, progress=None, **extra):
        event = {
            "stage": stage,
            "label": STAGE_LABELS.get(stage, stage),
            "message": message or STAGE_LABELS.get(stage, stage),
            "progress": progress,
            "elapsed_seconds": round(time.time() - started, 1),
        }
        event.update(extra)
        try:
            on_event(event)
        except Exception:
            pass
        return event

    def check_cancelled():
        if cancel_check and cancel_check():
            raise RemoteTranscribeError("Cancelled")

    endpoint = RemoteEndpoint(server_url, user=user, password=password)
    if not endpoint.password:
        log_event("remote_no_credentials", level="warn", url=server_url)

    # ── 1. Prepare ────────────────────────────────────────────────────────
    emit("preparing", "Compressing audio for transfer", 0.0)
    encoded_path, was_encoded = prepare_audio_for_upload(
        wav_path, progress_callback=lambda pct: emit("preparing", "Compressing audio for transfer", pct)
    )
    paths = job_paths(stem, remote_dir=remote_dir, audio_ext=encoded_path.suffix)
    size_mb = encoded_path.stat().st_size / (1024 * 1024)
    emit("preparing", f"Ready to send ({size_mb:.1f} MB)", 1.0, encoded=was_encoded)
    log_event("remote_job_start", request_id=req_id, job_id=paths["job_id"],
              recording=stem, size_mb=round(size_mb, 1), encoded=was_encoded,
              duration_seconds=round(duration) if duration else None)

    try:
        check_cancelled()

        # Only the Worker enforces this; a direct LAN endpoint has no such cap.
        via_worker = server_url.lower().startswith("https://")
        if via_worker and encoded_path.stat().st_size > (WORKER_BODY_LIMIT - WORKER_BODY_MARGIN):
            hours = (duration or 0) / 3600
            raise RemoteTranscribeError(
                f"This recording is {size_mb:.0f} MB after compression"
                f"{f' ({hours:.1f} hours of audio)' if hours else ''}, over the "
                f"100 MB body limit Cloudflare enforces on the Worker. Split the "
                f"recording, or transcribe it over a direct connection to the host "
                f"(an http:// LAN or Tailscale URL) which has no such limit."
            )

        # ── 2. Tell the host what to do with it ───────────────────────────
        # Uploaded first so it is already in place when the audio arrives and
        # triggers the hook.  The hook ignores non-audio uploads.
        manifest = {
            "client": client_hostname(),
            "recording": stem,
            "request_id": req_id,
            "submitted_at": time.time(),
            "original_seconds": duration,
        }
        if model:
            manifest["whisper_model"] = model
        if language:
            manifest["language"] = language
        if diarize is not None:
            manifest["diarization_enabled"] = bool(diarize)
        endpoint.upload_bytes(
            json.dumps(manifest, indent=2).encode("utf-8"),
            f"{paths['dir']}/{stem}.request.json",
        )

        # ── 3. Upload the audio ───────────────────────────────────────────
        emit("uploading", f"Uploading {size_mb:.1f} MB to the host", 0.0)

        def upload_progress(sent, total):
            emit("uploading",
                 f"Uploading {sent / (1024*1024):.1f} / {total / (1024*1024):.1f} MB",
                 sent / total if total else None)

        endpoint.upload_file(
            encoded_path, paths["audio"],
            content_type="audio/flac" if was_encoded else "audio/wav",
            progress_callback=upload_progress,
        )
        emit("uploading", "Upload complete — waiting for the host", 1.0)
        log_event("remote_upload_done", request_id=req_id, job_id=paths["job_id"])

        # ── 4. Watch the host work ────────────────────────────────────────
        final = _poll_until_done(
            endpoint, paths, emit, check_cancelled,
            poll_interval=poll_interval,
            job_timeout=max(MIN_JOB_TIMEOUT, (duration or 0) * JOB_TIMEOUT_PER_AUDIO_SECOND),
            request_id=req_id,
        )

        # ── 5. Bring the results home ─────────────────────────────────────
        emit("saving", "Downloading transcript", 0.0)
        written = []
        for key, suffix in (("txt", ".txt"), ("srt", ".srt"), ("json", ".json")):
            data = endpoint.download(paths[key], missing_ok=True)
            if data is None:
                continue
            local = wav_path.with_suffix(suffix)
            local.write_bytes(data)
            written.append(str(local))
        if not written:
            raise RemoteTranscribeError(
                "The host reported success but no transcript could be downloaded."
            )
        emit("saving", "Transcript saved", 1.0, outputs=written)

        elapsed = round(time.time() - started, 1)
        summary = {
            "job_id": paths["job_id"],
            "outputs": written,
            "txt_path": str(wav_path.with_suffix(".txt")),
            "language": final.get("language"),
            "model": final.get("model"),
            "elapsed_seconds": elapsed,
            "host_seconds": final.get("duration_seconds"),
        }
        emit("done", f"Transcript ready ({elapsed:.0f}s round trip)", 1.0, **summary)
        log_event("remote_job_complete", request_id=req_id, job_id=paths["job_id"],
                  elapsed_seconds=elapsed, outputs=len(written))
        return summary

    except RemoteTranscribeError as exc:
        emit("error", str(exc), None, error=str(exc))
        log_event("remote_job_failed", level="error", request_id=req_id,
                  job_id=paths["job_id"], error=str(exc))
        raise
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        emit("error", detail, None, error=detail)
        log_event("remote_job_failed", level="error", request_id=req_id,
                  job_id=paths["job_id"], error=detail)
        raise RemoteTranscribeError(detail) from exc
    finally:
        # The compressed copy is a transport artifact, not something the user
        # asked to keep sitting in their recordings folder.
        if was_encoded and not keep_encoded and encoded_path.exists():
            try:
                encoded_path.unlink()
            except OSError:
                pass


def _poll_until_done(endpoint, paths, emit, check_cancelled, poll_interval,
                     job_timeout, request_id):
    """Poll the host's status document until it reaches a terminal stage."""
    start = time.time()
    last_change = time.time()
    last_signature = None
    seen_any = False

    while True:
        check_cancelled()
        doc = endpoint.download_json(paths["status"], missing_ok=True)
        # A retry may see the previous attempt's result before the hook starts.
        if doc is not None and doc.get("request_id") != request_id:
            doc = None

        if doc is None:
            # Before the hook writes its first status the file simply isn't
            # there.  Only complain if it never shows up at all.
            if not seen_any and (time.time() - start) > STALL_TIMEOUT:
                raise RemoteTranscribeError(
                    "The host never picked the recording up. Check that exposer is "
                    "running there with UPLOAD_HOOK pointing at remote_job.py."
                )
            emit("queued", "Waiting for the host to pick up the recording", None)
        else:
            seen_any = True
            stage = doc.get("stage", "queued")
            signature = (stage, doc.get("message"), doc.get("progress"), doc.get("updated_at"))
            if signature != last_signature:
                last_change = time.time()
                last_signature = signature

            emit(stage, doc.get("message"), doc.get("progress"),
                 host_elapsed_seconds=doc.get("elapsed_seconds"),
                 eta_seconds=doc.get("eta_seconds"),
                 diarize_progress=doc.get("diarize_progress"),
                 diarize_step=doc.get("diarize_step"),
                 speakers=doc.get("speakers"))

            if stage == "done":
                return doc
            if stage == "error":
                raise RemoteTranscribeError(
                    doc.get("error") or doc.get("message") or "The host reported a failure."
                )

        now = time.time()
        if (now - last_change) > STALL_TIMEOUT and seen_any:
            raise RemoteTranscribeError(
                f"The host stopped reporting progress {int((now - last_change) / 60)} minutes ago. "
                f"Check the host log at {paths['log']}."
            )
        if (now - start) > job_timeout:
            raise RemoteTranscribeError(
                f"Timed out after {int((now - start) / 60)} minutes waiting for the host."
            )
        time.sleep(poll_interval)


def fetch_existing_transcript(wav_path, server_url, *, user=None, password=None,
                              remote_dir="audio-inbox"):
    """
    Pull down the transcript for a job that already ran on the host.

    Useful when the client was closed mid-job: the host keeps working, and this
    collects the result afterwards instead of re-uploading the audio.
    """
    wav_path = Path(wav_path)
    endpoint = RemoteEndpoint(server_url, user=user, password=password)
    paths = job_paths(wav_path.stem, remote_dir=remote_dir)
    doc = endpoint.download_json(paths["status"], missing_ok=True)
    if doc is None:
        raise RemoteTranscribeError("No remote job found for this recording.")
    if doc.get("stage") != "done":
        return {"status": doc, "outputs": []}

    written = []
    for key, suffix in (("txt", ".txt"), ("srt", ".srt"), ("json", ".json")):
        data = endpoint.download(paths[key], missing_ok=True)
        if data is not None:
            local = wav_path.with_suffix(suffix)
            local.write_bytes(data)
            written.append(str(local))
    return {"status": doc, "outputs": written}
