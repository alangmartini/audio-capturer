"""
Whisper model loader and transcription adapter for faster-whisper.

Provides load_whisper_model() and transcribe_audio() that return results
in the same dict format as openai-whisper, so the rest of the codebase
(diarization, formatting, saving) needs zero changes.
"""

import os


def load_whisper_model(model_name, progress_callback=None):
    """
    Load a Whisper model via faster-whisper (CTranslate2).

    Parameters
    ----------
    model_name : str
        Model name (e.g. "base", "large-v3") or path to a model directory.
    progress_callback : callable, optional
        Called with a float 0.0-1.0 during model loading.
        Note: faster-whisper handles download progress internally;
        this callback fires once at 0.0 (start) and 1.0 (loaded).

    Returns
    -------
    model : faster_whisper.WhisperModel
    """
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int16"

    if progress_callback:
        progress_callback(0.0)

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )

    if progress_callback:
        progress_callback(1.0)

    return model


def transcribe_audio(model, audio_path, progress_callback=None, **kwargs):
    """
    Transcribe audio and return result in openai-whisper-compatible dict format.

    Parameters
    ----------
    model : faster_whisper.WhisperModel
        Loaded model from load_whisper_model().
    audio_path : str or Path
        Path to audio file.
    progress_callback : callable, optional
        Called with a float 0.0-1.0 as segments are processed.
    **kwargs
        Additional arguments passed to model.transcribe()
        (e.g. initial_prompt, beam_size, language).

    Returns
    -------
    dict with keys: "text", "segments" (list of dicts), "language"
        Each segment dict has: "id", "start", "end", "text"
    """
    # faster-whisper and openai-whisper both default to beam_size=5
    kwargs.setdefault("beam_size", 5)

    # Silero VAD: filter non-speech before decoding.  Prevents hallucinated
    # text on silence/noise and CTranslate2 zero-length segment crashes.
    kwargs.setdefault("vad_filter", True)

    segments_gen, info = model.transcribe(str(audio_path), **kwargs)

    segments = []
    full_text_parts = []
    duration = info.duration if info.duration and info.duration > 0 else 0

    def _collect(gen):
        """Drain a segment generator, appending to segments/full_text_parts."""
        for seg in gen:
            segments.append({
                "id": len(segments) + 1,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            })
            full_text_parts.append(seg.text.strip())

            if progress_callback and duration > 0:
                pct = min(seg.end / duration, 1.0)
                progress_callback(pct)

    import sys
    try:
        _collect(segments_gen)
    except ValueError as e:
        if "maximum decoding length" not in str(e).lower():
            raise

        # CTranslate2 bug: segment too short to decode.
        # Retry from last known position instead of silently losing audio.
        last_end = segments[-1]["end"] if segments else 0
        remaining = (duration - last_end) if duration > 0 else 0

        print(f"\n  [WARN] CTranslate2 hit a zero-length segment at ~{last_end:.1f}s "
              f"({len(segments)} segment(s) so far).", file=sys.stderr)

        max_retries = 50
        retry = 0
        while remaining > 30 and retry < max_retries:
            retry += 1
            resume_at = last_end + 1.0  # skip 1s past the bad spot
            print(f"  [INFO] Retrying from {resume_at:.1f}s "
                  f"({remaining:.0f}s remaining, attempt {retry})...",
                  file=sys.stderr)
            try:
                retry_kwargs = dict(kwargs)
                retry_kwargs["clip_timestamps"] = str(resume_at)
                retry_gen, _ = model.transcribe(str(audio_path), **retry_kwargs)
                _collect(retry_gen)
                break  # consumed remaining audio
            except ValueError as retry_e:
                if "maximum decoding length" not in str(retry_e).lower():
                    raise
                last_end = segments[-1]["end"] if segments else resume_at
                remaining = (duration - last_end) if duration > 0 else 0
                print(f"  [WARN] Another bad segment at ~{last_end:.1f}s, "
                      f"skipping ahead...", file=sys.stderr)

        print(f"  [INFO] Recovery complete — {len(segments)} total segment(s).",
              file=sys.stderr)

    if progress_callback:
        progress_callback(1.0)

    return {
        "text": " ".join(full_text_parts),
        "segments": segments,
        "language": info.language,
    }
