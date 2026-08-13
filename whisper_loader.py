"""
Whisper model loader and transcription adapter for faster-whisper.

Provides load_whisper_model() and transcribe_audio() that return results
in the same dict format as openai-whisper, so the rest of the codebase
(diarization, formatting, saving) needs zero changes.
"""

import os
import sys


# ─── Hardware detection ──────────────────────────────────────────────────────
#
# faster-whisper is built on CTranslate2, which only supports NVIDIA CUDA for
# GPU inference (no AMD/ROCm, no DirectML, no Vulkan).  We detect via
# ctranslate2 directly so we don't depend on torch — torch is a heavy optional
# dep used only by pyannote diarization, and may be a CPU-only wheel even on a
# CUDA box.

# Compute types supported per device, in preference order (fastest → safest).
# Used to pick a sensible default when the user requests "auto", and as a
# fallback ladder if the user picks one the runtime doesn't support.
_CUDA_COMPUTE_PREFERENCE = ["float16", "int8_float16", "bfloat16", "int8", "float32"]
_CPU_COMPUTE_PREFERENCE = ["int8", "int8_float32", "int16", "float32"]


def detect_devices():
    """
    Probe what the local CTranslate2 install actually supports.

    Returns a dict shaped for direct JSON serialisation:
        {
          "cuda_available": bool,
          "cuda_device_count": int,
          "cuda_devices": [{"index": int, "name": str}, ...],
          "cpu_compute_types": [str, ...],
          "cuda_compute_types": [str, ...],   # empty when no CUDA
          "ctranslate2_version": str,
        }
    """
    info = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "cpu_compute_types": [],
        "cuda_compute_types": [],
        "ctranslate2_version": "unknown",
    }
    try:
        import ctranslate2
    except ImportError:
        return info

    info["ctranslate2_version"] = getattr(ctranslate2, "__version__", "unknown")
    try:
        info["cpu_compute_types"] = sorted(ctranslate2.get_supported_compute_types("cpu"))
    except Exception:
        pass

    try:
        n = ctranslate2.get_cuda_device_count()
    except Exception:
        n = 0
    info["cuda_device_count"] = n
    info["cuda_available"] = n > 0

    if n > 0:
        try:
            info["cuda_compute_types"] = sorted(ctranslate2.get_supported_compute_types("cuda"))
        except Exception:
            pass
        # Best-effort device names — torch gives nice names if it's the cuda
        # build, otherwise we fall back to generic labels.
        names = [None] * n
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(n):
                    try:
                        names[i] = torch.cuda.get_device_name(i)
                    except Exception:
                        pass
        except Exception:
            pass
        for i in range(n):
            info["cuda_devices"].append({
                "index": i,
                "name": names[i] or f"CUDA device {i}",
            })

    return info


def _resolve_device_and_compute_type(device, compute_type, cuda_device_index, detected):
    """
    Translate user preferences ("auto"/None or specific value) into concrete
    CTranslate2 args.  Returns (device, device_index, compute_type, fallbacks).

    `fallbacks` is a list of compute_types to try in order if the requested one
    isn't supported by the local CTranslate2 build — built from the device's
    preference list, with the requested type first.
    """
    # Device
    requested_device = (device or "auto").lower()
    if requested_device == "auto":
        effective_device = "cuda" if detected["cuda_available"] else "cpu"
    elif requested_device == "cuda":
        if not detected["cuda_available"]:
            raise RuntimeError(
                "compute_device='cuda' was requested but no CUDA device is "
                "available.  faster-whisper requires NVIDIA + CUDA libraries; "
                "AMD GPUs are not supported.  Set compute_device to 'auto' or "
                "'cpu' in settings."
            )
        effective_device = "cuda"
    elif requested_device == "cpu":
        effective_device = "cpu"
    else:
        raise ValueError(f"Unknown compute_device: {device!r}")

    # Device index (only meaningful for cuda)
    if effective_device == "cuda":
        idx = 0 if cuda_device_index is None else int(cuda_device_index)
        if idx < 0 or idx >= detected["cuda_device_count"]:
            print(f"  [WARN] cuda_device_index={idx} is out of range "
                  f"(have {detected['cuda_device_count']} device(s)). Using 0.",
                  file=sys.stderr)
            idx = 0
        device_index = idx
    else:
        device_index = 0  # ignored by ctranslate2 for cpu

    # Compute type
    pref = _CUDA_COMPUTE_PREFERENCE if effective_device == "cuda" else _CPU_COMPUTE_PREFERENCE
    requested_ct = (compute_type or "auto").lower()
    if requested_ct == "auto":
        primary = pref[0]
        fallbacks = list(pref)
    else:
        primary = requested_ct
        # Build a fallback ladder: requested first, then anything else from
        # the device's preference list we haven't already listed.
        fallbacks = [requested_ct] + [c for c in pref if c != requested_ct]

    return effective_device, device_index, primary, fallbacks


def load_whisper_model(
    model_name,
    progress_callback=None,
    *,
    device=None,
    compute_type=None,
    cuda_device_index=None,
):
    """
    Load a Whisper model via faster-whisper (CTranslate2).

    Parameters
    ----------
    model_name : str
        Model name (e.g. "base", "large-v3") or path to a model directory.
    progress_callback : callable, optional
        Called with a float 0.0-1.0 during model loading.  faster-whisper
        handles download progress internally; this callback fires once at 0.0
        (start) and 1.0 (loaded).
    device : str or None
        "auto" (default), "cuda", or "cpu".  None is treated as "auto".
    compute_type : str or None
        "auto" (default), or any CTranslate2 compute type ("float16",
        "int8_float16", "int8", "int16", "float32", ...).  None is treated as
        "auto" — float16 on CUDA, int8 on CPU.
    cuda_device_index : int or None
        Which GPU to use when several are present.  Defaults to 0.  Ignored on
        CPU.

    Returns
    -------
    model : faster_whisper.WhisperModel
    """
    from faster_whisper import WhisperModel

    detected = detect_devices()
    effective_device, device_index, primary_ct, fallbacks = _resolve_device_and_compute_type(
        device, compute_type, cuda_device_index, detected,
    )

    if progress_callback:
        progress_callback(0.0)

    # Try the requested compute_type first; on ValueError mentioning compute
    # type, walk the fallback ladder.  Any other ValueError is re-raised.
    last_error = None
    chosen_ct = None
    model = None
    seen = set()
    for ct in fallbacks:
        if ct in seen:
            continue
        seen.add(ct)
        try:
            model = WhisperModel(
                model_name,
                device=effective_device,
                device_index=device_index,
                compute_type=ct,
            )
            chosen_ct = ct
            break
        except ValueError as e:
            if "compute type" not in str(e).lower():
                raise
            last_error = e
            print(f"  [INFO] compute_type='{ct}' not supported here, trying next…",
                  file=sys.stderr)

    if model is None:
        # Exhausted the ladder — surface the original error.
        raise last_error if last_error else RuntimeError(
            "Could not load Whisper model with any compute type."
        )

    # Tell the user what we actually picked.  CLAUDE.md: "Every operation must
    # clearly communicate what is happening to the user at all times."
    where = "CPU"
    if effective_device == "cuda":
        names = [d["name"] for d in detected["cuda_devices"] if d["index"] == device_index]
        gpu_name = names[0] if names else f"CUDA device {device_index}"
        where = f"GPU (cuda:{device_index} — {gpu_name})"
    if chosen_ct != primary_ct:
        print(f"  [INFO] Whisper running on {where} (compute_type={chosen_ct}, "
              f"requested {primary_ct}).", file=sys.stderr)
    else:
        print(f"  [INFO] Whisper running on {where} (compute_type={chosen_ct}).",
              file=sys.stderr)

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
