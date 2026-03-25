#!/usr/bin/env python3
"""
Batch Transcriber
=================
Watch a folder for new recordings and transcribe them automatically,
or batch-process all existing recordings.

Usage:
    python batch_transcribe.py                        # Process all un-transcribed WAVs
    python batch_transcribe.py --watch                # Watch for new files and auto-transcribe
    python batch_transcribe.py --model medium          # Use a specific Whisper model
    python batch_transcribe.py --dir /path/to/folder   # Specify recordings folder
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

try:
    from faster_whisper import WhisperModel  # noqa: F401
except ImportError:
    print("=" * 60)
    print("ERROR: faster-whisper is required for transcription.")
    print("Install with:  pip install faster-whisper")
    print("=" * 60)
    sys.exit(1)


DEFAULT_DIR = Path.home() / "MeetingRecordings"


def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


CONFIG_FILE = Path.home() / ".audio_capture_config.json"


def load_config():
    """Load saved configuration or return defaults."""
    defaults = {
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


def transcribe_one(filepath: Path, model, output_formats=("txt", "srt", "json"), diarize=None):
    """Transcribe a single file and save outputs."""
    from whisper_loader import transcribe_audio

    config = load_config()
    total_start = time.time()
    steps = []

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

    print(f"\n  ⟳ Transcribing: {filepath.name}")

    # Speaker diarization check
    should_diarize = diarize if diarize is not None else config.get("diarization_enabled", False)
    diarize_available = False

    if should_diarize:
        from diarize import is_diarization_available
        if not is_diarization_available():
            print("    [WARN] Speaker diarization is enabled but pyannote.audio is not installed.")
            print("           Install it with: pip install pyannote.audio")
            print("           Skipping diarization — transcription will not have speaker labels.")
        else:
            from diarize import preload_pipeline
            print(f"    ⟳ Preloading diarization model...")
            t = time.time()
            preload_pipeline(
                hf_token=config.get("hf_token"),
                status_callback=lambda msg: print(f"      ⟳ {msg}"),
            )
            steps.append({"name": "Load diarization model", "seconds": round(time.time() - t, 1)})
            diarize_available = True

    if diarize_available:
        # --- Parallel execution: transcription + diarization ---
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

        print(f"    ⟳ Running transcription + diarization in parallel...")
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
                print(f"\r    [Whisper ] [{bar}] {pct*100:5.1f}% | {el_m}:{el_s:02d} elapsed | ETA: {eta_m}:{eta_s:02d}  ", end="", flush=True)

        _diarize_has_bar = [False]

        def _diarize_status(msg):
            with _print_lock:
                if _diarize_has_bar[0]:
                    print()
                    _diarize_has_bar[0] = False
                print(f"    [Diarize ] {msg}")

        def _diarize_progress(pct, step):
            with _print_lock:
                filled = int(pct / 100 * 20)
                bar = '█' * filled + '░' * (20 - filled)
                print(f"\r    [Diarize ] [{bar}] {pct:5.1f}% — {step}  ", end="", flush=True)
                _diarize_has_bar[0] = True

        def _run_whisper():
            return transcribe_audio(model, filepath, progress_callback=_whisper_progress, **transcribe_kwargs)

        def _run_diarize():
            return diarize_audio(
                str(filepath),
                hf_token=config.get("hf_token"),
                max_speakers=config.get("diarization_max_speakers"),
                status_callback=_diarize_status,
                progress_callback=_diarize_progress,
            )

        _whisper_start = time.time()
        _diarize_start = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            whisper_future = executor.submit(_run_whisper)
            diarize_future = executor.submit(_run_diarize)

            result = whisper_future.result()
            _w_elapsed = round(time.time() - _whisper_start, 1)
            steps.append({"name": "Transcription", "seconds": _w_elapsed})
            with _print_lock:
                print(f"\n    ✓ Transcription finished in {_w_elapsed}s")

            try:
                turns = diarize_future.result()
                _d_elapsed = round(time.time() - _diarize_start, 1)
                steps.append({"name": "Diarization", "seconds": _d_elapsed})
                with _print_lock:
                    if _diarize_has_bar[0]:
                        print()
                    print(f"    ✓ Diarization finished in {_d_elapsed}s")
            except Exception as e:
                with _print_lock:
                    if _diarize_has_bar[0]:
                        print()
                    print(f"    [WARN] Speaker diarization failed: {e}")
                    print(f"           Saving transcription without speaker labels.")
                turns = None

        diarized = False
        segments = result["segments"]
        speakers = []

        if turns is not None:
            print(f"    ⟳ Merging speaker labels with transcription...")
            t = time.time()
            segments = merge_transcription_with_diarization(segments, turns)

            profiles = load_profiles()
            if profiles:
                print(f"    ⟳ Matching speakers against {len(profiles)} enrolled profile(s)...")
                segments, speakers, _ = normalize_speaker_labels_with_profiles(
                    segments, turns, str(filepath),
                    hf_token=config.get("hf_token"),
                    status_callback=lambda msg: print(f"      ⟳ {msg}"),
                )
            else:
                segments = normalize_speaker_labels(segments)
                speakers = get_speaker_list(segments)
            steps.append({"name": "Merge speaker labels", "seconds": round(time.time() - t, 1)})
            diarized = True
            print(f"    ✓ Diarization complete — identified {len(speakers)} speaker(s): {', '.join(speakers)}")
    else:
        # --- Sequential: transcription only ---
        def _whisper_progress_seq(pct):
            elapsed = time.time() - _seq_start
            eta = (elapsed / pct - elapsed) if pct > 0.01 else 0
            bar_len = 30
            filled = int(bar_len * pct)
            bar = '█' * filled + '░' * (bar_len - filled)
            el_m, el_s = int(elapsed // 60), int(elapsed % 60)
            eta_m, eta_s = int(eta // 60), int(eta % 60)
            print(f"\r    [{bar}] {pct*100:5.1f}% | {el_m}:{el_s:02d} elapsed | ETA: {eta_m}:{eta_s:02d}  ", end="", flush=True)

        _seq_start = time.time()
        result = transcribe_audio(model, filepath, progress_callback=_whisper_progress_seq, **transcribe_kwargs)
        steps.append({"name": "Transcription", "seconds": round(time.time() - _seq_start, 1)})
        print()  # newline after progress bar

        diarized = False
        segments = result["segments"]
        speakers = []

    base = filepath.with_suffix("")

    if "txt" in output_formats:
        txt_path = base.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            if diarized:
                from diarize import format_txt_with_speakers
                f.write(format_txt_with_speakers(segments))
            else:
                f.write(result["text"].strip())
        print(f"    ✓ {txt_path.name}")

    if "srt" in output_formats:
        srt_path = base.with_suffix(".srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            if diarized:
                from diarize import format_srt_with_speakers
                f.write(format_srt_with_speakers(segments, format_srt_time))
            else:
                for i, seg in enumerate(segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n")
                    f.write(f"{seg['text'].strip()}\n\n")
        print(f"    ✓ {srt_path.name}")

    if "json" in output_formats:
        t = time.time()
        total_seconds = round(time.time() - total_start, 1)
        timing = {
            "total_seconds": total_seconds,
            "model": "batch",
            "steps": steps,
        }
        json_path = base.with_suffix(".json")
        export = {
            "file": filepath.name,
            "language": result.get("language", "unknown"),
            "duration_seconds": segments[-1]["end"] if segments else 0,
            "timing": timing,
            "text": result["text"].strip(),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                    **({"speaker": seg["speaker"]} if diarized else {}),
                }
                for seg in segments
            ],
        }
        if diarized:
            export["speakers"] = speakers
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        steps.append({"name": "Save outputs", "seconds": round(time.time() - t, 1)})
        print(f"    ✓ {json_path.name}")

    total_elapsed = round(time.time() - total_start, 1)
    print(f"    Done in {total_elapsed}s  |  Language: {result.get('language', '?')}")
    return result


def find_untranscribed(directory: Path):
    """Find WAV files that don't have a corresponding .txt file."""
    wavs = sorted(directory.glob("**/*.wav"))
    return [w for w in wavs if not w.with_suffix(".txt").exists()]


def batch_process(directory: Path, model_name="base", diarize=None):
    """Transcribe all un-transcribed WAV files in the directory."""
    pending = find_untranscribed(directory)

    if not pending:
        print("  No un-transcribed recordings found.")
        return

    config = load_config()
    should_diarize = diarize if diarize is not None else config.get("diarization_enabled", False)

    print(f"  ── Batch transcription starting ──")
    print(f"  Files:        {len(pending)}")
    print(f"  Whisper model: {model_name}")
    print(f"  Diarization:  {'ON (speakers will be identified)' if should_diarize else 'OFF'}")
    print()

    print(f"  Loading Whisper model '{model_name}' (downloading on first use, may take a few minutes)...")
    from whisper_loader import load_whisper_model

    def _print_load_progress(pct):
        bar_len = 30
        filled = int(bar_len * pct)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r    [{bar}] {pct*100:5.1f}% loading model weights  ", end="", flush=True)

    model = load_whisper_model(model_name, progress_callback=_print_load_progress)
    print(f"\r  ✓ Whisper model '{model_name}' ready.{' ' * 40}\n")

    for i, wav in enumerate(pending, 1):
        print(f"  [{i}/{len(pending)}]", end="")
        transcribe_one(wav, model, diarize=diarize)

    print(f"\n  ✓ Batch complete. {len(pending)} file(s) transcribed.")


def watch_mode(directory: Path, model_name="base", diarize=None):
    """Watch directory for new WAV files and transcribe them."""
    print(f"  Watching: {directory}")
    print(f"  Model:    {model_name}")
    print(f"  Press Ctrl+C to stop.\n")

    print(f"  Loading Whisper model '{model_name}' (downloading on first use, may take a few minutes)...")
    from whisper_loader import load_whisper_model

    def _print_load_progress(pct):
        bar_len = 30
        filled = int(bar_len * pct)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r    [{bar}] {pct*100:5.1f}% loading model weights  ", end="", flush=True)

    model = load_whisper_model(model_name, progress_callback=_print_load_progress)
    print(f"\r  ✓ Whisper model '{model_name}' ready. Waiting for new files...{' ' * 20}\n")
    seen = {f.name: f for f in directory.glob("**/*.wav")}

    try:
        while True:
            current = {f.name: f for f in directory.glob("**/*.wav")}
            new_names = set(current.keys()) - set(seen.keys())

            for name in sorted(new_names):
                wav = current[name]
                # Wait a moment to ensure file is fully written
                time.sleep(2)
                prev_size = -1
                while wav.stat().st_size != prev_size:
                    prev_size = wav.stat().st_size
                    time.sleep(1)

                transcribe_one(wav, model, diarize=diarize)

            seen = current
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n  Watch stopped.")


def main():
    parser = argparse.ArgumentParser(description="Batch transcribe meeting recordings")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_DIR), help="Recordings directory")
    parser.add_argument("--model", type=str, default="base", help="Whisper model")
    parser.add_argument("--watch", action="store_true", help="Watch for new files")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization (requires pyannote.audio)")
    args = parser.parse_args()

    directory = Path(args.dir)
    directory.mkdir(parents=True, exist_ok=True)

    print(f"\n  ╔═══════════════════════════════════╗")
    print(f"  ║   BATCH TRANSCRIBER               ║")
    print(f"  ╚═══════════════════════════════════╝\n")

    # --diarize flag overrides config; if not set, let config decide (None)
    diarize = True if args.diarize else None

    if args.watch:
        watch_mode(directory, args.model, diarize=diarize)
    else:
        batch_process(directory, args.model, diarize=diarize)


if __name__ == "__main__":
    main()
