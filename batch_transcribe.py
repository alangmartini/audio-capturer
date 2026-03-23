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
import time
from pathlib import Path

try:
    import whisper
except ImportError:
    print("=" * 60)
    print("ERROR: openai-whisper is required for transcription.")
    print("Install with:  pip install openai-whisper")
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
    print(f"\n  ⟳ Transcribing: {filepath.name}")
    start = time.time()
    result = model.transcribe(str(filepath), verbose=False)
    elapsed = time.time() - start

    # Speaker diarization
    config = load_config()
    should_diarize = diarize if diarize is not None else config.get("diarization_enabled", False)
    diarized = False
    segments = result["segments"]
    speakers = []

    if should_diarize:
        from diarize import is_diarization_available
        if not is_diarization_available():
            print("    [WARN] Diarization enabled but pyannote.audio not installed. Skipping.")
        else:
            print(f"    ⟳ Identifying speakers...")
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
                print(f"    ✓ Identified {len(speakers)} speaker(s): {', '.join(speakers)}")
            except Exception as e:
                print(f"    [WARN] Diarization failed: {e}. Saving without speaker labels.")

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
        json_path = base.with_suffix(".json")
        export = {
            "file": filepath.name,
            "language": result.get("language", "unknown"),
            "duration_seconds": segments[-1]["end"] if segments else 0,
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
        print(f"    ✓ {json_path.name}")

    print(f"    Done in {elapsed:.1f}s  |  Language: {result.get('language', '?')}")
    return result


def find_untranscribed(directory: Path):
    """Find WAV files that don't have a corresponding .txt file."""
    wavs = sorted(directory.glob("*.wav"))
    return [w for w in wavs if not w.with_suffix(".txt").exists()]


def batch_process(directory: Path, model_name="base", diarize=None):
    """Transcribe all un-transcribed WAV files in the directory."""
    pending = find_untranscribed(directory)

    if not pending:
        print("  No un-transcribed recordings found.")
        return

    print(f"  Found {len(pending)} file(s) to transcribe.")
    print(f"  Loading Whisper model '{model_name}'...\n")
    model = whisper.load_model(model_name)

    for i, wav in enumerate(pending, 1):
        print(f"  [{i}/{len(pending)}]", end="")
        transcribe_one(wav, model, diarize=diarize)

    print(f"\n  ✓ Batch complete. {len(pending)} file(s) transcribed.")


def watch_mode(directory: Path, model_name="base", diarize=None):
    """Watch directory for new WAV files and transcribe them."""
    print(f"  Watching: {directory}")
    print(f"  Model:    {model_name}")
    print(f"  Press Ctrl+C to stop.\n")

    model = whisper.load_model(model_name)
    seen = set(f.name for f in directory.glob("*.wav"))

    try:
        while True:
            current = set(f.name for f in directory.glob("*.wav"))
            new_files = current - seen

            for name in sorted(new_files):
                wav = directory / name
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
