#!/usr/bin/env python3
"""
Speaker Diarization Module
===========================
Identifies different speakers in audio and labels them as Person 1, Person 2, etc.
Uses pyannote.audio for speaker diarization, merged with Whisper transcription segments.

Requirements (optional):
    pip install pyannote.audio
    # Requires a HuggingFace token with accepted model terms:
    # https://huggingface.co/pyannote/speaker-diarization-3.1
"""

_pipeline = None


def is_diarization_available():
    """Check whether pyannote.audio is installed and importable."""
    try:
        import pyannote.audio  # noqa: F401
        return True
    except ImportError:
        return False


def diarize_audio(filepath, hf_token=None, num_speakers=None, min_speakers=None, max_speakers=None):
    """
    Run speaker diarization on an audio file.

    Returns a list of speaker turns:
        [{"start": float, "end": float, "speaker": str}, ...]
    """
    global _pipeline
    from pyannote.audio import Pipeline

    if _pipeline is None:
        if not hf_token:
            raise ValueError(
                "HuggingFace token is required for pyannote.audio. "
                "Set it in settings (hf_token) or get one at https://huggingface.co/settings/tokens"
            )
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

    params = {}
    if num_speakers is not None:
        params["num_speakers"] = num_speakers
    if min_speakers is not None:
        params["min_speakers"] = min_speakers
    if max_speakers is not None:
        params["max_speakers"] = max_speakers

    diarization = _pipeline(str(filepath), **params)

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })
    return turns


def merge_transcription_with_diarization(whisper_segments, diarization_turns):
    """
    Merge Whisper transcription segments with diarization speaker turns.

    For each Whisper segment, finds the speaker with the most temporal overlap
    and assigns that speaker label to the segment.

    Returns a new list of segments with an added "speaker" key.
    """
    merged = []
    for seg in whisper_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Find speaker with maximum overlap
        speaker_overlap = {}
        for turn in diarization_turns:
            overlap_start = max(seg_start, turn["start"])
            overlap_end = min(seg_end, turn["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > 0:
                speaker_overlap[turn["speaker"]] = (
                    speaker_overlap.get(turn["speaker"], 0.0) + overlap
                )

        if speaker_overlap:
            speaker = max(speaker_overlap, key=speaker_overlap.get)
        else:
            speaker = "Unknown"

        merged.append({
            **seg,
            "speaker": speaker,
        })
    return merged


def normalize_speaker_labels(segments):
    """
    Convert pyannote's internal speaker IDs (SPEAKER_00, SPEAKER_01, ...)
    into user-friendly labels (Person 1, Person 2, ...) ordered by first appearance.

    Returns a new list of segments with normalized speaker labels.
    """
    label_map = {}
    counter = 1

    normalized = []
    for seg in segments:
        raw = seg["speaker"]
        if raw not in label_map:
            if raw == "Unknown":
                label_map[raw] = "Unknown"
            else:
                label_map[raw] = f"Person {counter}"
                counter += 1
        normalized.append({
            **seg,
            "speaker": label_map[raw],
        })
    return normalized


def get_speaker_list(segments):
    """Extract ordered list of unique speaker labels from segments."""
    seen = []
    for seg in segments:
        if seg["speaker"] not in seen:
            seen.append(seg["speaker"])
    return seen


def format_txt_with_speakers(segments):
    """
    Format segments into plain text with speaker labels.

    Groups consecutive segments by the same speaker into paragraphs:
        Person 1: Hello everyone. Welcome to the meeting.
        Person 2: Thanks for having me.
    """
    if not segments:
        return ""

    lines = []
    current_speaker = None
    current_texts = []

    for seg in segments:
        speaker = seg["speaker"]
        text = seg["text"].strip()
        if not text:
            continue

        if speaker != current_speaker:
            if current_speaker is not None and current_texts:
                lines.append(f"{current_speaker}: {' '.join(current_texts)}")
            current_speaker = speaker
            current_texts = [text]
        else:
            current_texts.append(text)

    # Flush last speaker
    if current_speaker is not None and current_texts:
        lines.append(f"{current_speaker}: {' '.join(current_texts)}")

    return "\n\n".join(lines)


def format_srt_with_speakers(segments, format_srt_time_fn):
    """
    Format segments into SRT subtitle format with speaker labels.

    Each subtitle line is prefixed with [Speaker]:
        1
        00:00:00,000 --> 00:00:03,500
        [Person 1] Welcome everyone to the meeting.
    """
    lines = []
    for i, seg in enumerate(segments, 1):
        start_ts = format_srt_time_fn(seg["start"])
        end_ts = format_srt_time_fn(seg["end"])
        text = seg["text"].strip()
        speaker = seg["speaker"]
        lines.append(f"{i}\n{start_ts} --> {end_ts}\n[{speaker}] {text}\n")
    return "\n".join(lines)
