#!/usr/bin/env python3
"""
Speaker Diarization Module
===========================
Identifies different speakers in audio and labels them as Person 1, Person 2, etc.
Uses pyannote.audio for speaker diarization, merged with Whisper transcription segments.

Also supports speaker profile enrollment and recognition via speaker embeddings.

Requirements (optional):
    pip install pyannote.audio
    # Requires a HuggingFace token with accepted model terms:
    # https://huggingface.co/pyannote/speaker-diarization-3.1
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

_pipeline = None
_embedding_model = None

PROFILES_FILE = Path.home() / ".audio_capture_speaker_profiles.json"


class _DiarizeProgressHook:
    """Hook that captures pyannote diarization pipeline progress for UI reporting.

    Pyannote's pipeline calls hook(step_name, artifact, file, total, completed)
    at each processing chunk. We map known steps to approximate time weights
    and compute an overall percentage.
    """

    # Approximate wall-clock weight for each pipeline step
    WEIGHTS = {"segmentation": 0.40, "embeddings": 0.50, "clustering": 0.10}

    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self._completed_steps = []
        self._current_step = None

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None, **kwargs):
        if not self.progress_callback:
            return

        # Track step transitions — mark previous step as done
        if step_name != self._current_step:
            if self._current_step is not None and self._current_step not in self._completed_steps:
                self._completed_steps.append(self._current_step)
            self._current_step = step_name

        # Calculate overall progress
        base = sum(self.WEIGHTS.get(s, 0.05) for s in self._completed_steps)
        step_weight = self.WEIGHTS.get(step_name, 0.05)

        if total and completed:
            intra = completed / total
        else:
            intra = 0.0

        overall = (base + step_weight * intra) * 100
        step_label = step_name.replace("_", " ").capitalize()
        self.progress_callback(min(overall, 99.0), step_label)


def _patch_torchaudio():
    """Patch torchaudio for compatibility with speechbrain on newer torchaudio (>=2.11)
    where list_audio_backends() was removed."""
    try:
        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]
    except ImportError:
        pass


def is_diarization_available():
    """Check whether pyannote.audio is installed and importable."""
    try:
        _patch_torchaudio()
        import pyannote.audio  # noqa: F401
        return True
    except ImportError:
        return False


def preload_pipeline(hf_token=None, status_callback=None):
    """
    Eagerly load the diarization pipeline so it's cached for later calls.
    Call this before parallel execution to avoid model loading inside threads.
    """
    global _pipeline
    if _pipeline is not None:
        return

    _patch_torchaudio()
    from pyannote.audio import Pipeline

    def _status(msg):
        if status_callback:
            status_callback(msg)

    if not hf_token:
        raise ValueError(
            "HuggingFace token is required for pyannote.audio. "
            "Set it in settings (hf_token) or get one at https://huggingface.co/settings/tokens"
        )
    _status("Downloading/loading diarization model (pyannote/speaker-diarization-3.1)... this may take a while on first run")
    _pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    _status("Diarization model loaded")


def diarize_audio(filepath, hf_token=None, num_speakers=None, min_speakers=None, max_speakers=None,
                   status_callback=None, progress_callback=None,
                   start_time=None, end_time=None):
    """
    Run speaker diarization on an audio file.

    Args:
        status_callback: Optional callable(message_str) invoked to report progress
                         (e.g. model download, pipeline loading, analysis start).
        progress_callback: Optional callable(percent: float, step_label: str) invoked
                           with overall diarization progress (0-99) and current step name.
        start_time: Optional float, start of region to diarize (seconds).
        end_time: Optional float, end of region to diarize (seconds).

    Returns a list of speaker turns:
        [{"start": float, "end": float, "speaker": str}, ...]
    """
    global _pipeline
    _patch_torchaudio()

    def _status(msg):
        if status_callback:
            status_callback(msg)

    if _pipeline is None:
        preload_pipeline(hf_token=hf_token, status_callback=status_callback)

    params = {}
    if num_speakers is not None:
        params["num_speakers"] = num_speakers
    if min_speakers is not None:
        params["min_speakers"] = min_speakers
    if max_speakers is not None:
        params["max_speakers"] = max_speakers

    _status("Analyzing audio to identify different speakers...")

    # Preload audio as waveform dict to bypass torchcodec (which needs
    # FFmpeg shared DLLs that may not be available on Windows).
    import torch
    import soundfile as sf
    data, sample_rate = sf.read(str(filepath), dtype="float32")
    if data.ndim == 1:
        data = data[:, None]  # (samples,) -> (samples, 1)

    # Slice to selected time range if specified
    time_offset = 0.0
    if start_time is not None or end_time is not None:
        start_sample = int((start_time or 0) * sample_rate)
        end_sample = int(end_time * sample_rate) if end_time is not None else len(data)
        data = data[start_sample:end_sample]
        time_offset = start_time or 0

    waveform = torch.from_numpy(data.T)  # (channels, samples)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    hook = _DiarizeProgressHook(progress_callback=progress_callback)
    try:
        diarization = _pipeline(audio_input, hook=hook, **params)
    except TypeError:
        # Fallback: pipeline version doesn't support hook parameter
        diarization = _pipeline(audio_input, **params)

    # pyannote v4 returns a DiarizeOutput dataclass; extract the Annotation
    if hasattr(diarization, "speaker_diarization"):
        diarization = diarization.speaker_diarization

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": turn.start + time_offset,
            "end": turn.end + time_offset,
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


def format_txt_time(seconds):
    """Format seconds to HH:MM:SS for plain text output."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_txt_with_speakers(segments):
    """
    Format segments into plain text with speaker labels and timestamps.

    Groups consecutive segments by the same speaker into paragraphs:
        [00:00:00] Person 1: Hello everyone. Welcome to the meeting.
        [00:05:30] Person 2: Thanks for having me.
    """
    if not segments:
        return ""

    lines = []
    current_speaker = None
    current_start = None
    current_texts = []

    for seg in segments:
        speaker = seg["speaker"]
        text = seg["text"].strip()
        if not text:
            continue

        if speaker != current_speaker:
            if current_speaker is not None and current_texts:
                ts = format_txt_time(current_start)
                lines.append(f"[{ts}] {current_speaker}: {' '.join(current_texts)}")
            current_speaker = speaker
            current_start = seg["start"]
            current_texts = [text]
        else:
            current_texts.append(text)

    # Flush last speaker
    if current_speaker is not None and current_texts:
        ts = format_txt_time(current_start)
        lines.append(f"[{ts}] {current_speaker}: {' '.join(current_texts)}")

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


# ─── Speaker Embedding Extraction ──────────────────────────────────────────

def _load_embedding_model(hf_token=None, status_callback=None):
    """Load the pyannote embedding model (cached after first call)."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    def _status(msg):
        if status_callback:
            status_callback(msg)

    _patch_torchaudio()
    from pyannote.audio import Inference, Model

    if not hf_token:
        raise ValueError(
            "HuggingFace token is required for speaker embeddings. "
            "Set it in settings (hf_token)."
        )

    _status("Loading speaker embedding model (pyannote/wespeaker-voxceleb-resnet34-LM)... this may take a while on first run")
    model = Model.from_pretrained(
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        token=hf_token,
    )
    _embedding_model = Inference(model, window="whole")
    _status("Speaker embedding model loaded")
    return _embedding_model


def extract_speaker_embeddings(filepath, diarization_turns, hf_token=None, status_callback=None):
    """
    Extract embedding vectors for each speaker found in diarization.
    Uses pyannote/wespeaker-voxceleb-resnet34-LM to get an embedding vector per speaker.

    For each speaker, collects all their segments, extracts embeddings, and averages
    them for a more robust voiceprint.

    Returns: dict mapping speaker_id -> {"embedding": list[float], "duration": float}
    """
    import torch
    import soundfile as sf

    def _status(msg):
        if status_callback:
            status_callback(msg)

    inference = _load_embedding_model(hf_token, status_callback)

    _status("Reading audio file for embedding extraction...")
    data, sample_rate = sf.read(str(filepath), dtype="float32")
    if data.ndim == 1:
        data = data[:, None]  # (samples,) -> (samples, 1)

    # Group turns by speaker
    speaker_turns = {}
    for turn in diarization_turns:
        spk = turn["speaker"]
        if spk not in speaker_turns:
            speaker_turns[spk] = []
        speaker_turns[spk].append(turn)

    embeddings = {}
    for spk, turns in speaker_turns.items():
        _status(f"Extracting embedding for {spk} ({len(turns)} segment(s))...")
        spk_embeddings = []
        total_duration = 0.0

        for turn in turns:
            start_sample = int(turn["start"] * sample_rate)
            end_sample = int(turn["end"] * sample_rate)
            segment_data = data[start_sample:end_sample]

            # Skip very short segments (< 0.5s)
            duration = turn["end"] - turn["start"]
            if duration < 0.5:
                continue
            total_duration += duration

            waveform = torch.from_numpy(segment_data.T)  # (channels, samples)
            audio_input = {"waveform": waveform, "sample_rate": sample_rate}

            try:
                emb = inference(audio_input)
                spk_embeddings.append(emb)
            except Exception:
                # Skip segments that fail embedding extraction
                continue

        if spk_embeddings:
            # Average all embeddings for this speaker
            avg_emb = np.mean(spk_embeddings, axis=0)
            # Normalize to unit length
            norm = np.linalg.norm(avg_emb)
            if norm > 0:
                avg_emb = avg_emb / norm
            embeddings[spk] = {
                "embedding": avg_emb.tolist(),
                "duration": round(total_duration, 2),
            }

    _status(f"Extracted embeddings for {len(embeddings)} speaker(s)")
    return embeddings


# ─── Speaker Profile Storage ──────────────────────────────────────────────

def load_profiles():
    """Load speaker profiles from disk.
    Returns: {name: {"embedding": [...], "enrolled_from": "file.wav", "enrolled_date": "..."}}
    """
    if not PROFILES_FILE.exists():
        return {}
    try:
        with open(PROFILES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_profiles(profiles):
    """Save speaker profiles to disk."""
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)


def enroll_speaker(name, embedding, source_file):
    """Add or update a speaker profile."""
    profiles = load_profiles()
    profiles[name] = {
        "embedding": embedding if isinstance(embedding, list) else embedding.tolist(),
        "enrolled_from": source_file,
        "enrolled_date": datetime.now().isoformat(),
    }
    save_profiles(profiles)


def delete_profile(name):
    """Remove a speaker profile. Returns True if found and deleted."""
    profiles = load_profiles()
    if name in profiles:
        del profiles[name]
        save_profiles(profiles)
        return True
    return False


def _cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def match_speakers(embeddings, threshold=0.5):
    """
    Match speaker embeddings against stored profiles using cosine similarity.

    Args:
        embeddings: dict from extract_speaker_embeddings: {speaker_id: {"embedding": [...], "duration": float}}
        threshold: minimum cosine similarity to consider a match (0-1)

    Returns: {speaker_id: {"name": str, "confidence": float} or None}
    """
    profiles = load_profiles()
    if not profiles:
        return {spk: None for spk in embeddings}

    matches = {}
    for spk, spk_data in embeddings.items():
        spk_emb = spk_data["embedding"]
        best_name = None
        best_score = -1.0

        for profile_name, profile_data in profiles.items():
            score = _cosine_similarity(spk_emb, profile_data["embedding"])
            if score > best_score:
                best_score = score
                best_name = profile_name

        if best_score >= threshold:
            matches[spk] = {"name": best_name, "confidence": round(best_score, 4)}
        else:
            matches[spk] = None

    return matches


def normalize_speaker_labels_with_profiles(segments, diarization_turns, filepath,
                                            hf_token=None, threshold=0.5,
                                            status_callback=None):
    """
    After diarization and merging, extract embeddings, match against stored profiles,
    and replace matched speaker labels with real names.

    Unmatched speakers keep "Person N" labels.

    Returns: (normalized_segments, speaker_list, matches_info)
        matches_info: {raw_speaker_id: {"name": str, "confidence": float} or None}
    """
    def _status(msg):
        if status_callback:
            status_callback(msg)

    profiles = load_profiles()
    if not profiles:
        # No profiles enrolled — fall back to standard normalization
        normalized = normalize_speaker_labels(segments)
        speakers = get_speaker_list(normalized)
        return normalized, speakers, {}

    _status("Extracting speaker embeddings for profile matching...")
    try:
        embeddings = extract_speaker_embeddings(
            filepath, diarization_turns,
            hf_token=hf_token, status_callback=status_callback,
        )
    except Exception as e:
        _status(f"Embedding extraction failed ({e}), falling back to generic labels")
        normalized = normalize_speaker_labels(segments)
        speakers = get_speaker_list(normalized)
        return normalized, speakers, {}

    _status("Matching speakers against enrolled profiles...")
    matches = match_speakers(embeddings, threshold=threshold)

    # Build label map: raw speaker id -> display name
    label_map = {}
    counter = 1
    used_names = set()

    # First pass: assign matched names
    for seg in segments:
        raw = seg["speaker"]
        if raw in label_map:
            continue
        if raw == "Unknown":
            label_map[raw] = "Unknown"
            continue
        match = matches.get(raw)
        if match and match["name"] not in used_names:
            label_map[raw] = match["name"]
            used_names.add(match["name"])

    # Second pass: assign "Person N" to unmatched
    for seg in segments:
        raw = seg["speaker"]
        if raw not in label_map:
            label_map[raw] = f"Person {counter}"
            counter += 1

    normalized = []
    for seg in segments:
        normalized.append({
            **seg,
            "speaker": label_map[seg["speaker"]],
        })

    speakers = get_speaker_list(normalized)
    matched_names = [m["name"] for m in matches.values() if m]
    if matched_names:
        _status(f"Recognized speaker(s): {', '.join(matched_names)}")

    return normalized, speakers, matches
