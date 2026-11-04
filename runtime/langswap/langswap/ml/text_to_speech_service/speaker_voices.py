"""Speaker references and verified segment caches, without model dependencies."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import textwrap


def split_speech_text(text, max_chars):
    """Bound inference chunks without truncating text or splitting words."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    return textwrap.wrap(text, width=max_chars, break_long_words=False,
                         break_on_hyphens=False)


def speaker_label(segment):
    return getattr(segment, "speaker", None) or "SPEAKER_00"


def validate_segment_intervals(segments):
    seen = set()
    for segment in segments:
        interval = float(segment.start), float(segment.end)
        start, end = interval
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("Synthesis segment has invalid timestamps")
        if interval in seen:
            raise ValueError(f"Duplicate synthesis interval {start}-{end} would overwrite another segment")
        seen.add(interval)


def isolated_reference_segments(segments):
    """Full isolated clips with matching transcripts for transcript-based clones."""
    validate_segment_intervals(segments)
    grouped = {}
    for segment in segments:
        speaker = speaker_label(segment)
        overlapping = any(
            speaker_label(other) != speaker and
            float(other.start) < float(segment.end) and float(other.end) > float(segment.start)
            for other in segments
        )
        if segment.source_file and not overlapping:
            grouped.setdefault(speaker, []).append(segment)
    missing = {speaker_label(segment) for segment in segments} - grouped.keys()
    if missing:
        raise ValueError(f"No isolated reference with matching transcript for speakers: {sorted(missing)}")
    return grouped


def file_digest(path):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_audio(path):
    """Reject unreadable, empty, non-finite or silent synthesis/reference audio."""
    import numpy as np
    import soundfile as sf

    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Missing audio: {path}")
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate <= 0 or len(audio) < 0.03 * rate or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")
    if float(np.max(np.abs(audio.mean(axis=1)))) <= 1e-6:
        raise ValueError(f"Silent audio: {path}")
    return audio, rate


def build_speaker_references(vocals, segments, folder, max_seconds=8.0):
    """Collect speech per label, excluding intervals occupied by other speakers.

    Never use an arbitrary full-track clip or another speaker as a fallback.
    Reference filenames do not trust diarization labels as filesystem paths.
    """
    import numpy as np
    import soundfile as sf

    validate_segment_intervals(segments)
    os.makedirs(folder, exist_ok=True)
    audio, rate = validate_audio(vocals)
    audio = audio.mean(axis=1)
    duration = len(audio) / rate
    by_speaker = {}
    for segment in segments:
        start, end = float(segment.start), float(segment.end)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("Speaker reference segment has invalid timestamps")
        start, end = max(0.0, start), min(duration, end)
        if end > start:
            by_speaker.setdefault(speaker_label(segment), []).append((start, end))
    refs = {}
    for speaker, intervals in by_speaker.items():
        # Merge own overlapping intervals before removing competing speech.
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        clean = merged
        for other, competing in by_speaker.items():
            if other == speaker:
                continue
            for left, right in competing:
                remaining = []
                for start, end in clean:
                    if right <= start or left >= end:
                        remaining.append((start, end))
                    else:
                        if start < left:
                            remaining.append((start, left))
                        if right < end:
                            remaining.append((right, end))
                clean = remaining
        pieces, collected = [], 0
        limit = int(max_seconds * rate)
        for start, end in sorted(clean, key=lambda x: x[1] - x[0], reverse=True):
            piece = audio[int(start * rate):int(end * rate)][:limit - collected]
            if len(piece) < int(0.3 * rate) or np.max(np.abs(piece)) <= 1e-6:
                continue
            pieces.append(piece)
            collected += len(piece)
            if collected >= limit:
                break
        if not pieces:
            raise ValueError(f"No isolated audible reference for speaker {speaker!r}")
        name = hashlib.sha256(speaker.encode("utf-8")).hexdigest()[:20]
        path = os.path.join(folder, f"ref_{name}.wav")
        sf.write(path, np.concatenate(pieces), rate)
        validate_audio(path)
        refs[speaker] = path
    missing = {speaker_label(s) for s in segments} - refs.keys()
    if missing:
        raise ValueError(f"No audio reference for speakers: {sorted(missing)}")
    return refs


def synthesize_cached_segment(segment, folder, language, engine, voice_key, generate):
    """Publish audio and its provenance only after successful validation."""
    if not (segment.translation or "").strip():
        raise ValueError("Cannot synthesize an empty translation")
    validate_segment_intervals([segment])
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{segment.start}_{segment.end}.wav")
    metadata_path = path + ".tts.json"
    identity = {
        "version": 1, "engine": engine, "language": language,
        "speaker": speaker_label(segment), "voice": voice_key,
        "text": segment.translation, "start": segment.start, "end": segment.end,
    }
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        if (isinstance(metadata, dict) and metadata.get("identity") == identity and
                metadata.get("sha256") == file_digest(path)):
            validate_audio(path)
            segment.generated_file = path
            return path
    except (OSError, ValueError, RuntimeError):
        pass
    pending = path + ".pending.wav"
    pending_metadata = metadata_path + ".pending"
    try:
        generate(pending)
        validate_audio(pending)
        metadata = {"identity": identity, "sha256": file_digest(pending)}
        with open(pending_metadata, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, sort_keys=True)
        os.replace(pending, path)
        os.replace(pending_metadata, metadata_path)
        segment.generated_file = path
        return path
    finally:
        for temporary in (pending, pending_metadata):
            if os.path.exists(temporary):
                os.remove(temporary)
