import attr


@attr.s(auto_attribs=True)
class Segment:
    end: float
    start: float
    text: str
    words: list[dict]
    speaker: str = None


@attr.s(auto_attribs=True)
class Output:
    detected_language: str
    device: str
    model: str
    transcription: str
    translation: str = None
    segments: list[Segment] = attr.ib(factory=list)


@attr.s(auto_attribs=True)
class TranscriptionData:
    output: Output


# Minimum silence (seconds) that counts as a real, dubbing-relevant pause.
# 0.25 s is the standard prosodic-phrase boundary: above the ~0.18 s stop-closure
# of voiceless consonants (so we don't split on articulation) yet low enough to
# keep every linguistically meaningful pause. Splitting here makes the silence a
# gap *between* segments, which is the only pause the downstream merge reinserts.
# NOTE: keep this in sync with the _remap_pauses default in speech_to_text_manager.py.
PAUSE_THRESHOLD_SECONDS = 0.25


def _group_words_into_segments(words: list[dict], pause_threshold: float = PAUSE_THRESHOLD_SECONDS) -> list[dict]:
    """
    Group word-level timestamps into segments by pause length and speaker changes.
    Returns list of dicts: {text, start, end, words, speaker}
    """
    if not words:
        return []

    segments = []
    current = [words[0]]

    for word in words[1:]:
        prev_end = current[-1].get("end", 0)
        cur_start = word.get("start", 0)
        split = (
            (cur_start - prev_end) >= pause_threshold
            or word.get("speaker", "SPEAKER_00") != current[-1].get("speaker", "SPEAKER_00")
        )
        if split:
            segments.append(_make_segment(current))
            current = [word]
        else:
            current.append(word)

    segments.append(_make_segment(current))
    return segments


def _make_segment(words: list[dict]) -> dict:
    return {
        "text": " ".join(w["word"] for w in words),
        "start": words[0].get("start", 0.0),
        "end": words[-1].get("end", 0.0),
        "words": words,
        "speaker": words[0].get("speaker", "SPEAKER_00"),
    }
