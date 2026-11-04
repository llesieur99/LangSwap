"""CPU regressions for music-bridged narration and speaker transitions."""
from langswap.ml.speech_to_text_service.asr_vad_client import _group_words_by_vad
from langswap.ml.speech_to_text_service.speech_to_text_manager import SpeechToTextManager
from langswap.ml.speech_to_text_service.asr_types import _group_words_into_segments


def _word(text, start, end, speaker="SPEAKER_00"):
    return dict(word=text, start=start, end=end, speaker=speaker)


def test_music_bridged_narration_keeps_separate_timeline_slots():
    # Music makes VAD see one long region although speech pauses for 28 seconds.
    words = [_word("first", 0, 2), _word("second", 30, 32)]
    segments = _group_words_by_vad(words, [(0, 120)])
    assert [(s["text"], s["start"], s["end"]) for s in segments] == [
        ("first", 0.0, 2.0), ("second", 30.0, 32.0),
    ]


def test_one_music_region_keeps_both_speakers():
    words = [_word("hello", 0, 1, "SPEAKER_00"),
             _word("reply", 1.1, 2, "SPEAKER_01")]
    segments = _group_words_by_vad(words, [(0, 10)])
    assert [(s["text"], s["speaker"]) for s in segments] == [
        ("hello", "SPEAKER_00"), ("reply", "SPEAKER_01"),
    ]


def test_long_narration_splits_at_sentence_pause():
    words = [_word("begin", 0, 5), _word("continue", 5.1, 15),
             _word("next", 15.4, 17)]
    segments = _group_words_by_vad(words, [(0, 30)])
    assert [s["text"] for s in segments] == ["begin continue", "next"]
    assert segments[1]["start"] == 15.4


def test_unbroken_long_sentence_does_not_lose_words():
    words = [_word("begin", 0, 10), _word("continue", 10.1, 20),
             _word("end", 20.1, 25)]
    segments = _group_words_by_vad(words, [(0, 30)])
    assert len(segments) == 1
    assert segments[0]["text"] == "begin continue end"


def test_gap_boundary_keeps_narration_in_order():
    words = [_word("before", 0, 1), _word("after", 2.5, 3)]
    segments = _group_words_by_vad(words, [(0, 10)])
    assert [s["text"] for s in segments] == ["before", "after"]
    assert segments[1]["start"] == 2.5


def test_no_vad_keeps_speaker_turns_and_long_pauses():
    words = [_word("one", 0, 1), _word("reply", 1, 2, "SPEAKER_01"),
             _word("return", 30, 31)]
    segments = _group_words_by_vad(words, [])
    assert [(s["text"], s["speaker"], s["start"], s["end"]) for s in segments] == [
        ("one", "SPEAKER_00", 0, 1), ("reply", "SPEAKER_01", 1, 2),
        ("return", "SPEAKER_00", 30, 31),
    ]


def test_unsplit_vad_region_does_not_clip_words_outside_its_edges():
    segments = _group_words_by_vad([_word("edge", 0.5, 1.5)], [(0.7, 1.2)])
    assert (segments[0]["start"], segments[0]["end"]) == (0.5, 1.5)


def test_single_utterance_does_not_inherit_music_padding():
    segments = _group_words_by_vad([_word("speech", 20, 22)], [(0, 120)])
    assert (segments[0]["start"], segments[0]["end"]) == (20, 22)


def test_short_trailing_reply_keeps_its_speaker():
    result = SpeechToTextManager._remap_pauses(None, [
        dict(text="question", start=0, end=3, speaker="SPEAKER_00"),
        dict(text="yes", start=3, end=3.5, speaker="SPEAKER_01"),
    ])
    assert [(s.text, s.speaker) for s in result] == [
        ("question", "SPEAKER_00"), ("yes", "SPEAKER_01"),
    ]


def test_short_trailing_utterance_does_not_bridge_long_pause():
    result = SpeechToTextManager._remap_pauses(None, [
        dict(text="before", start=0, end=1, speaker="SPEAKER_00"),
        dict(text="after", start=30, end=31, speaker="SPEAKER_00"),
    ])
    assert [(s.start, s.end) for s in result] == [(0, 1), (30, 31)]


def test_remap_empty_and_punctuation_only_transcript():
    assert SpeechToTextManager._remap_pauses(None, []) == []
    assert SpeechToTextManager._remap_pauses(None, [
        dict(text="...", start=0, end=1, speaker="SPEAKER_00")]) == []


def test_remap_keeps_numeric_speech_and_filters_initial_punctuation():
    result = SpeechToTextManager._remap_pauses(None, [
        dict(text="...", start=0, end=0.2, speaker="SPEAKER_01"),
        dict(text="2026", start=1, end=2, speaker="SPEAKER_00"),
    ])
    assert [(s.text, s.start, s.speaker) for s in result] == [("2026", 1, "SPEAKER_00")]


def test_api_word_grouping_preserves_exact_pause_threshold():
    result = _group_words_into_segments([_word("one", 0, 1), _word("two", 1.25, 2)])
    assert [s["text"] for s in result] == ["one", "two"]
