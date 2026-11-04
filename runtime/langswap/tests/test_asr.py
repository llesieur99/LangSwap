"""
Tests for ASR (Automatic Speech Recognition) components.
Following the albumentations testing pattern with fixtures and parametrization.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langswap.ml.speech_to_text_service.asr_types import Segment, Output, TranscriptionData
from langswap.ml.speech_to_text_service.asr_vad_client import _group_words_by_vad
from langswap.ml.speech_to_text_service.speech_to_text_manager import SpeechToTextManager
from langswap.pipeline_models.models import TextedSegment


def _word(text, start, end, speaker="SPEAKER_00"):
    return {"word": text, "start": start, "end": end, "speaker": speaker}


class TestGroupWordsByVad:
    """_group_words_by_vad assigns whisper words to Silero VAD speech regions.

    Pure logic (no silero-vad / faster-whisper needed): runs in the lean env.
    """

    def test_words_grouped_by_midpoint(self):
        """Each word lands in the region containing its midpoint; segment
        boundaries come from the VAD region, text from whisper."""
        words = [
            _word("hello", 0.0, 0.4),   # mid 0.2 -> region 0
            _word("there", 0.5, 0.9),   # mid 0.7 -> region 0
            _word("world", 2.1, 2.5),   # mid 2.3 -> region 1
        ]
        regions = [(0.0, 1.0), (2.0, 3.0)]
        segments = _group_words_by_vad(words, regions)
        assert len(segments) == 2
        assert segments[0]["text"] == "hello there"
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 1.0
        assert segments[1]["text"] == "world"
        assert segments[1]["start"] == 2.0
        assert segments[1]["end"] == 3.0

    def test_word_in_gap_assigned_to_nearest_region(self):
        """A word whose midpoint falls in VAD silence goes to the nearest region,
        so no transcript text is lost."""
        words = [
            _word("hi", 0.0, 0.4),      # mid 0.2 -> region 0
            _word("gap", 1.3, 1.5),     # mid 1.4 -> nearest is region 0 (d=0.4) over region 1 (d=0.6)
            _word("bye", 2.1, 2.5),     # mid 2.3 -> region 1
        ]
        regions = [(0.0, 1.0), (2.0, 3.0)]
        segments = _group_words_by_vad(words, regions)
        rendered = " ".join(s["text"] for s in segments)
        assert "hi" in rendered and "gap" in rendered and "bye" in rendered
        assert segments[0]["text"] == "hi gap"
        assert segments[1]["text"] == "bye"

    def test_empty_regions_fallback_single_segment(self):
        """No VAD regions -> a single fallback segment spanning the words."""
        words = [
            _word("only", 0.5, 0.9),
            _word("speech", 1.0, 1.4),
        ]
        segments = _group_words_by_vad(words, [])
        assert len(segments) == 1
        assert segments[0]["text"] == "only speech"
        assert segments[0]["start"] == 0.5
        assert segments[0]["end"] == 1.4

    def test_empty_words(self):
        """No words -> no segments, with or without regions."""
        assert _group_words_by_vad([], [(0.0, 1.0)]) == []
        assert _group_words_by_vad([], []) == []



class TestSegment:
    """Test Segment data class."""

    def test_segment_creation(self):
        """Test basic segment creation."""
        segment = Segment(
            start=0.0,
            end=1.0,
            text="test text",
            words=[],
            speaker="SPEAKER_00"
        )
        assert segment.start == 0.0
        assert segment.end == 1.0
        assert segment.text == "test text"
        assert segment.speaker == "SPEAKER_00"

    @pytest.mark.parametrize("start,end,text", [
        (0.0, 1.0, "short"),
        (1.5, 3.2, "longer test sentence"),
        (10.0, 15.5, ""),
    ])
    def test_segment_parametrized(self, start, end, text):
        """Test segment creation with different parameters."""
        segment = Segment(start=start, end=end, text=text, words=[])
        assert segment.start == start
        assert segment.end == end
        assert segment.text == text


class TestOutput:
    """Test Output data class."""

    def test_output_creation(self):
        """Test basic output creation."""
        segments = [
            Segment(start=0.0, end=1.0, text="test", words=[])
        ]
        output = Output(
            detected_language="en",
            device="cpu",
            model="whisper",
            transcription="test transcription",
            segments=segments
        )
        assert output.detected_language == "en"
        assert output.device == "cpu"
        assert len(output.segments) == 1


class TestTranscriptionDataStructures:
    """Test transcription data structures and conversions."""

    def test_transcription_data_creation(self):
        """Test TranscriptionData creation."""
        output = Output(
            detected_language="en",
            device="cpu",
            model="whisper",
            transcription="test",
            segments=[]
        )
        data = TranscriptionData(output=output)
        assert data.output == output

    def test_segment_with_speaker_assignment(self):
        """Test segment creation with speaker assignment."""
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
            {"word": "world", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"}
        ]
        segment = Segment(
            start=0.0,
            end=1.0,
            text="hello world",
            words=words,
            speaker="SPEAKER_00"
        )
        assert segment.speaker == "SPEAKER_00"
        assert len(segment.words) == 2
        assert all(word["speaker"] == "SPEAKER_00" for word in segment.words)


class TestRemapPauses:
    """Test SpeechToTextManager._remap_pauses segment merging.

    _remap_pauses never touches `self`, so we exercise it as the pure function
    it is by passing None as the bound instance — this avoids constructing the
    Manager (which would eagerly build a heavy ASR backend).
    """

    @staticmethod
    def _remap(entries, **kwargs):
        return SpeechToTextManager._remap_pauses(None, entries, **kwargs)

    def test_merges_on_short_pause(self):
        """Words separated by a sub-threshold gap collapse into one segment."""
        entries = [
            {"text": "one", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"text": "two", "start": 1.1, "end": 4.0, "speaker": "SPEAKER_00"},  # gap 0.1 < 0.25
        ]
        result = self._remap(entries)
        assert len(result) == 1
        assert isinstance(result[0], TextedSegment)
        assert result[0].text == "one two"
        assert result[0].start == 0.0
        assert result[0].end == 4.0

    def test_splits_on_long_pause(self):
        """A gap >= pause_threshold ends the current segment."""
        entries = [
            {"text": "one", "start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"text": "two", "start": 3.5, "end": 6.5, "speaker": "SPEAKER_00"},  # gap 0.5 >= 0.25
        ]
        result = self._remap(entries)
        assert len(result) == 2
        assert result[0].text == "one"
        assert result[1].text == "two"
        assert result[1].start == 3.5

    def test_splits_on_speaker_change(self):
        """A speaker change ends the current segment even with no pause."""
        entries = [
            {"text": "one", "start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"text": "two", "start": 3.0, "end": 6.0, "speaker": "SPEAKER_01"},
        ]
        result = self._remap(entries)
        assert len(result) == 2
        assert result[0].speaker == "SPEAKER_00"
        assert result[1].speaker == "SPEAKER_01"

    def test_skips_non_text_entries(self):
        """Entries without alphabetic characters are dropped, not merged in."""
        entries = [
            {"text": "one", "start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"text": "...", "start": 3.1, "end": 3.3, "speaker": "SPEAKER_00"},  # no letters
            {"text": "two", "start": 4.0, "end": 7.0, "speaker": "SPEAKER_00"},  # gap from 3.0 >= 0.25
        ]
        result = self._remap(entries)
        assert len(result) == 2
        assert all("..." not in seg.text for seg in result)
        assert result[0].text == "one"
        assert result[1].text == "two"

    def test_short_trailing_segment_folds_into_previous(self):
        """A short (<3s) final segment is merged into its predecessor and removed.

        Regression: the trailing-segment merge used to fold the short last
        segment's text/end into pairs[-2] without dropping pairs[-1], so the
        last text appeared twice (e.g. ["one two", "two"]).
        """
        entries = [
            {"text": "one", "start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"text": "two", "start": 3.5, "end": 4.5, "speaker": "SPEAKER_00"},  # gap 0.5 split; 1.0s < 3
        ]
        result = self._remap(entries)
        assert len(result) == 1
        assert result[0].text == "one two"
        assert result[0].start == 0.0
        assert result[0].end == 4.5