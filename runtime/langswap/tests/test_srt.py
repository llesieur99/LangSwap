"""SRT generation test.

Guarded for the full GPU/api venv: importing the pipeline pulls in pyrubberband
and silero_vad at module top, which are absent in the lean test env, so this
skips cleanly here and runs in production.
"""
import pytest

pytest.importorskip("pyrubberband")
pytest.importorskip("silero_vad")

from langswap.translation_pipeline import VideoTranslationPipeline  # noqa: E402
from langswap.file_repository import LocalOnlyFileRepository  # noqa: E402
from langswap.pipeline_models.models import (  # noqa: E402
    VideoTranslation,
    TextedSegment,
    TranslatedTextedSegment,
)

SRT_TIME = r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"


def test_generate_srt_files(tmp_path):
    repo = LocalOnlyFileRepository("job1", str(tmp_path))

    recognized = [
        TextedSegment(text="Hello world", start=0.0, end=1.5, speaker="SPEAKER_00"),
        TextedSegment(text="How are you", start=2.0, end=3.25, speaker="SPEAKER_00"),
    ]
    translated = [
        TranslatedTextedSegment(
            text="Hello world", start=0.0, end=1.5, translation="Privet mir",
            source_file=None, generated_file=None, speaker="SPEAKER_00",
        ),
        TranslatedTextedSegment(
            text="How are you", start=2.0, end=3.25, translation="Kak dela",
            source_file=None, generated_file=None, speaker="SPEAKER_00",
        ),
    ]

    pipeline = object.__new__(VideoTranslationPipeline)
    pipeline._file_repository = repo
    pipeline.video_translation = VideoTranslation(
        public_id="job1",
        recognized_texts=recognized,
        translated_texts=translated,
    )

    source_srt, translated_srt = pipeline.generate_srt_files()

    import re

    source_text = open(source_srt.file_path, encoding="utf-8").read()
    translated_text = open(translated_srt.file_path, encoding="utf-8").read()

    # Correctly formatted SRT timestamps.
    assert re.search(SRT_TIME, source_text)
    assert "00:00:00,000 --> 00:00:01,500" in source_text
    assert "00:00:02,000 --> 00:00:03,250" in source_text

    # Source SRT carries the recognized text, translated SRT the translation.
    assert "Hello world" in source_text
    assert "How are you" in source_text
    assert "Privet mir" in translated_text
    assert "Kak dela" in translated_text
