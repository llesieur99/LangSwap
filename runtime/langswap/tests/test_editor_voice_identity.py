"""Editing a narrator validates all voice caches after refreshing source slices."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from langswap import editor
from langswap.file_repository import LocalOnlyFileRepository
from langswap.pipeline_models.models import (
    TextedSegment,
    TranslatedTextedSegment,
    TranslationPipelineConfig,
    VideoTranslation,
)
from langswap.utils.logging import Logger


def session(tmp_path):
    repo = LocalOnlyFileRepository("edit-test", str(tmp_path))
    vt = VideoTranslation(
        public_id="edit-test", background_audio={"vocals.wav": "vocals.wav"}
    )
    for start, end, speaker, text in [(0.0, 1.0, "A", "one"), (2.0, 3.0, "B", "two")]:
        src = tmp_path / "edit-test" / "splitted_audio" / f"{start}_{end}.wav"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"existing source clip")
        vt.recognized_texts.append(TextedSegment(text, start, end, speaker))
        vt.translated_texts.append(
            TranslatedTextedSegment(
                text, start, end, text, str(src), "old.wav", speaker
            )
        )
    cfg = TranslationPipelineConfig(
        "english", "french", "source.mp4", str(tmp_path), "edit-test", device="cpu"
    )
    pipeline = SimpleNamespace(
        _file_repository=repo,
        logger=Logger(repo.directory),
        _release_models=Mock(),
        _merge=Mock(return_value=vt),
        generate_srt_files=Mock(),
        video_translation=vt,
    )
    return editor.EditorSession(
        "edit-test", str(tmp_path), repo.directory, cfg, pipeline, vt
    )


def fake_tts(monkeypatch, events):
    class TTS:
        tts_sample_rate = 24000

        def __init__(self, *args, **kwargs):
            pass

        def prepare_speakers(self, vt):
            from pathlib import Path

            assert all(Path(s.source_file).is_file() for s in vt.translated_texts)
            events.append("references")

        def synthesize_segment(self, seg, **kwargs):
            events.append((seg.speaker, seg.translation))

    monkeypatch.setattr(editor, "TextToSpeechManager", TTS)


def test_speaker_edit_redubs_and_validates_unchanged_person(monkeypatch, tmp_path):
    current = session(tmp_path)
    events = []
    fake_tts(monkeypatch, events)
    editor.apply_edits(
        current, [[1, 0, 1, "B", "one", "one"], [2, 2, 3, "B", "two", "two"]]
    )
    assert events == ["references", ("B", "one"), ("B", "two")]
    current.pipeline._merge.assert_called_once()


def test_new_segment_source_is_cut_before_reference_preparation(monkeypatch, tmp_path):
    current = session(tmp_path)
    events = []
    fake_tts(monkeypatch, events)

    def cut(self, vt, vocals, folder, **kwargs):
        from pathlib import Path

        events.append("source slices")
        for s in vt.translated_texts:
            Path(s.source_file).write_bytes(b"new source slice")
        return vt

    monkeypatch.setattr(editor.AudioDubbingManager, "split_audio_seconds", cut)
    editor.apply_edits(
        current,
        [
            [1, 0, 1, "A", "one", "one"],
            [2, 2, 3, "B", "two", "two"],
            ["", 4, 5, "A", "three", "trois"],
        ],
    )
    assert events[:2] == ["source slices", "references"]
    assert ("A", "trois") in events


@pytest.mark.parametrize(
    "job_id", ["../outside", "..", "other/job", "other\\job", "C:elsewhere"]
)
def test_editor_cannot_open_job_outside_data_directory(tmp_path, job_id):
    with pytest.raises(ValueError):
        editor.load_job(str(tmp_path), job_id)


@pytest.mark.parametrize("timestamp", [float("inf"), float("nan")])
def test_editor_rejects_nonfinite_time_before_writing(tmp_path, timestamp):
    current = session(tmp_path)
    with pytest.raises(ValueError):
        editor.apply_edits(current, [[1, 0, timestamp, "A", "one", "one"]])
    assert not current.pipeline._merge.called
