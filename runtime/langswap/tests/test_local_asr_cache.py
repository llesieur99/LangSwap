"""Source and speaker options must invalidate cached ASR without loading models."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from langswap.file_repository import LocalOnlyFileRepository
from langswap.ml.speech_to_text_service.speech_to_text_manager import SpeechToTextManager


def _manager(tmp_path, monkeypatch):
    repo = LocalOnlyFileRepository("job", str(tmp_path))
    logger = Mock()

    def log_json(file_name, data):
        with open(repo.get_file(file_name).file_path, "w", encoding="utf-8") as output:
            json.dump(data, output)

    logger.log_json.side_effect = log_json
    manager = SpeechToTextManager("en", "job", repo, "cpu", logger)
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.transcribe.return_value = SimpleNamespace(
        detected_language="en", segments=[SimpleNamespace(
            text="first", start=0, end=1, speaker="SPEAKER_00")])
    monkeypatch.setattr(manager, "_ensure_asr_client", lambda: setattr(manager, "_asr_client", client))
    source = tmp_path / "source.wav"
    source.write_bytes(b"old audio")
    return manager, client, source


@pytest.mark.parametrize("change", ["source", "language", "backend", "skip_diarization", "num_speakers"])
def test_transcription_cache_tracks_source_and_speaker_options(tmp_path, monkeypatch, change):
    manager, client, source = _manager(tmp_path, monkeypatch)
    manager._load_or_transcribe_audio(str(source), num_speakers=1)
    manager._load_or_transcribe_audio(str(source), num_speakers=1)
    assert client.transcribe.call_count == 1
    if change == "source":
        source.write_bytes(b"new audio")  # same path and size, different content
    elif change == "language":
        manager._language = "fr"
    elif change == "backend":
        manager._backend = "openai"
    elif change == "skip_diarization":
        manager._skip_diarization = True
    manager._load_or_transcribe_audio(str(source), num_speakers=2 if change == "num_speakers" else 1)
    assert client.transcribe.call_count == 2


def test_legacy_asr_cache_without_identity_is_not_reused(tmp_path, monkeypatch):
    manager, client, source = _manager(tmp_path, monkeypatch)
    manager.logger.log_json("raw_transcribed_info.json", [dict(text="stale", start=0, end=1, speaker="SPEAKER_99")])
    manager.logger.log_json("lang_detect_info.json", dict(detected_language="fr"))
    segments, language = manager._load_or_transcribe_audio(str(source))
    assert client.transcribe.call_count == 1
    assert segments[0]["text"] == "first"
    assert language == "en"


def test_remap_cache_tracks_speaker_assignments(tmp_path, monkeypatch):
    manager, _, _ = _manager(tmp_path, monkeypatch)
    segments = [dict(text="first", start=0, end=1, speaker="SPEAKER_00")]
    assert manager._load_or_remap_segments(segments)[0].speaker == "SPEAKER_00"
    segments[0]["speaker"] = "SPEAKER_01"
    assert manager._load_or_remap_segments(segments)[0].speaker == "SPEAKER_01"


def test_asr_failure_joins_demucs_before_releasing_manager_holder(tmp_path, monkeypatch):
    import threading

    manager, _, source = _manager(tmp_path, monkeypatch)
    released = threading.Event()
    finished = threading.Event()
    holder = object()
    manager._asr_client = holder
    monkeypatch.setattr(manager, "_get_audio_file", lambda video: str(source))

    def separate(*args):
        assert released.wait(timeout=5)
        finished.set()
        return {}

    def fail(*args):
        released.set()
        raise RuntimeError("ASR failed")

    monkeypatch.setattr(manager, "_separate_vocals_and_background", separate)
    monkeypatch.setattr(manager, "_load_or_transcribe_audio", fail)
    video = SimpleNamespace(source_file=SimpleNamespace(file_path=str(source)))
    with pytest.raises(RuntimeError, match="ASR failed"):
        manager.extract_and_transcribe(video)
    assert finished.is_set()
    assert manager._asr_client is None


def test_video_audio_extraction_does_not_limit_the_source_to_sixty_seconds(tmp_path, monkeypatch):
    from langswap.ml.speech_to_text_service import speech_to_text_manager

    manager, _, _ = _manager(tmp_path, monkeypatch)
    ffmpeg = Mock()
    ffmpeg.extract_audio.return_value = (b"", b"")
    monkeypatch.setattr(speech_to_text_manager, "FFmpegClient", lambda: ffmpeg)
    result = manager._extract_audio("long-video.mp4")
    ffmpeg.extract_audio.assert_called_once_with(
        "long-video.mp4", result.file_path, time_limit=None)
