"""CPU regressions for voice consistency and honest synthesis failures."""
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from langswap.ml.text_to_speech_service import speaker_voices as voices
from langswap.ml.text_to_speech_service.tts_chatterbox_client import ChatterboxClient
from langswap.ml.text_to_speech_service.tts_edge_client import EdgeTTSClient
from langswap.ml.text_to_speech_service.tts_eleven_client import ElevenTTSClient
from langswap.ml.text_to_speech_service.tts_kokoro_client import KokoroClient
from langswap.ml.text_to_speech_service.tts_manager import TextToSpeechManager
from langswap.ml.text_to_speech_service.tts_omnivoice_client import OmniVoiceClient
from langswap.ml.text_to_speech_service.tts_openvoice_client import OpenVoiceClient
from langswap.ml.text_to_speech_service.tts_xtts_client import XttsClient
from langswap.pipeline_models.models import VideoTranslation


def segment(start=0, end=1, speaker="A", text="Hello", translation="Bonjour", source=None):
    return SimpleNamespace(start=start, end=end, speaker=speaker, text=text,
                           translation=translation, source_file=source, generated_file=None)


def wav(path, value=0.1, seconds=1, rate=16000):
    sf.write(path, np.full(int(rate * seconds), value, dtype=np.float32), rate)
    return str(path)


def job(segments, source=None):
    return SimpleNamespace(public_id="test", translated_texts=segments,
                           background_audio={"vocals.wav": source})


def test_references_exclude_competing_speech_and_use_safe_names(tmp_path):
    source = tmp_path / "vocals.wav"
    # A owns [0,2], B owns [1,3]; overlap [1,2] must enter neither reference.
    sf.write(source, np.concatenate([np.full(16000, value) for value in (.1, .8, .3)]), 16000)
    references = voices.build_speaker_references(str(source),
        [segment(0, 2, "../été"), segment(1, 3, "B")], str(tmp_path / "refs"))
    assert set(references) == {"../été", "B"}
    for speaker, expected in (("../été", .1), ("B", .3)):
        data, _ = sf.read(references[speaker])
        assert len(data) == 16000
        assert np.allclose(data, expected, atol=1e-4)
        assert str(tmp_path / "refs") in references[speaker]


def test_no_reference_fallback_to_another_speaker(tmp_path):
    source = wav(tmp_path / "source.wav", seconds=2)
    with pytest.raises(ValueError, match="No isolated audible reference"):
        voices.build_speaker_references(source, [segment(0, 2, "A"), segment(.5, 1.5, "B")],
                                        str(tmp_path / "refs"))


def test_transcript_references_skip_mixed_clips(tmp_path):
    source = wav(tmp_path / "source.wav")
    mixed_a, mixed_b = segment(0, 2, "A", source=source), segment(1, 3, "B", source=source)
    isolated_a, isolated_b = segment(4, 5, "A", source=source), segment(5, 6, "B", source=source)
    references = voices.isolated_reference_segments([mixed_a, mixed_b, isolated_a, isolated_b])
    assert references == {"A": [isolated_a], "B": [isolated_b]}
    with pytest.raises(ValueError, match="No isolated reference"):
        voices.isolated_reference_segments([mixed_a, mixed_b])


@pytest.mark.parametrize("kind", ["empty", "silent", "corrupt", "nan", "too-short", "cancelled-stereo"])
def test_audio_validation_rejects_invalid_output(tmp_path, kind):
    path = tmp_path / "invalid.wav"
    if kind == "corrupt":
        path.write_bytes(b"not audio")
    else:
        data = np.array([]) if kind == "empty" else np.zeros(1000)
        if kind == "nan":
            data[:] = np.nan
        if kind == "too-short":
            data = np.full(100, .1)
        if kind == "cancelled-stereo":
            data = np.column_stack((np.full(1000, .1), np.full(1000, -.1)))
        sf.write(path, data, 16000, subtype="FLOAT")
    with pytest.raises((ValueError, RuntimeError)):
        voices.validate_audio(str(path))


@pytest.mark.parametrize("changed", ["text", "language", "engine", "voice", "speaker", "audio"])
def test_segment_cache_invalidates_when_identity_or_audio_changes(tmp_path, changed):
    item = segment()
    render = Mock(side_effect=lambda path: wav(path))
    arguments = [item, str(tmp_path), "fr", "openvoice", "ref-hash", render]
    voices.synthesize_cached_segment(*arguments)
    voices.synthesize_cached_segment(*arguments)
    assert render.call_count == 1
    if changed == "text":
        item.translation = "Salut"
    elif changed == "language":
        arguments[2] = "en"
    elif changed == "engine":
        arguments[3] = "kokoro"
    elif changed == "voice":
        arguments[4] = "different-reference"
    elif changed == "speaker":
        item.speaker = "B"
    else:
        wav(item.generated_file, value=0)
    voices.synthesize_cached_segment(*arguments)
    assert render.call_count == 2


def test_failure_never_publishes_partial_or_silent_audio(tmp_path):
    item = segment()
    with pytest.raises(ValueError, match="Silent"):
        voices.synthesize_cached_segment(item, str(tmp_path), "fr", "kokoro", "voice",
                                         lambda path: wav(path, value=0))
    assert item.generated_file is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("metadata", ["null", "[]", "invalid json"])
def test_malformed_cache_metadata_is_regenerated(tmp_path, metadata):
    item = segment()
    output = tmp_path / "0_1.wav"
    wav(output)
    (tmp_path / "0_1.wav.tts.json").write_text(metadata)
    render = Mock(side_effect=lambda path: wav(path, value=.4))
    voices.synthesize_cached_segment(item, str(tmp_path), "fr", "kokoro", "voice", render)
    render.assert_called_once()


def test_empty_translation_is_not_silently_synthesized(tmp_path):
    with pytest.raises(ValueError, match="empty translation"):
        voices.synthesize_cached_segment(segment(translation="  "), str(tmp_path),
                                         "fr", "kokoro", "voice", Mock())


@pytest.mark.parametrize("client_class", [KokoroClient, XttsClient, ChatterboxClient])
def test_chunking_keeps_long_and_repeated_text_intact(client_class):
    text = "Bonjour tout le monde. " * 200
    chunks = client_class("cpu")._chunk_text(text)
    assert " ".join(chunks) == text.strip()
    assert all(len(chunk) <= 280 for chunk in chunks)


@pytest.mark.parametrize("client_class", [KokoroClient, XttsClient, ChatterboxClient])
def test_engine_errors_do_not_become_silent_success(tmp_path, monkeypatch, client_class):
    client = client_class("cpu")
    monkeypatch.setattr(client, "load_models", lambda: None)
    monkeypatch.setattr(client, "_tts_one", Mock(side_effect=RuntimeError("engine failed")))
    source = wav(tmp_path / "source.wav")
    output = tmp_path / "output.wav"
    with pytest.raises(RuntimeError):
        client.generate_audio("Bonjour", source, "Hello", str(output), "fr")
    assert not output.exists()


def test_kokoro_empty_generator_is_an_error(tmp_path):
    client = KokoroClient("cpu")
    client._pipe = lambda *args, **kwargs: iter([])
    with pytest.raises(RuntimeError, match="no audio"):
        client._tts_one("Bonjour", str(tmp_path / "output.wav"))


def test_kokoro_rejects_unsupported_language_without_loading_models(tmp_path, monkeypatch):
    client = KokoroClient("cpu")
    load = Mock()
    monkeypatch.setattr(client, "load_models", load)
    with pytest.raises(ValueError, match="French only"):
        client.generate_audio("Hello", "", "", str(tmp_path / "out.wav"), "en")
    load.assert_not_called()


def test_openvoice_unknown_speaker_never_uses_first_embedding(tmp_path, monkeypatch):
    client = OpenVoiceClient("cpu")
    client._se = {"A": object()}
    monkeypatch.setattr(client, "load_models", lambda: None)
    with pytest.raises(RuntimeError, match="speaker 'B'"):
        client.generate_audio("Bonjour", "", "", str(tmp_path / "out.wav"), "fr", speaker="B")


def test_openvoice_constructor_matches_strict_upstream_base_signature(tmp_path, monkeypatch):
    import sys

    from langswap.ml.text_to_speech_service import tts_openvoice_client as module
    client = OpenVoiceClient("cpu")
    monkeypatch.setattr(client._base, "load_models", lambda: None)
    monkeypatch.setattr(module, "_ensure_checkpoints", lambda: str(tmp_path))
    captured = {}
    class UpstreamBase:
        def __init__(self, config_path, device="cuda:0"):
            captured["config_path"] = config_path
            captured["device"] = device

    class UpstreamConverter(UpstreamBase):
        def __init__(self, *args, **kwargs):
            # The installed converter forwards kwargs to this strict base.
            # enable_watermark is inspected only after that incompatible call.
            super().__init__(*args, **kwargs)
            self.watermark_model = object()
            self.load_ckpt = Mock()

    api = ModuleType("openvoice.api")
    api.ToneColorConverter = UpstreamConverter
    monkeypatch.setitem(sys.modules, "openvoice.api", api)
    client.load_models()
    assert captured == {"config_path": str(tmp_path / "converter" / "config.json"), "device": "cpu"}
    client._conv.load_ckpt.assert_called_once_with(str(tmp_path / "converter" / "checkpoint.pth"))
    assert client._conv.watermark_model is None


def test_openvoice_conversion_error_never_changes_to_kokoro(tmp_path, monkeypatch):
    client = OpenVoiceClient("cpu")
    client._se = {"A": object()}
    monkeypatch.setattr(client, "load_models", lambda: None)
    monkeypatch.setattr(client._base, "generate_audio", lambda **kwargs: wav(kwargs["save_path"]))
    monkeypatch.setattr(client, "_convert", Mock(side_effect=RuntimeError("converter failed")))
    output = tmp_path / "out.wav"
    with pytest.raises(RuntimeError, match="synthesis failed"):
        client.generate_audio("Bonjour", "", "", str(output), "fr", speaker="A")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("client_class", [XttsClient, ChatterboxClient, OpenVoiceClient])
def test_clone_pipeline_reuses_one_reference_per_speaker(tmp_path, monkeypatch, client_class):
    client = client_class("cpu")
    source = tmp_path / "vocals.wav"
    sf.write(source, np.concatenate([np.full(16000, value) for value in (.1, .3, .1)]), 16000)
    items = [segment(0, 1, "A"), segment(1, 2, "B"), segment(2, 3, "A")]
    monkeypatch.setattr(client, "load_models", lambda: None)
    calls = []
    if isinstance(client, OpenVoiceClient):
        monkeypatch.setattr(client._base, "generate_audio", lambda **kwargs: wav(kwargs["save_path"]))
        extracted = Mock(side_effect=lambda path: object())
        monkeypatch.setattr(client, "_extract_se", extracted)
        source_embeddings = []
        def convert(src_wav, target, path):
            source_embeddings.append(client._source_se)
            wav(path)
        monkeypatch.setattr(client, "_convert", convert)
    else:
        monkeypatch.setattr(client, "generate_audio", lambda **kwargs:
            (calls.append((kwargs["speaker"], kwargs["source_audio_file"])), wav(kwargs["save_path"])))
    client.tts_pipeline(job(items, str(source)), str(tmp_path / "generated"), "fr")
    if isinstance(client, OpenVoiceClient):
        assert extracted.call_count == 3  # two speakers and one fixed Kokoro calibration
        assert len(source_embeddings) == 3
        assert all(embedding is source_embeddings[0] for embedding in source_embeddings)
    else:
        assert calls[0][1] == calls[2][1]
        assert calls[0][1] != calls[1][1]


def test_edge_explicit_per_speaker_voices_are_stable(monkeypatch):
    monkeypatch.setenv("LANGSWAP_EDGE_VOICE", "fr-CA-JeanNeural")
    client = EdgeTTSClient("cpu", {"B": "fr-CA-SylvieNeural"})
    assert client._voice_for("fr", "B") == client._voice_for("french", "B") == "fr-CA-SylvieNeural"
    assert client._voice_for("fr", "A") == "fr-CA-JeanNeural"


def test_elevenlabs_never_clones_multiple_speakers_together(tmp_path):
    client = ElevenTTSClient.__new__(ElevenTTSClient)
    client._preset_voice_id = None
    client.client = Mock()
    with pytest.raises(ValueError, match="exactly one speaker"):
        client.clone_voice(job([segment(speaker="A"), segment(speaker="B")]))
    client.client.voices.ivc.create.assert_not_called()


def test_elevenlabs_cloned_voice_is_deleted_after_synthesis_error(tmp_path, monkeypatch):
    client = ElevenTTSClient.__new__(ElevenTTSClient)
    client._preset_voice_id = None
    client._source_groups = {}
    client.client = Mock()
    source = wav(tmp_path / "source.wav")
    monkeypatch.setattr(client, "clone_voice", Mock(return_value=(SimpleNamespace(voice_id="A"), True)))
    monkeypatch.setattr(client, "generate_audio", Mock(side_effect=RuntimeError("TTS failed")))
    with pytest.raises(RuntimeError, match="TTS failed"):
        client.tts_pipeline(job([segment(source=source)]), str(tmp_path / "out"))
    client.client.voices.delete.assert_called_once_with(voice_id="A")


def test_elevenlabs_short_reference_error_is_explicit_and_handles_close(tmp_path):
    client = ElevenTTSClient.__new__(ElevenTTSClient)
    client._preset_voice_id = None
    client.client = Mock()
    handles = []
    def fail_clone(**kwargs):
        handles.extend(kwargs["files"])
        raise RuntimeError("voice_too_short")
    client.client.voices.ivc.create.side_effect = fail_clone
    with pytest.raises(RuntimeError, match="explicit preset"):
        client.clone_voice(job([segment(source=wav(tmp_path / "source.wav"))]))
    assert handles and all(handle.closed for handle in handles)
    client.client.voices.get_all.assert_not_called()


def test_omnivoice_uses_fixed_matching_audio_and_transcript_per_speaker(tmp_path, monkeypatch):
    client = OmniVoiceClient.__new__(OmniVoiceClient)
    a = wav(tmp_path / "a.wav", .1)
    b = wav(tmp_path / "b.wav", .3)
    a_short = wav(tmp_path / "a-short.wav", .5)
    items = [segment(0, 2, "A", text="Original A", source=a),
             segment(2, 4, "B", text="Original B", source=b),
             segment(4, 5, "A", text="Short A", source=a_short)]
    calls = []
    def render(**kwargs):
        calls.append((kwargs["source_audio_file"], kwargs["source_text"]))
        wav(kwargs["save_path"])
    monkeypatch.setattr(client, "generate_audio", render)
    client.tts_pipeline(job(items), str(tmp_path / "generated"), "fr")
    assert calls == [(a, "Original A"), (b, "Original B"), (a, "Original A")]


def test_manager_edit_keeps_speaker_and_reference(tmp_path):
    manager = TextToSpeechManager.__new__(TextToSpeechManager)
    manager._file_repository = SimpleNamespace(subdir=lambda _: str(tmp_path / "generated"))
    manager.tts_name = "openvoice"
    a, b = wav(tmp_path / "a.wav", .1), wav(tmp_path / "b.wav", .3)
    calls = []
    def render(**kwargs):
        calls.append(kwargs)
        wav(kwargs["save_path"])
    manager._tts_client = SimpleNamespace(_refs={"A": a, "B": b},
        _base=SimpleNamespace(voice="ff_siwis"), generate_audio=render)
    manager.synthesize_segment(segment(speaker="B"), "fr", "unused-vocals.wav")
    assert calls[0]["speaker"] == "B"
    assert calls[0]["source_audio_file"] == b


@pytest.mark.parametrize("operation", ["prepare_speakers", "synthesize"])
def test_duplicate_intervals_are_rejected_before_loading_or_writing(tmp_path, operation):
    manager = TextToSpeechManager.__new__(TextToSpeechManager)
    manager._tts_client = Mock()
    manager._file_repository = Mock()
    video = job([segment(0, 1, "A"), segment(0.0, 1.0, "B")])
    with pytest.raises(ValueError, match="would overwrite another segment"):
        if operation == "synthesize":
            manager.synthesize(video, "en", "fr")
        else:
            manager.prepare_speakers(video)
    manager._tts_client.prepare_speakers.assert_not_called()
    manager._file_repository.subdir.assert_not_called()


def test_manager_does_not_rewrite_vocals_and_refreshes_source_clips(tmp_path):
    source = wav(tmp_path / "vocals.wav", .1, rate=16000)
    manager = TextToSpeechManager.__new__(TextToSpeechManager)
    def subdir(name):
        folder = tmp_path / name
        folder.mkdir(exist_ok=True)
        return str(folder)
    manager._file_repository = SimpleNamespace(subdir=subdir, save_dir=lambda _: None)
    manager.tts_sample_rate = 24000
    manager.logger = Mock()
    manager._tts_client = SimpleNamespace(tts_pipeline=lambda video, *args, **kwargs: video)
    processed = object()
    video = VideoTranslation(public_id="test", translated_texts=[segment()],
                             background_audio={"vocals.wav": source},
                             source_lang_code="en", processed_video=processed)
    original = voices.file_digest(source)
    result = manager.synthesize(video, "english", "french")
    assert result is not video
    assert result.source_lang_code == "en"
    assert result.processed_video is processed
    assert voices.file_digest(source) == original
    clip = video.translated_texts[0].source_file
    data, rate = sf.read(clip)
    assert rate == 24000
    assert np.allclose(data, .1, atol=1e-4)
    wav(source, .4, rate=16000)
    manager.synthesize(video, "english", "french")
    data, rate = sf.read(clip)
    assert np.allclose(data, .4, atol=1e-4)


@pytest.mark.parametrize("engine", ["openvoice", "xtts", "chatterbox", "omnivoice"])
def test_mutable_voice_clients_are_scoped_to_job(engine, monkeypatch):
    from langswap import model_pool
    module_name = f"langswap.ml.text_to_speech_service.tts_{engine}_client"
    class_names = {"openvoice": "OpenVoiceClient", "xtts": "XttsClient",
                   "chatterbox": "ChatterboxClient", "omnivoice": "OmniVoiceClient"}
    monkeypatch.setattr(f"{module_name}.{class_names[engine]}",
                        lambda **kwargs: SimpleNamespace(_refs={}, sample_rate=24000))
    monkeypatch.setattr(model_pool, "_POOL", {})
    repository = Mock()
    first = TextToSpeechManager("first", repository, 24000, Mock(), "cpu", engine)
    second = TextToSpeechManager("second", repository, 24000, Mock(), "cpu", engine)
    first_again = TextToSpeechManager("first", repository, 24000, Mock(), "cpu", engine)
    assert first._tts_client is first_again._tts_client
    assert first._tts_client is not second._tts_client
    first._tts_client._refs["A"] = "original-voice"
    assert second._tts_client._refs == {}
