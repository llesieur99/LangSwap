from main import _build_parser


def test_cli_default_french_profile_keeps_narrator_identities():
    args = _build_parser().parse_args(["local", "in.mp4"])
    assert args.target_lang == "french"
    assert args.tts == "openvoice"
    assert args.translation == "marian"
    assert args.with_diarization is True


def test_cli_single_voice_and_explicit_preset_override_are_available():
    args = _build_parser().parse_args(
        [
            "local",
            "in.mp4",
            "--no-diarization",
            "--speakers",
            "1",
            "--speaker-voice",
            "SPEAKER_00=fr-CA-AntoineNeural",
        ]
    )
    assert args.with_diarization is False and args.speakers == 1
    assert args.speaker_voice == ["SPEAKER_00=fr-CA-AntoineNeural"]


def test_multiple_edge_narrators_require_explicit_voices(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import pytest
    from langswap.ml.text_to_speech_service.tts_edge_client import EdgeTTSClient

    monkeypatch.delenv("LANGSWAP_EDGE_VOICE", raising=False)
    video = SimpleNamespace(
        translated_texts=[
            SimpleNamespace(start=0.0, end=1.0, speaker="A"),
            SimpleNamespace(start=2.0, end=3.0, speaker="B"),
        ]
    )
    with pytest.raises(ValueError, match="every narrator"):
        EdgeTTSClient().prepare_speakers(video, str(tmp_path))
    client = EdgeTTSClient(
        speaker_voices={"A": "fr-CA-JeanNeural", "B": "fr-CA-SylvieNeural"}
    )
    client.prepare_speakers(video, str(tmp_path))
    assert client._voice_for("fr", "A") != client._voice_for("fr", "B")
