"""Speaker identity regressions using CPU embeddings and an injected encoder."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from langswap.ml.speech_to_text_service import embed_diarize
from langswap.ml.speech_to_text_service.asr_vad_client import VADWhisperASR


def test_returning_speaker_keeps_id_despite_embedding_variation():
    embeddings = np.array([[1, 0, 0], [0.98, 0.12, 0], [0, 1, 0],
                           [1, 0.02, 0], [0, 0.95, 0.1]])
    assert embed_diarize._cluster(embeddings).tolist() == [0, 0, 1, 0, 1]


def test_short_distinct_reply_is_not_smoothed_into_narrator():
    embeddings = np.array([[1, 0], [0, 1], [1, 0]])
    assert embed_diarize._cluster(embeddings).tolist() == [0, 1, 0]


def test_mixed_transition_window_cannot_create_an_unsupported_identity():
    embeddings = np.array([[1, 0, 0], [0.98, 0.1, 0], [0, 0, 1],
                           [0, 1, 0], [0.1, 0.98, 0]], dtype=np.float32)
    centers = embed_diarize._anchor_centroids(embeddings, [False] * 5)
    assert len(centers) == 2
    assert np.max(centers[:, 2]) == 0


def test_complete_short_utterance_remains_a_distinct_person():
    embeddings = np.array([[1, 0, 0], [0.98, 0.1, 0], [0, 0, 1],
                           [0, 1, 0], [0.1, 0.98, 0]], dtype=np.float32)
    centers = embed_diarize._anchor_centroids(embeddings, [False, False, True, False, False])
    assert len(centers) == 3
    assert np.max(centers[:, 2]) == 1
    assert len(embed_diarize._anchor_centroids(embeddings, [False] * 5, num_speakers=3)) == 3


def test_returning_voice_variants_share_one_sustained_identity():
    # Synthetic geometry reproduces the measured failure without publishing
    # any real speaker features: average pair distance is 0.496, whereas the
    # sustained prototypes of these two variants have distance 0.4 (< 0.42).
    first = np.array([1, 0, 0, 0, 0], dtype=np.float32)
    returning = np.array([0.6, 0.8, 0, 0, 0], dtype=np.float32)
    distinct = np.array([0.36, 0.18, 0, 0, np.sqrt(0.838)], dtype=np.float32)
    noise_first = np.array([0, 0, 0.4, 0, 0], dtype=np.float32)
    noise_returning = np.array([0, 0, 0, 0.4, 0], dtype=np.float32)
    scale = np.sqrt(0.84)
    embeddings = np.stack([
        distinct, distinct,
        scale * first + noise_first, scale * first - noise_first,
        scale * returning + noise_returning, scale * returning - noise_returning,
        scale * returning + noise_returning, scale * returning - noise_returning,
    ])
    whole = [False] * len(embeddings)
    assert len(set(embed_diarize._cluster(embeddings))) == 3
    centers = embed_diarize._anchor_centroids(embeddings, whole)
    assert len(centers) == 2
    assigned = np.argmax(embeddings @ centers.T, axis=1)
    assert len(set(assigned[:2])) == 1  # distinct person
    assert len(set(assigned[2:])) == 1  # same person's returning phonetic variants
    assert assigned[0] != assigned[2]
    merged_voice = centers[assigned[2]]
    # Four returning windows provide more evidence than the two first windows.
    assert merged_voice @ returning > merged_voice @ first
    assert np.linalg.norm(merged_voice) == pytest.approx(1)
    # A caller-provided known count remains explicit rather than being merged.
    assert len(embed_diarize._anchor_centroids(embeddings, whole, num_speakers=3)) == 3


def test_auto_speaker_count_has_no_arbitrary_eight_speaker_cap():
    assert embed_diarize._cluster(np.eye(9)).tolist() == list(range(9))


def test_explicit_count_handles_two_windows_and_distance_ties():
    assert embed_diarize._cluster(np.eye(2), num_speakers=2).tolist() == [0, 1]
    assert len(set(embed_diarize._cluster(np.eye(4), num_speakers=3))) == 3
    with pytest.raises(ValueError, match="num_speakers"):
        embed_diarize._cluster(np.eye(2), num_speakers=3)


class _Encoder:
    def __init__(self):
        self.calls = []

    def encode_batch(self, audio):
        self.calls.append(audio)
        # Two overlapping windows carry distinct speaker embeddings.
        return torch.tensor([[[1.0, 0.0] if len(self.calls) == 1 else [0.0, 1.0]]])


def test_word_selects_nearest_window_center_and_does_not_smooth_turn(monkeypatch):
    encoder = _Encoder()
    monkeypatch.setattr(embed_diarize, "_classifier", lambda device: encoder)
    words = [dict(word="one", start=0.1, end=0.6),
             dict(word="reply", start=1, end=1.2)]
    embed_diarize.assign_word_speakers(words, np.full(26400, 0.1, dtype=np.float32), device="cpu")
    assert [w["speaker"] for w in words] == ["SPEAKER_00", "SPEAKER_01"]


def test_music_only_windows_are_not_encoded(monkeypatch):
    encoder = _Encoder()
    monkeypatch.setattr(embed_diarize, "_classifier", lambda device: encoder)
    words = [dict(word="speech", start=0.1, end=0.6)]
    embed_diarize.assign_word_speakers(words, np.full(160000, 0.1, dtype=np.float32), device="cpu")
    assert len(encoder.calls) == 1  # remaining windows contain only loud music


def test_returning_person_uses_sustained_reference_not_phonetic_window_ids(monkeypatch):
    class Encoder:
        def encode_batch(self, audio):
            # Loudness serves as the identity feature of this injected encoder.
            second = audio.mean() > 0.15
            return torch.tensor([[[0.0, 1.0] if second else [1.0, 0.0]]])

    monkeypatch.setattr(embed_diarize, "_classifier", lambda device: Encoder())
    audio = np.full(16 * 16000, 0.1, dtype=np.float32)
    audio[round(5.5 * 16000):round(10.5 * 16000)] = 0.2
    words = [dict(word="first", start=0, end=5),
             dict(word="other", start=5.5, end=10.5),
             dict(word="return", start=11, end=16)]
    embed_diarize.assign_word_speakers(words, audio, device="cpu")
    assert [word["speaker"] for word in words] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def _asr(monkeypatch, words=True):
    asr = VADWhisperASR.__new__(VADWhisperASR)
    asr.device = "cpu"
    asr.language = "en"
    asr.skip_diarization = False
    asr._embed_diarize = True
    asr.vad_model = object()
    asr._load_audio_16k = lambda source: np.full(32000, 0.1, dtype=np.float32)
    segment = SimpleNamespace(text="hello", start=0, end=1,
                              words=[SimpleNamespace(word="hello", start=0, end=1)] if words else [])
    asr.asr_model = SimpleNamespace(transcribe=lambda *args, **kwargs: ([segment], SimpleNamespace(language="en")))
    monkeypatch.setitem(sys.modules, "silero_vad", SimpleNamespace(
        get_speech_timestamps=lambda *args, **kwargs: [dict(start=0, end=1)]))
    return asr


def test_diarization_failure_is_explicit_and_does_not_collapse_all_people(monkeypatch):
    asr = _asr(monkeypatch)

    def fail(*args, **kwargs):
        raise ImportError("SpeechBrain unavailable")

    monkeypatch.setattr(embed_diarize, "assign_word_speakers", fail)
    with pytest.raises(RuntimeError, match="confirmed single speaker"):
        asr.transcribe("unused.wav")
    output = asr.transcribe("unused.wav", num_speakers=1)
    assert output.segments[0].speaker == "SPEAKER_00"


def test_segment_text_is_kept_if_whisper_omits_word_timestamps(monkeypatch):
    asr = _asr(monkeypatch, words=False)
    output = asr.transcribe("unused.wav", num_speakers=1)
    assert output.transcription == "hello"
    assert output.segments[0].text == "hello"


def test_ecapa_cache_is_released_with_model_pool(monkeypatch):
    from langswap import model_pool

    model_pool.clear_pool()
    calls = []

    def load(device):
        calls.append(device)
        return object()

    monkeypatch.setattr(embed_diarize, "_load_classifier", load)
    try:
        first = embed_diarize._classifier("cpu")
        assert embed_diarize._classifier("cpu") is first
        model_pool.clear_pool()
        assert embed_diarize._classifier("cpu") is not first
        assert calls == ["cpu", "cpu"]
    finally:
        model_pool.clear_pool()
