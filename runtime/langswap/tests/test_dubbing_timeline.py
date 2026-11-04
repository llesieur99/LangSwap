"""CPU waveform properties: dubbing follows source time without rewriting TTS."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch

from langswap.ml.video_dubbing_manager import VideoDubbingManager
from langswap.streaming import MIX_SR, StreamingDubber


def _write(path, duration, sr=8000, amplitude=0.1, channels=1):
    values = np.full((round(duration * sr), channels), amplitude, dtype=np.float32)
    sf.write(str(path), values, sr, subtype="FLOAT")
    return str(path)


def _segment(tmp_path, name, start, end, generated_duration=None, sr=8000,
             amplitude=0.1, channels=1):
    return SimpleNamespace(
        start=start, end=end, text=name, translation=name, speaker=name,
        source_file=_write(tmp_path / f"{name}-source.wav", end - start, sr),
        generated_file=_write(tmp_path / f"{name}-generated.wav",
                              generated_duration if generated_duration is not None else end - start,
                              sr, amplitude, channels),
    )


@pytest.fixture
def dub():
    # Avoid model construction; these waveform checks have no speech detector.
    manager = VideoDubbingManager.__new__(VideoDubbingManager)
    manager.logger = Mock()
    manager.model_vad = object()
    manager._file_repository = None
    manager.get_pause = Mock(return_value=(0, []))
    return manager


def _mean(audio, sr, start, end):
    return float(audio[:, round(start * sr):round(end * sr)].mean())


def _silence(audio, sr, start, end):
    assert float(audio[:, round(start * sr):round(end * sr)].abs().max()) < 1e-5


def test_overlong_first_phrase_does_not_move_next_person(tmp_path, dub):
    segments = [_segment(tmp_path, "first", 0.5, 1.5, 3, amplitude=0.1),
                _segment(tmp_path, "reply", 2, 3, amplitude=-0.2)]
    vocals = _write(tmp_path / "vocals.wav", 4, amplitude=0)
    audio, sr = dub.merge_timestamps_speedup(SimpleNamespace(translated_texts=segments), vocals)
    assert audio.shape == (1, 4 * sr)
    assert _mean(audio, sr, 2.1, 2.9) == pytest.approx(-0.2, abs=0.01)
    _silence(audio, sr, 0, 0.4)
    _silence(audio, sr, 3.1, 4)


def test_fitted_phrase_and_remaining_pause_share_exactly_one_source_slot(tmp_path, dub):
    segment = _segment(tmp_path, "long", 0, 1, 3)
    audio, sr, pause = dub.fit_segment(segment, source_sr=8000, available_pause=0.5)
    assert pause >= 0
    assert audio.shape[0] == 1
    assert audio.shape[1] / sr + pause == pytest.approx(1.5, abs=1 / sr)
    # Borrowing may consume at most half the available gap.
    assert audio.shape[1] / sr <= 1.25 + 1 / sr


def test_single_transcript_keeps_source_duration_and_silence(tmp_path, dub):
    segments = [_segment(tmp_path, "only", 1, 2)]
    vocals = _write(tmp_path / "vocals.wav", 5, amplitude=0)
    audio, sr = dub.merge_timestamps_speedup(SimpleNamespace(translated_texts=segments), vocals)
    assert audio.shape == (1, 5 * sr)
    _silence(audio, sr, 0, 0.9)
    _silence(audio, sr, 2.1, 5)
    assert _mean(audio, sr, 1.1, 1.9) == pytest.approx(0.1, abs=0.005)


def test_empty_transcript_fails_explicitly_instead_of_producing_a_muted_dub(tmp_path, dub):
    vocals = _write(tmp_path / "vocals.wav", 5, amplitude=0)
    with pytest.raises(ValueError, match="No translated speech"):
        dub.merge_timestamps_speedup(SimpleNamespace(translated_texts=[]), vocals)


def test_mixed_sample_rates_and_stereo_do_not_change_phrase_times(tmp_path, dub):
    segments = [_segment(tmp_path, "first", 0, 1, sr=8000, amplitude=0.1, channels=2),
                _segment(tmp_path, "reply", 2, 3, sr=16000, amplitude=-0.2)]
    vocals = _write(tmp_path / "vocals.wav", 4, sr=24000, amplitude=0, channels=2)
    audio, sr = dub.merge_timestamps_speedup(SimpleNamespace(translated_texts=segments), vocals)
    assert audio.shape == (1, 4 * sr)
    assert _mean(audio, sr, 0.1, 0.9) == pytest.approx(0.1, abs=0.005)
    assert _mean(audio, sr, 2.1, 2.9) == pytest.approx(-0.2, abs=0.005)
    _silence(audio, sr, 1.1, 1.9)


def test_overlapping_source_turn_does_not_delay_second_person(tmp_path, dub):
    segments = [_segment(tmp_path, "first", 0, 1.5, amplitude=0.1),
                _segment(tmp_path, "reply", 1, 2, amplitude=-0.2)]
    vocals = _write(tmp_path / "vocals.wav", 3, amplitude=0)
    audio, sr = dub.merge_timestamps_speedup(SimpleNamespace(translated_texts=segments), vocals)
    assert audio.shape == (1, 3 * sr)
    # An overlap can mix both speakers or fit the first before the second, but
    # the second person's distinct waveform must be present at its own timestamp.
    assert _mean(audio, sr, 1.1, 1.4) < -0.05
    assert _mean(audio, sr, 1.6, 1.9) == pytest.approx(-0.2, abs=0.005)


def test_pause_adjustment_uses_seconds_and_keeps_generated_cache_unchanged(tmp_path, dub):
    sr = 8000
    source = _write(tmp_path / "source.wav", 1.25, sr)
    generated = tmp_path / "generated.wav"
    sf.write(str(generated), np.concatenate([
        np.full(sr // 2, 0.1), np.zeros(sr), np.full(sr // 2, 0.1),
    ]), sr, subtype="FLOAT")
    before = generated.read_bytes()

    def pauses(path, sample_rate, seconds=True):
        assert seconds is True
        if path == source:
            return 0.25, []
        return 1.0, [dict(start=0, end=0.5, pause=0),
                     dict(start=1.5, end=2, pause=1)]

    dub.get_pause = pauses
    audio = dub.change_pauses(str(generated), source, sr, sr)
    assert audio.shape == (1, round(1.25 * sr))
    _silence(audio, sr, 0.5, 0.75)
    assert _mean(audio, sr, 0.8, 1.2) == pytest.approx(0.1, abs=0.005)
    assert generated.read_bytes() == before
    repeated = dub.change_pauses(str(generated), source, sr, sr)
    assert torch.equal(audio, repeated)


@pytest.mark.parametrize("samples", [[], [(1, 2, 0.1)], [(0.5, 1.5, 0.1), (2, 3, -0.2)]])
def test_real_streaming_chunk_builder_anchors_each_phrase(tmp_path, dub, samples):
    segments = [_segment(tmp_path, f"person{i}", start, end,
                         generated_duration=3 if i == 0 and len(samples) > 1 else None,
                         amplitude=amplitude, sr=8000 if i == 0 else 16000)
                for i, (start, end, amplitude) in enumerate(samples)]
    stream = StreamingDubber.__new__(StreamingDubber)
    stream.vt = SimpleNamespace(translated_texts=segments, background_audio={})
    stream.tts = SimpleNamespace(tts_sample_rate=8000)
    stream.dub = dub
    stream.logger = Mock()
    audio = stream._build_chunk_audio(list(range(len(segments))), 0, 4)
    assert audio.shape == (1, 4 * MIX_SR)
    for i, (start, end, amplitude) in enumerate(samples):
        if i == 0 and len(samples) > 1:
            continue  # stretching the first phrase is allowed; the next must stay fixed
        assert _mean(audio, MIX_SR, start + 0.1, end - 0.1) == pytest.approx(amplitude, abs=0.01)
    _silence(audio, MIX_SR, 3.1, 4)


def test_streaming_chunk_offset_uses_absolute_source_timestamps(tmp_path, dub):
    segment = _segment(tmp_path, "reply", 10.5, 11.5, amplitude=-0.2)
    stream = StreamingDubber.__new__(StreamingDubber)
    stream.vt = SimpleNamespace(translated_texts=[segment], background_audio={})
    stream.tts = SimpleNamespace(tts_sample_rate=8000)
    stream.dub = dub
    stream.logger = Mock()
    audio = stream._build_chunk_audio([0], 10, 2)
    assert audio.shape == (1, 2 * MIX_SR)
    _silence(audio, MIX_SR, 0, 0.4)
    assert _mean(audio, MIX_SR, 0.6, 1.4) == pytest.approx(-0.2, abs=0.005)
    _silence(audio, MIX_SR, 1.6, 2)


def test_streaming_rejects_a_video_cut_inside_a_spoken_phrase(tmp_path, dub):
    segment = _segment(tmp_path, "speech", 0, 1)
    stream = StreamingDubber.__new__(StreamingDubber)
    stream.vt = SimpleNamespace(translated_texts=[segment], background_audio={})
    stream.tts = SimpleNamespace(tts_sample_rate=8000)
    stream.dub = dub
    stream.logger = Mock()
    with pytest.raises(ValueError, match="truncate a spoken phrase"):
        stream._build_chunk_audio([0], 0, 0.5)
