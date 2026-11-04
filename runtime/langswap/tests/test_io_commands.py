"""Regression tests for subprocess failures, cache integrity and update dispatch."""
import importlib.util
import subprocess
import sys
import types
import wave
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torchaudio

from langswap.ml.ffmpeg import FFmpegClient
from langswap.ml.text_to_speech_service.demucs_client import DemucsClient


def test_ffmpeg_nonzero_exit_is_an_error_even_when_output_exists(tmp_path, monkeypatch):
    destination = tmp_path / 'old.wav'
    destination.write_bytes(b'old output')

    def fail(cmd, **kwargs):
        assert kwargs['check'] is True
        raise subprocess.CalledProcessError(1, cmd, stderr=b'invalid input')

    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='invalid input'):
        FFmpegClient().extract_audio('missing.mp4', str(destination))


@pytest.mark.parametrize('empty_success', [False, True])
def test_in_place_watermark_failure_preserves_video(tmp_path, monkeypatch, empty_success):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'original video')
    client = FFmpegClient()

    def failed(args):
        if empty_success:
            return b'', b''
        raise RuntimeError('encoding failed')

    monkeypatch.setattr(client, 'run_command', failed)
    with pytest.raises((RuntimeError, ValueError)):
        client.add_watermark(str(source), str(source))
    assert source.read_bytes() == b'original video'
    assert list(tmp_path.iterdir()) == [source]


def test_extract_audio_passes_time_limit(tmp_path, monkeypatch):
    client = FFmpegClient()
    destination = tmp_path / 'out.wav'

    def run(args):
        assert args[args.index('-t') + 1] == '3'
        destination.write_bytes(b'audio')
        return b'', b''

    monkeypatch.setattr(client, 'run_command', run)
    client.extract_audio('input.mp4', str(destination), time_limit=3)


@pytest.mark.parametrize('empty_success', [False, True])
def test_mux_failure_preserves_previous_result(tmp_path, monkeypatch, empty_success):
    destination = tmp_path / 'result.mp4'
    destination.write_bytes(b'previous result')
    client = FFmpegClient()

    def failed(args):
        if empty_success:
            return b'', b''
        Path(args[-1]).write_bytes(b'partial result')
        raise RuntimeError('encoding failed')

    monkeypatch.setattr(client, 'run_command', failed)
    with pytest.raises(RuntimeError):
        client.replace_audio('source.mp4', 'audio.wav', str(destination))
    assert destination.read_bytes() == b'previous result'
    assert list(tmp_path.iterdir()) == [destination]


def _wav(path):
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b'\0\0' * 16)


@pytest.fixture
def separator(tmp_path, monkeypatch):
    import demucs.api
    client = DemucsClient(load_separator=False)
    model = Mock(samplerate=16000)
    model.separate_audio_file.return_value = (None, {mode: torch.zeros(1, 16) for mode in ['bass', 'drums', 'other', 'vocals']})
    loader = Mock(side_effect=lambda: setattr(client, '_separator', model))
    monkeypatch.setattr(client, '_ensure_separator', loader)
    monkeypatch.setattr(demucs.api, 'save_audio', lambda source, path, samplerate: _wav(path))
    source = tmp_path / 'input.wav'
    _wav(source)
    return client, loader, model, source, tmp_path / 'stems'


def test_demucs_reuses_only_complete_source_matched_cache(separator):
    client, loader, model, source, stems = separator
    assert len(client.separate(str(source), str(stems))) == 4
    assert len(client.separate(str(source), str(stems))) == 4
    assert loader.call_count == 1
    assert model.separate_audio_file.call_count == 1


@pytest.mark.parametrize('damage', ['missing', 'empty', 'truncated', 'changed_source', 'missing_manifest', 'corrupt_manifest'])
def test_demucs_damaged_or_stale_cache_is_regenerated(separator, damage):
    client, loader, model, source, stems = separator
    client.separate(str(source), str(stems))
    if damage == 'missing':
        (stems / 'vocals.wav').unlink()
    elif damage == 'empty':
        (stems / 'vocals.wav').write_bytes(b'')
    elif damage == 'truncated':
        path = stems / 'vocals.wav'
        path.write_bytes(path.read_bytes()[:-4])
    elif damage == 'changed_source':
        source.write_bytes(source.read_bytes() + b'changed')
    elif damage == 'missing_manifest':
        (stems / 'source.sha256').unlink()
    else:
        (stems / 'source.sha256').write_bytes(b'\xffinvalid digest')
    client.separate(str(source), str(stems))
    assert model.separate_audio_file.call_count == 2


def test_failed_demucs_does_not_publish_cache_manifest(separator):
    client, loader, model, source, stems = separator
    model.separate_audio_file.return_value = (None, {'vocals': torch.zeros(1, 16)})
    with pytest.raises(RuntimeError, match='all four'):
        client.separate(str(source), str(stems))
    assert not (stems / 'source.sha256').exists()


def test_background_mix_resamples_in_memory_preserving_four_stems(tmp_path, monkeypatch):
    import demucs.api
    separator_loader = Mock(side_effect=AssertionError('merge must not load the Demucs separator'))
    monkeypatch.setattr(demucs.api, 'Separator', separator_loader)
    client = DemucsClient(load_separator=False)
    speech = tmp_path / 'speech.wav'
    samples = 1600
    torchaudio.save(str(speech), torch.tensor([0.1, 0.2, 0.3, 0.4])[:, None].repeat(1, samples), 16000)
    backgrounds = {}
    specifications = [('bass', 8000, 600, [0.02]), ('drums', 48000, 6000, [0.03, 0.03]),
                      ('other', 16000, 1600, [0.01, 0.05, 0.09]), ('vocals', 22050, 2205, [0.1, 0.2])]
    for mode, sample_rate, count, channels in specifications:
        path = tmp_path / f'{mode}.wav'
        torchaudio.save(str(path), torch.tensor(channels)[:, None].repeat(1, count), sample_rate)
        backgrounds[f'{mode}.wav'] = str(path)
    original = {name: Path(path).read_bytes() for name, path in backgrounds.items()}
    mixed, sample_rate = client.merge_background(str(speech), backgrounds)
    assert sample_rate == 16000
    assert mixed.shape == (1, samples)
    assert torch.isfinite(mixed).all()
    assert float(mixed.abs().max()) <= 0.95
    assert float(mixed[:, 400:800].mean()) == pytest.approx(0.35, abs=0.002)
    assert {name: Path(path).read_bytes() for name, path in backgrounds.items()} == original
    assert client._separator is None
    separator_loader.assert_not_called()


def test_serverless_update_returns_update_result_without_full_translation(monkeypatch):
    api = types.ModuleType('langswap.api')
    api.process_translation = Mock()
    api.process_update_translation = Mock(return_value={'s3_result_video_url': 'updated'})
    monkeypatch.setitem(sys.modules, 'langswap.api', api)
    monkeypatch.setitem(sys.modules, 'runpod', types.SimpleNamespace(serverless=Mock()))
    monkeypatch.setattr(subprocess, 'run', Mock())
    spec = importlib.util.spec_from_file_location('isolated_serverless', Path(__file__).parents[1] / 'serverless.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.handler({'input': {'update_request': True}}) == {'s3_result_video_url': 'updated'}
    api.process_update_translation.assert_called_once()
    api.process_translation.assert_not_called()


def test_cli_defaults_and_directory_input_rejection(tmp_path):
    spec = importlib.util.spec_from_file_location('isolated_main', Path(__file__).parents[1] / 'main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module._build_parser().parse_args(['local', 'input.mp4', 'french', 'english'])
    assert args.tts == 'openvoice'
    assert args.translation == 'marian'
    assert module.run_local(str(tmp_path), 'french', 'english', 'cpu', args.tts, args.asr,
                            args.translation, args.dubbing, True, None) == 2
