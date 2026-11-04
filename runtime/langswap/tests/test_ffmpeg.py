"""Regression tests for FFmpegClient command construction.

The core test needs no ffmpeg binary or media: it monkeypatches subprocess.run
to assert paths with spaces are passed as single argv elements (the bug was
building a shell string + shlex.split, which split spaced paths into tokens).
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from langswap.ml.ffmpeg import FFmpegClient


def _capture_cmd(monkeypatch, write_output=False):
    captured = {}

    class _Result:
        stdout = b""
        stderr = b""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        if write_output:
            Path(cmd[-1]).write_bytes(b'muxed video')
        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def test_spaced_paths_stay_single_argv_elements(monkeypatch):
    captured = _capture_cmd(monkeypatch)
    FFmpegClient().resample_audio("/tmp/a b c.wav", "/tmp/out d e.wav")
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "ffmpeg"
    # The spaced paths must each be ONE element, not split on spaces.
    assert "/tmp/a b c.wav" in cmd
    assert "/tmp/out d e.wav" in cmd


def test_replace_audio_includes_both_inputs_and_time_limit(monkeypatch, tmp_path):
    captured = _capture_cmd(monkeypatch, write_output=True)
    destination = tmp_path / 'v out.mp4'
    FFmpegClient().replace_audio("/v in.mp4", "/a in.wav", str(destination), time_limit=60)
    cmd = captured["cmd"]
    assert "/v in.mp4" in cmd and "/a in.wav" in cmd
    assert Path(cmd[-1]).parent == destination.parent
    assert destination.read_bytes() == b'muxed video'
    assert not Path(cmd[-1]).exists()
    assert "-t" in cmd and "60" in cmd


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_extract_audio_raises_on_missing_input(tmp_path):
    with pytest.raises(RuntimeError):
        FFmpegClient().extract_audio(
            str(tmp_path / "does not exist.mp4"), str(tmp_path / "out.wav")
        )


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_extract_audio_end_to_end_with_spaces_in_filename(tmp_path):
    """The bug that broke real uploads: a video whose path contains spaces
    (e.g. 'messages span 23 year.mp4') yielded no extracted audio. Build a tiny
    real video at a spaced path and confirm extraction produces a non-empty wav."""
    src = tmp_path / "messages span 23 year.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1", "-shortest",
         "-pix_fmt", "yuv420p", str(src)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    out = tmp_path / "extracted audio out.wav"
    FFmpegClient().extract_audio(str(src), str(out))
    assert out.exists() and out.stat().st_size > 0
