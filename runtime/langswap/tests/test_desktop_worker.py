"""Native app worker returns real files without using a browser or cloud API."""
from pathlib import Path
from types import SimpleNamespace

import desktop_worker


def test_youtube_mode_writes_video_and_two_srt_files(monkeypatch, tmp_path):
    original = tmp_path / "original.mp4"
    dubbed = tmp_path / "dubbed.mp4"
    english = tmp_path / "english.srt"
    french = tmp_path / "french.srt"
    for path in (original, dubbed, english, french):
        path.write_bytes(b"finished artifact")
    monkeypatch.setenv("LANGSWAP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(desktop_worker, "download_original", lambda *_: (original, "AbCdEfGhI12"))
    def use_track(url, directory, language):
        assert language == "fr"
        return dubbed, english, french
    monkeypatch.setattr(desktop_worker, "download_existing_track", use_track)
    output = tmp_path / "chosen"
    args = SimpleNamespace(url="https://www.youtube.com/watch?v=AbCdEfGhI12&t=10s",
                           file=None, mode="youtube", source_language="english",
                           target_language="fr", output_dir=str(output))
    names = desktop_worker.run(args)
    assert len(names) == 3
    assert all((output / name).is_file() for name in names)
    assert (output / names[0]).read_bytes() == b"finished artifact"


def test_local_mode_refuses_non_french_target_before_loading_models(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setenv("LANGSWAP_DATA_DIR", str(tmp_path / "data"))
    args = SimpleNamespace(url=None, file=str(source), mode="local", source_language="english",
                           target_language="de", output_dir=str(tmp_path / "out"))
    try:
        desktop_worker.run(args)
    except ValueError as exc:
        assert "français" in str(exc)
    else:
        raise AssertionError("Unsupported local voice language was accepted")
