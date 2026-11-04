"""YouTube URL boundary and local output behavior, without a network request."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from langswap import youtube_source
from langswap.youtube_source import (
    caption_json3_to_srt, download_original, qwen_sentences_from_json3, video_id_from_url,
)


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=AbCdEfGhI12&t=10s",
    "https://youtu.be/AbCdEfGhI12?t=10",
    "https://m.youtube.com/watch?list=old&v=AbCdEfGhI12",
])
def test_video_url_discards_time_and_playlist(url):
    assert video_id_from_url(url) == "AbCdEfGhI12"


@pytest.mark.parametrize("url", [
    "http://youtube.com/watch?v=AbCdEfGhI12",
    "https://youtube.com.evil.test/watch?v=AbCdEfGhI12",
    "https://youtube.com/playlist?list=abc",
    "https://www.youtube.com/watch?v=../etc/passwd",
    "file:///etc/passwd",
])
def test_rejects_unsupported_or_unsafe_urls(url):
    with pytest.raises(ValueError):
        video_id_from_url(url)


def test_download_uses_one_original_english_track_and_reuses_complete_file(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(youtube_source, "_usable_source",
                        lambda path: path.is_file() and path.read_bytes().startswith(b"good"))

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options
            calls.append(self)
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def extract_info(self, url, download):
            assert download is True
            assert url == "https://www.youtube.com/watch?v=AbCdEfGhI12"
            assert "language^=en" in self.options["format"]
            assert "vcodec^=avc1" in self.options["format"]
            assert self.options["noplaylist"] is True
            Path(self.options["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"good" + b"x" * 100_001)
            return {"id": "AbCdEfGhI12"}

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    first, video_id = download_original("https://youtu.be/AbCdEfGhI12?t=10", tmp_path)
    second, _ = download_original("https://youtu.be/AbCdEfGhI12", tmp_path)
    assert video_id == "AbCdEfGhI12" and first == second
    assert len(calls) == 1


def test_cache_requires_audio_and_video_streams(monkeypatch, tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"x" * 100_001)
    fake = lambda output: SimpleNamespace(stdout=output)
    monkeypatch.setattr(youtube_source.subprocess, "run", lambda *a, **kw: fake("video\n"))
    assert not youtube_source._usable_source(video)
    monkeypatch.setattr(youtube_source.subprocess, "run",
                        lambda *a, **kw: fake("video\naudio\n"))
    assert youtube_source._usable_source(video)


def test_invalid_cache_is_replaced_after_successful_download(monkeypatch, tmp_path):
    cached = tmp_path / "AbCdEfGhI12" / "source.mp4"
    cached.parent.mkdir()
    cached.write_bytes(b"video only")
    monkeypatch.setattr(youtube_source, "_usable_source",
                        lambda path: path.is_file() and path.read_bytes().startswith(b"good"))
    class FakeYoutubeDL:
        def __init__(self, options): self.options = options
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def extract_info(self, url, download):
            Path(self.options["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"good merged")
            return {"id": "AbCdEfGhI12"}
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    result, _ = download_original("https://youtu.be/AbCdEfGhI12", tmp_path)
    assert result == cached and result.read_bytes() == b"good merged"


def test_source_requires_a_completed_video(monkeypatch, tmp_path):
    class FakeYoutubeDL:
        def __init__(self, options): self.options = options
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def extract_info(self, url, download): return {"id": "AbCdEfGhI12"}
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    with pytest.raises(RuntimeError, match="not downloaded completely"):
        download_original("https://youtu.be/AbCdEfGhI12", tmp_path)


def test_export_preserves_previous_run(tmp_path):
    import gradio_demo
    video, en, fr = (tmp_path / name for name in ("video.mp4", "en.srt", "fr.srt"))
    for path in (video, en, fr): path.write_bytes(b"first")
    output = tmp_path / "selected"
    first = gradio_demo._export_results(str(video), str(en), str(fr), str(output), "video", "french")
    video.write_bytes(b"second")
    second = gradio_demo._export_results(str(video), str(en), str(fr), str(output), "video", "french")
    assert all(Path(path).is_file() for path in first + second)
    assert Path(first[0]).read_bytes() == b"first"
    assert Path(second[0]).read_bytes() == b"second"
    assert Path(first[0]).parent == output


def test_ui_accepts_youtube_url_and_returns_chosen_folder(monkeypatch, tmp_path):
    import gradio_demo
    downloaded = tmp_path / "source.mp4"
    downloaded.write_bytes(b"video")
    produced = tmp_path / "result.mp4"
    produced.write_bytes(b"translated video")
    source_srt = tmp_path / "en.srt"
    translated_srt = tmp_path / "fr.srt"
    source_srt.write_text("source", encoding="utf-8")
    translated_srt.write_text("français", encoding="utf-8")
    def fake_download(url, folder):
        assert "t=10s" in url
        return downloaded, "AbCdEfGhI12"
    monkeypatch.setattr(gradio_demo, "download_original", fake_download)
    monkeypatch.setattr(gradio_demo, "BASE_DIR", str(tmp_path / "jobs"))
    pipeline = Mock()
    pipeline.config.dubbing_algo = "speedup"
    pipeline._merge.return_value.processed_video.file_path = str(produced)
    pipeline.generate_srt_files.return_value = [SimpleNamespace(file_path=str(source_srt)),
                                                SimpleNamespace(file_path=str(translated_srt))]
    monkeypatch.setattr(gradio_demo, "VideoTranslationPipeline", Mock(return_value=pipeline))
    output = tmp_path / "chosen"
    files = gradio_demo.translate_video(
        None, "english", "french", "openvoice", "vad", "marian", "speedup", "cpu", False,
        video_url="https://www.youtube.com/watch?v=AbCdEfGhI12&t=10s", output_dir=str(output),
        progress=lambda *args, **kwargs: None,
    )
    assert all(Path(path).parent == output for path in files[:3])
    assert Path(files[0]).read_bytes() == b"translated video"
    assert "Success" in files[3]


def test_youtube_auto_caption_words_are_grouped_and_empty_cues_ignored(tmp_path):
    import json
    source=tmp_path/'captions.fr-orig.json3'
    source.write_text(json.dumps({'events':[
        {'tStartMs':1000,'dDurationMs':1000,'segs':[{'utf8':'Bonjour'}]},
        {'tStartMs':2000,'dDurationMs':1000,'segs':[{'utf8':' le monde.'}]},
        {'tStartMs':2500,'dDurationMs':50,'segs':[{'utf8':' '}]},
        {'tStartMs':9000,'dDurationMs':800,'segs':[{'utf8':'Ensuite.'}]},
    ]}),encoding='utf-8')
    output=tmp_path/'captions.srt'
    assert caption_json3_to_srt(source,output)==2
    content=output.read_text(encoding='utf-8')
    assert '00:00:01,000 --> 00:00:03,000' in content
    assert 'Bonjour le monde.' in content
    assert 'Ensuite.' in content


def test_ui_can_export_the_existing_youtube_french_track(monkeypatch,tmp_path):
    import gradio_demo
    source=tmp_path/'source.mp4'
    source.write_bytes(b'source')
    french=tmp_path/'youtube-fr.mp4'
    french.write_bytes(b'French YouTube track')
    en=tmp_path/'english.srt'
    fr=tmp_path/'french.srt'
    en.write_text('Original',encoding='utf-8')
    fr.write_text('Français',encoding='utf-8')
    monkeypatch.setattr(gradio_demo,'download_original',lambda *_:(source,'AbCdEfGhI12'))
    monkeypatch.setattr(gradio_demo,'download_existing_french',lambda *_:(french,en,fr))
    monkeypatch.setattr(gradio_demo,'BASE_DIR',str(tmp_path/'jobs'))
    destination=tmp_path/'selected'
    result=gradio_demo.translate_video(None,'english','french','openvoice','vad','marian',
        'speedup','cpu',False,video_url='https://youtu.be/AbCdEfGhI12',
        output_dir=str(destination),audio_mode='youtube-fr',progress=lambda *a,**kw:None)
    assert all(Path(path).is_file() and Path(path).parent==destination for path in result[:3])
    assert Path(result[0]).read_bytes()==b'French YouTube track'
    assert 'YouTube French audio' in result[3]


def test_qwen_groups_caption_fragments_into_complete_sentences(tmp_path):
    import json
    source = tmp_path / "captions.fr-orig.json3"
    source.write_text(json.dumps({"events": [
        {"tStartMs": 1000, "dDurationMs": 1500, "segs": [{"utf8": "Bonjour"}]},
        {"tStartMs": 2500, "dDurationMs": 1600, "segs": [{"utf8": " à tous."}]},
        {"tStartMs": 4200, "dDurationMs": 2200, "segs": [{"utf8": "Voici notre vidéo."}]},
    ]}), encoding="utf-8")
    srt = tmp_path / "sentences.srt"
    sentences = qwen_sentences_from_json3(source, srt)
    assert [item["text"] for item in sentences] == ["Bonjour à tous.", "Voici notre vidéo."]
    assert sentences[0]["start"] == 1.0
    assert sentences[0]["end"] <= sentences[1]["start"]
    assert "Bonjour à tous." in srt.read_text(encoding="utf-8")
