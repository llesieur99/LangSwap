"""Download one public YouTube video's original audio for local dubbing."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse


_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def video_id_from_url(url: str) -> str:
    """Accept a single YouTube video and discard playlist/time parameters."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme != "https" or parsed.hostname not in _HOSTS or parsed.port:
        raise ValueError("Enter an HTTPS YouTube video URL.")
    if parsed.hostname == "youtu.be":
        video_id = parsed.path.strip("/")
    elif parsed.path == "/watch":
        video_id = (parse_qs(parsed.query).get("v") or [""])[0]
    elif parsed.path.startswith(("/shorts/", "/live/")):
        video_id = parsed.path.strip("/").split("/")[1]
    else:
        raise ValueError("Enter a YouTube video URL, not a channel or playlist.")
    if not _VIDEO_ID.fullmatch(video_id):
        raise ValueError("The YouTube video id is invalid.")
    return video_id


def download_original(url: str, job_dir: str | Path) -> tuple[Path, str]:
    """Fetch the highest available H.264 video and original English audio."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("Install the LangSwap YouTube dependency: yt-dlp[default].") from exc
    video_id = video_id_from_url(url)
    target = Path(job_dir) / video_id
    target.mkdir(parents=True, exist_ok=True)
    source = target / "source.mp4"
    if _usable_source(source):
        return source, video_id
    pending = target / "source-download.mp4"
    if _usable_source(pending):
        os.replace(pending, source)
        return source, video_id
    if pending.is_file():
        pending.unlink()
    options = {
        "noplaylist": True,
        "format": "bv*[vcodec^=avc1][height<=1080]+ba[language^=en]/bv*[height<=1080]+ba[language^=en]",
        "outtmpl": str(target / "source-download.%(ext)s"),
        "merge_output_format": "mp4",
        "js_runtimes": {"deno": {}},
        "quiet": True,
        "cachedir": False,
        "no_warnings": False,
    }
    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
    if not info or info.get("id") != video_id or not _usable_source(pending):
        raise RuntimeError("The YouTube video or its original audio was not downloaded completely.")
    os.replace(pending, source)
    return source, video_id


def _usable_source(path: Path) -> bool:
    """Do not reuse a large partial file or a silent video-only cache entry."""
    if not path.is_file() or path.stat().st_size < 100_000:
        return False
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return {"audio", "video"}.issubset(set(probe.stdout.splitlines()))


def _format_srt_time(milliseconds: int) -> str:
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def caption_json3_to_srt(source: Path, destination: Path) -> int:
    """Group YouTube's timed caption pieces into short, readable cues."""
    import json

    raw = json.loads(source.read_text(encoding="utf-8"))
    pieces = []
    for event in raw.get("events", []):
        text = "".join(piece.get("utf8", "") for piece in event.get("segs", [])).strip()
        if text:
            start = int(event["tStartMs"])
            pieces.append((start, start + max(int(event.get("dDurationMs", 0)), 200), text))
    groups, current = [], []
    for piece in pieces:
        if current and (piece[0] - current[0][0] > 5500 or sum(len(p[2]) for p in current) > 85):
            groups.append(current)
            current = []
        current.append(piece)
        if piece[2].endswith((".", "?", "!")):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    lines = []
    for i, group in enumerate(groups):
        start, end = group[0][0], group[-1][1]
        if i + 1 < len(groups):
            end = min(end, groups[i + 1][0][0])
        end = max(start + 400, end)
        text = " ".join(part[2] for part in group).replace("\n", " ")
        lines.append(f"{i + 1}\n{_format_srt_time(start)} --> {_format_srt_time(end)}\n{text}\n")
    if not lines:
        raise ValueError("The selected YouTube subtitle track is empty.")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def qwen_sentences_from_json3(source: Path, destination: Path) -> list[dict]:
    """Use complete sentences as voice-generation units and keep their times."""
    import json

    events = []
    for event in json.loads(source.read_text(encoding="utf-8")).get("events", []):
        text = "".join(item.get("utf8", "") for item in event.get("segs", [])).strip()
        if not text:
            continue
        start = event["tStartMs"] / 1000
        end = (event["tStartMs"] + max(event.get("dDurationMs", 0), 150)) / 1000
        boundaries = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=\S)", text)] + [len(text)]
        for left, right in zip(boundaries, boundaries[1:]):
            fragment = text[left:right].strip()
            if fragment:
                events.append((start + (end - start) * left / len(text),
                               start + (end - start) * right / len(text), fragment))
    groups, current = [], []
    for piece in events:
        if current and (piece[0] - current[0][0] > 26 or sum(len(x[2]) for x in current) > 500):
            groups.append(current)
            current = []
        current.append(piece)
        if piece[2].endswith((".", "?", "!")) and piece[1] - current[0][0] >= 2:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    sentences = []
    for index, group in enumerate(groups):
        start, end = group[0][0], group[-1][1]
        if index + 1 < len(groups):
            end = min(end, groups[index + 1][0][0])
        end = max(start + 0.3, end)
        text = " ".join(item[2] for item in group).replace("\n", " ").strip()
        text = re.sub(r"(?<=\d)%(?=\w)", "% ", text)
        if text:
            sentences.append({"start": start, "end": end, "text": text})
    if not sentences:
        raise ValueError("No French sentences were found in the YouTube captions.")
    destination.write_text("\n".join(
        f"{i+1}\n{_format_srt_time(round(row['start']*1000))} --> "
        f"{_format_srt_time(round(row['end']*1000))}\n{row['text']}\n"
        for i, row in enumerate(sentences)
    ) + "\n", encoding="utf-8")
    return sentences


def download_caption(url: str, job_dir: str | Path, code: str) -> Path:
    """Download one automatic JSON3 caption track with yt-dlp's own transport."""
    from yt_dlp import YoutubeDL

    video_id = video_id_from_url(url)
    folder = Path(job_dir) / video_id
    folder.mkdir(parents=True, exist_ok=True)
    caption = folder / f"captions.{code}.json3"
    if not caption.is_file():
        with YoutubeDL({
            "noplaylist": True, "skip_download": True,
            "writeautomaticsub": True, "subtitleslangs": [code],
            "subtitlesformat": "json3",
            "outtmpl": str(folder / "captions.%(ext)s"),
            "js_runtimes": {"deno": {}}, "quiet": True, "cachedir": False,
        }) as downloader:
            downloader.download([f"https://www.youtube.com/watch?v={video_id}"])
    if not caption.is_file():
        raise RuntimeError(f"YouTube has no {code} automatic transcript for this video.")
    return caption


def download_existing_track(url: str, job_dir: str | Path, language: str) -> tuple[Path, Path, Path]:
    """Prepare a YouTube audio track and its subtitles for one target language."""
    from yt_dlp import YoutubeDL

    if language not in {"fr", "es", "de", "it", "pt"}:
        raise ValueError("Choose a supported YouTube dubbing language.")
    source, video_id = download_original(url, job_dir)
    folder = source.parent
    canonical = f"https://www.youtube.com/watch?v={video_id}"
    tracks = [path for path in folder.glob(f"youtube-{language}.*")
              if path.suffix in {".m4a", ".webm", ".mp4"} and path.stat().st_size > 10_000]
    if not tracks:
        with YoutubeDL({
            "noplaylist": True,
            "format": f"ba[language^={language}][ext=m4a]/ba[language^={language}]",
            "outtmpl": str(folder / f"youtube-{language}.%(ext)s"),
            "js_runtimes": {"deno": {}},
            "quiet": True,
            "cachedir": False,
        }) as downloader:
            info = downloader.extract_info(canonical, download=True)
        tracks = [path for path in folder.glob(f"youtube-{language}.*")
                  if path.suffix in {".m4a", ".webm", ".mp4"} and path.stat().st_size > 10_000]
        if not info or info.get("id") != video_id or not tracks:
            raise RuntimeError(f"This video has no usable {language} YouTube audio track.")
    audio = tracks[0]

    caption_paths = []
    for code in ("en-orig", f"{language}-orig"):
        caption = download_caption(url, job_dir, code)
        srt = folder / f"captions.{code}.srt"
        caption_json3_to_srt(caption, srt)
        caption_paths.append(srt)

    result = folder / f"youtube-{language}.mp4"
    if not result.is_file() or result.stat().st_size < 100_000:
        from langswap.ml.ffmpeg import FFmpegClient
        FFmpegClient().replace_audio(str(source), str(audio), str(result))
    return result, caption_paths[0], caption_paths[1]


def download_existing_french(url: str, job_dir: str | Path) -> tuple[Path, Path, Path]:
    """Backward-compatible French option for the Gradio comparison interface."""
    return download_existing_track(url, job_dir, "fr")
