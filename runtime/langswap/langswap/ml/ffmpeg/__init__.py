import subprocess
from enum import Enum, auto
import os
from tempfile import NamedTemporaryFile
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Util(Enum):
    ffmpeg = auto()
    ffprobe = auto()

class FFmpegClient:
    def __init__(self, ffmpeg_path='ffmpeg', ffprobe_path='ffprobe'):
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

    def run_command(self, args: List[str], util: Util = Util.ffmpeg):
        """Run ffmpeg/ffprobe with `args` as an argument LIST (no shell).

        Passing a list means paths are individual argv elements, so filenames
        with spaces or special characters work without quoting/escaping.
        """
        util_path = self.ffprobe_path if util == Util.ffprobe else self.ffmpeg_path
        cmd = [util_path, "-hide_banner", "-loglevel", "error", *args]
        try:
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as exc:
            error = exc.stderr.decode('utf-8', 'replace') if isinstance(exc.stderr, bytes) else exc.stderr
            raise RuntimeError(f'{util_path} failed (exit {exc.returncode}): {error}') from exc
        if process.stderr:
            logger.debug(f"Command error: {process.stderr}")
        return process.stdout, process.stderr

    def extract_audio(self, input_path, output_path, time_limit: Optional[int] = None, target_sr=24000):
        """Extract audio from video. Raises if ffmpeg produced no output file."""
        args = ["-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le",
                "-ar", str(target_sr), "-ac", "1", "-f", "wav", output_path]
        if time_limit is not None:
            args[-1:-1] = ['-t', str(time_limit)]
        stdout, stderr = self.run_command(args)
        if not os.path.exists(output_path):
            err = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr
            raise RuntimeError(
                f"ffmpeg failed to extract audio from {input_path!r}: {err.strip()}"
            )
        return stdout, stderr

    def resample_audio(self, input_path, output_path, sample_rate: int = 16_000):
        args = ["-y", "-i", input_path, "-ar", str(sample_rate), "-f", "wav", output_path]
        return self.run_command(args)

    # ── streaming helpers (langswap/streaming.py) ─────────────────────────────
    def probe_keyframes(self, input_path: str) -> List[float]:
        """Presentation timestamps (seconds) of every video keyframe.

        The streaming path snaps chunk boundaries onto keyframes so each segment
        can be cut with ``-c:v copy`` and stay independently decodable
        (docs/streaming_dubbing_design.md §3.2). ``-skip_frame nokey`` makes the
        decoder emit only keyframes, so this is cheap.

        We request both pts_time and best_effort_timestamp_time and take whichever
        is present per line: with ``-skip_frame nokey`` older ffmpeg (4.2.x) leaves
        pts_time blank while populating best_effort_timestamp_time.
        """
        args = ["-select_streams", "v:0", "-skip_frame", "nokey",
                "-show_entries", "frame=pts_time,best_effort_timestamp_time",
                "-of", "csv=p=0", input_path]
        stdout, _ = self.run_command(args, util=Util.ffprobe)
        times = []
        for line in stdout.decode("utf-8", "ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            for f in line.split(","):
                f = f.strip()
                if not f or f == "N/A":
                    continue
                try:
                    times.append(float(f))
                    break
                except ValueError:
                    continue
        return sorted(set(times))

    def probe_duration(self, input_path: str) -> float:
        """Container duration in seconds (ffprobe format=duration)."""
        args = ["-show_entries", "format=duration", "-of", "csv=p=0", input_path]
        stdout, _ = self.run_command(args, util=Util.ffprobe)
        out = stdout.decode("utf-8", "ignore").strip()
        return float(out) if out and out != "N/A" else 0.0

    def probe_fps(self, input_path: str) -> float:
        """Average frame rate as a float (parses ffprobe ``r_frame_rate``)."""
        args = ["-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                "-of", "csv=p=0", input_path]
        stdout, _ = self.run_command(args, util=Util.ffprobe)
        out = stdout.decode("utf-8", "ignore").strip()
        if "/" in out:
            num, den = out.split("/")
            return float(num) / float(den) if float(den) else 0.0
        return float(out) if out else 0.0

    def split_video_at_frames(self, video_input_path: str, frame_indices: List[int],
                              output_pattern: str):
        """Split video-only into clean keyframe-aligned pieces with ``-c:v copy``.

        ``frame_indices`` are the frame numbers at which a new piece begins (each
        must be a keyframe). Unlike per-chunk ``-ss``/``-t`` cutting — which pulls
        extra trailing frames and overlaps the next chunk — the segment muxer cuts
        losslessly with no overlap or gap. Audio is dropped here; it's muxed back
        per chunk by ``mux_chunk_fmp4`` once the dubbed track is ready, so this
        whole-video split costs nothing toward TTFS (design §3.2).
        """
        frames = [str(int(f)) for f in frame_indices if f > 0]
        if not frames:
            # No interior splits: the whole video is one piece. The segment muxer
            # rejects an empty -segment_frames, so copy directly to %03d=0.
            out0 = output_pattern % 0 if "%" in output_pattern else output_pattern
            args = ["-y", "-i", video_input_path, "-an", "-c:v", "copy", out0]
            return self.run_command(args)
        args = ["-y", "-i", video_input_path, "-an", "-c:v", "copy",
                "-f", "segment", "-segment_frames", ",".join(frames),
                "-reset_timestamps", "1", "-segment_format", "mp4", output_pattern]
        return self.run_command(args)

    def mux_chunk_fmp4(self, video_piece_path: str, audio_input_path: str,
                       output_path: str, audio_sr: int = 48000):
        """Mux one pre-cut video piece + its dubbed audio into a CMAF fragment.

        No ``-ss``/``-t`` (the video piece is already an exact keyframe-aligned
        slice from ``split_video_at_frames``), so the cut is clean. Video is
        copied; audio is AAC at a fixed rate so the ``moov`` is identical across
        chunks and one shared init segment is valid for them all.
        ``+empty_moov+frag_keyframe+default_base_moof`` yields a fragment that
        ``langswap.hls.split_fragmented_mp4`` splits into init + ``.m4s``.
        """
        args = ["-y", "-i", video_piece_path, "-i", audio_input_path,
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-ar", str(audio_sr), "-ac", "2",
                "-movflags", "+empty_moov+frag_keyframe+default_base_moof",
                "-f", "mp4", output_path]
        return self.run_command(args)

    def replace_audio(self, video_input_path: str, audio_input_path: str,
                      video_output_path: str,
                      time_limit: Optional[int] = None):
        with NamedTemporaryFile(suffix='.mp4', dir=os.path.dirname(video_output_path) or '.', delete=False) as tmp:
            temporary_output = tmp.name
        args = ["-y", "-i", video_input_path, "-i", audio_input_path,
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0"]
        if time_limit:
            args += ["-t", str(time_limit)]
        args += ["-f", "mp4", "-shortest", temporary_output]
        try:
            result = self.run_command(args)
            if os.path.getsize(temporary_output) == 0:
                raise RuntimeError('FFmpeg produced an empty muxed video')
            os.replace(temporary_output, video_output_path)
            return result
        finally:
            if os.path.exists(temporary_output):
                os.unlink(temporary_output)

    def add_watermark(self, input_path: str, output_path: str,
                      text: str = "translated with LangSwap",
                      fontcolor: str = "white",
                      fontsize: int = 16,
                      x: str = "w-tw-10",
                      y: str = "h-th-10",
                      fontfile: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        """Adds a watermark using FFmpeg's drawtext filter. Handles in-place operations via a temporary file."""
        in_place = os.path.abspath(input_path) == os.path.abspath(output_path)
        if in_place:
            tmp_dir, ext = os.path.dirname(input_path), os.path.splitext(input_path)[1]
            with NamedTemporaryFile(suffix=ext, dir=tmp_dir, delete=False) as tmp:
                temp_output_path = tmp.name
        else:
            temp_output_path = output_path

        vf = (f"drawtext=text='{text}':fontfile={fontfile}:fontcolor={fontcolor}:"
              f"fontsize={fontsize}:x={x}:y={y}")
        args = ["-y", "-i", input_path, "-vf", vf, "-codec:a", "copy", temp_output_path]
        try:
            stdout, stderr = self.run_command(args)
            if not os.path.isfile(temp_output_path) or os.path.getsize(temp_output_path) == 0:
                raise ValueError(f"FFmpeg failed to produce an output file. Error: {stderr}")
            if in_place:
                os.replace(temp_output_path, input_path)
            return stdout, stderr
        finally:
            if in_place and os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
