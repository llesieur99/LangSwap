"""Streaming / chunked dubbing orchestrator (Phase 0 — serial).

Instead of returning one final MP4, this emits the dubbed video as HLS fMP4
segments as they are produced, so a viewer can start watching within seconds.
See docs/streaming_dubbing_design.md.

Phase 0 is deliberately **serial** (no async overlap yet): it proves the
muxing / manifest / playback chain end-to-end with correct A/V sync, and is the
spine the Phase 1 async pipeline wraps. The batch path
(`api.process_translation`, `VideoTranslationPipeline.translate_video`) is the
separate, untouched orchestrator in `langswap/translation_pipeline.py`.

Pipeline:
    demucs + ASR (whole audio)  →  translate (whole)  →  TTS setup
    then per chunk, in playlist order:
        synthesize segments → fit each (no global stretch) → build chunk vocals
        → mix continuous background slice → mux pre-cut video piece + audio
        → fMP4 fragment → split init/media → append EVENT manifest → yield event

Public entrypoint: ``stream_dub(input, repo=None) -> generator of events``.
"""

from __future__ import annotations

import os
import time
import uuid

import torch
import torchaudio
import torchaudio.transforms as T

from langswap.pipeline_models.models import TranslationPipelineConfig
from langswap.translation_pipeline import VideoTranslationPipeline
from langswap.ml.text_to_speech_service import TextToSpeechManager
from langswap.ml.text_to_speech_service.audio_dubbing_manager import AudioDubbingManager
from langswap.ml.video_dubbing_manager import VideoDubbingManager
from langswap.ml.ffmpeg import FFmpegClient
from langswap.hls import HlsManifest, split_fragmented_mp4


MIX_SR = 48000          # sample rate for the muxed AAC track (constant across chunks)
FIRST_CHUNK_TARGET = 0.1  # seconds → first chunk is a single segment (low TTFS)
DEFAULT_CHUNK_TARGET = 4.0  # seconds of source timeline per subsequent chunk
AUDIO_EXTENSIONS = ("mp3", "wav")


def _nearest_keyframe_in(keyframes, lo, hi):
    """Keyframe nearest the midpoint of the pause ``(lo, hi]``, or None.

    We only split a chunk where a keyframe actually falls inside an
    inter-segment pause — that keeps the cut out of speech so the silence slop
    between the pause boundary and the keyframe is inaudible (design §3.2). If
    no keyframe lands in the pause, the caller grows the chunk and tries the
    next pause (graceful degradation to coarser chunks / higher TTFS).
    """
    mid = 0.5 * (lo + hi)
    best, best_d = None, None
    for k in keyframes:
        if lo <= k <= hi:
            d = abs(k - mid)
            if best is None or d < best_d:
                best, best_d = k, d
    return best


def _plan_chunks(segments, keyframes, first_target, target):
    """Group segments into chunks and pick keyframe cut points between them.

    Returns ``(chunks, boundaries)`` where ``chunks`` is a list of segment-index
    lists and ``boundaries`` is the list of keyframe times to cut the video at
    (len == len(chunks) - 1).
    """
    if not segments:
        raise ValueError("No translated speech segments are available to stream.")
    chunks, boundaries = [], []
    group = [0]
    group_start = segments[0].start
    limit = first_target
    for i in range(1, len(segments)):
        kf = _nearest_keyframe_in(keyframes, segments[i - 1].end, segments[i].start)
        span_ok = (segments[i - 1].end - group_start) >= limit
        # boundaries must be strictly increasing
        new_boundary = kf is not None and (not boundaries or kf > boundaries[-1])
        if span_ok and new_boundary:
            boundaries.append(kf)
            chunks.append(group)
            group = [i]
            group_start = segments[i].start
            limit = target
        else:
            group.append(i)
    chunks.append(group)
    return chunks, boundaries


class StreamingDubber:
    """Serial streaming orchestrator. Reuses warm stage clients via the pool."""

    def __init__(self, config: TranslationPipelineConfig, repo, *,
                 chunk_target: float = DEFAULT_CHUNK_TARGET):
        self.config = config
        self.repo = repo
        self.chunk_target = chunk_target
        self.ff = FFmpegClient()
        self._stem_cache = {}  # name -> mono tensor at MIX_SR

    # -- front matter -------------------------------------------------------

    def _prepare(self):
        """Run demucs+ASR, translate, and set up per-segment TTS source files."""
        pipeline = VideoTranslationPipeline(config=self.config, file_repository=self.repo)
        pipeline._generate_asr()          # demucs (once, whole audio) + ASR + pauses
        pipeline._generate_translation()  # translate all segments
        vt = pipeline.video_translation
        pipeline._release_models()

        # TTS setup: resample vocals to the TTS rate and split per-segment source
        # clips (sets segment.source_file), mirroring TextToSpeechManager.synthesize
        # without running batch TTS — we synthesize per chunk instead.
        self.tts = TextToSpeechManager(
            self.config.public_id, self.repo, tts_sample_rate=44100,
            logger=pipeline.logger, device=self.config.device,
            tts_name=self.config.tts_model, speaker_voices=self.config.speaker_voices)
        self.vocals_path = vt.background_audio["vocals.wav"]
        db = AudioDubbingManager(self.repo)
        splitted = self.repo.subdir("splitted_audio")
        vt = db.split_audio_seconds(vt, self.vocals_path, splitted,
                                    sample_rate=self.tts.tts_sample_rate)
        self.tts.prepare_speakers(vt)

        self.vt = vt
        self.dub = VideoDubbingManager(self.repo, pipeline.logger)
        self.logger = pipeline.logger
        return vt

    # -- background mixing (non-mutating; merge_background rewrites stems) ---

    def _stem(self, name):
        """Load a background stem once, mono, resampled to MIX_SR, and cache it."""
        if name not in self._stem_cache:
            path = self.vt.background_audio[name]
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != MIX_SR:
                wav = T.Resample(sr, MIX_SR)(wav)
            self._stem_cache[name] = wav
        return self._stem_cache[name]

    def _mix_background(self, vocals, t0, t1, normalize=True, target_peak=0.95):
        """Mix dubbed vocals with the [t0,t1] slice of the continuous background.

        Slicing one continuous separation (rather than re-separating per chunk)
        keeps seams clean (design §3.3). ``vocals`` is [1, N] at MIX_SR.
        """
        n = vocals.shape[1]
        audio = vocals.clone()
        a0, a1 = int(round(t0 * MIX_SR)), int(round(t1 * MIX_SR))
        for name in ("drums.wav", "bass.wav", "other.wav"):
            if name not in self.vt.background_audio:
                continue
            stem = self._stem(name)
            seg = stem[:, a0:a1]
            if seg.shape[1] < n:
                seg = torch.cat([seg, torch.zeros((1, n - seg.shape[1]))], dim=1)
            elif seg.shape[1] > n:
                seg = seg[:, :n]
            audio = audio + seg
        if normalize:
            peak = torch.abs(audio).max()
            if peak > target_peak:
                audio = audio * (target_peak / peak)
        return audio

    # -- per-chunk audio ----------------------------------------------------

    def _build_chunk_audio(self, seg_indices, chunk_start, chunk_dur):
        """Build the chunk's dubbed+remixed audio, anchored to the video slice.

        The video-slice duration is the master clock: the dubbed vocals are
        padded/trimmed to exactly ``chunk_dur`` so each chunk re-anchors to the
        real video PTS and drift cannot accumulate across chunks (design §4.2).
        """
        segs = self.vt.translated_texts
        target_n = round(chunk_dur * MIX_SR)
        vocals = torch.zeros((1, target_n))
        covered = []
        for j, idx in enumerate(seg_indices):
            seg = segs[idx]
            if seg.start < chunk_start - 0.03 or seg.end > chunk_start + chunk_dur + 0.03:
                raise ValueError("A streaming video cut would truncate a spoken phrase.")
            next_start = segs[seg_indices[j + 1]].start if j + 1 < len(seg_indices) else seg.end
            gap = max(next_start - seg.end, 0.0)
            audio, rate, _ = self.dub.fit_segment(seg, self.tts.tts_sample_rate, gap)
            if rate != MIX_SR:
                audio = T.Resample(rate, MIX_SR)(audio)
            offset = max(0, round((seg.start - chunk_start) * MIX_SR))
            if offset + audio.shape[1] > target_n + 4:
                raise ValueError("Fitted streaming audio would exceed its video chunk.")
            n = min(audio.shape[1], target_n - offset)
            vocals[:, offset:offset + n] += audio[:, :n]
            covered.append((offset, max(offset + n, round((seg.end - chunk_start) * MIX_SR))))
        if "vocals.wav" in self.vt.background_audio:
            source = self._stem("vocals.wav")
            energy = source.float().pow(2).mean().sqrt()
            cursor = 0
            for start, end in sorted(covered) + [(target_n, target_n)]:
                if start > cursor:
                    pause = vocals[:, cursor:start].clone()
                    filled = self.dub._fill_gap_with_vocals(
                        pause, source, MIX_SR, MIX_SR,
                        chunk_start + cursor / MIX_SR, (start - cursor) / MIX_SR, energy)
                    vocals[:, cursor:start] = filled
                    if torch.any(filled):
                        self.logger.file_logger.warning("Original vocals retained in a streaming ASR gap; review the transcript.")
                cursor = min(target_n, max(cursor, end))
        return self._mix_background(vocals, chunk_start, chunk_start + chunk_dur).float()

    # -- main loop ----------------------------------------------------------

    def run(self):
        """Generator yielding one event dict per emitted segment.

        Event: {seg_index, uri, media_path, init_path, manifest_path,
                start, duration, n_segments, ttfs (first only)}.
        """
        t_start = time.perf_counter()
        base, ext = os.path.splitext(self.config.source_video_path)
        if ext.lower().lstrip(".") in AUDIO_EXTENSIONS:
            raise ValueError("Streaming path requires a video input (got audio).")

        self._prepare()
        segments = self.vt.translated_texts
        if not segments:
            return

        video = self.config.source_video_path
        keyframes = self.ff.probe_keyframes(video)
        fps = self.ff.probe_fps(video)
        video_dur = self.ff.probe_duration(video)
        self.logger.file_logger.info(
            f"[stream] {len(segments)} segments, {len(keyframes)} keyframes, fps={fps:.3f}")

        chunks, boundaries = _plan_chunks(
            segments, keyframes, FIRST_CHUNK_TARGET, self.chunk_target)

        # split video-only into clean keyframe-aligned pieces (one cheap copy pass)
        hls_dir = self.repo.subdir("hls")
        piece_dir = self.repo.subdir("video_pieces")
        frame_indices = [int(round(b * fps)) for b in boundaries]
        piece_pattern = os.path.join(piece_dir, "piece%03d.mp4")
        self.ff.split_video_at_frames(video, frame_indices, piece_pattern)

        manifest = HlsManifest(os.path.join(hls_dir, "index.m3u8"))
        cut_points = [0.0] + boundaries + [video_dur]
        init_path = os.path.join(hls_dir, "init.mp4")
        ttfs = None

        for c, seg_indices in enumerate(chunks):
            piece = os.path.join(piece_dir, f"piece{c:03d}.mp4")
            if not os.path.exists(piece):
                raise FileNotFoundError(f"Missing streaming video piece: {piece}")
            chunk_start = cut_points[c]
            chunk_dur = self.ff.probe_duration(piece)

            # synthesize each segment in this chunk (per-segment, no batching)
            for idx in seg_indices:
                self.tts.synthesize_segment(
                    segments[idx], target_lang=self.config.target_lang,
                    vocals_path=self.vocals_path)

            audio = self._build_chunk_audio(seg_indices, chunk_start, chunk_dur)
            audio_path = os.path.join(hls_dir, f"chunk{c:05d}.wav")
            torchaudio.save(audio_path, audio, MIX_SR)

            frag = os.path.join(hls_dir, f"frag{c:05d}.mp4")
            self.ff.mux_chunk_fmp4(piece, audio_path, frag, audio_sr=MIX_SR)
            init_bytes, media_bytes = split_fragmented_mp4(frag, timeline_offset=chunk_start)
            if c == 0:
                with open(init_path, "wb") as f:
                    f.write(init_bytes)
            seg_uri = f"seg{c:05d}.m4s"
            with open(os.path.join(hls_dir, seg_uri), "wb") as f:
                f.write(media_bytes)
            manifest.add_segment(seg_uri, chunk_dur)

            if ttfs is None:
                ttfs = time.perf_counter() - t_start
                self.logger.file_logger.info(f"[stream] TTFS={ttfs:.1f}s")

            yield {
                "seg_index": c,
                "uri": seg_uri,
                "media_path": os.path.join(hls_dir, seg_uri),
                "init_path": init_path,
                "manifest_path": manifest.playlist_path,
                "start": chunk_start,
                "duration": chunk_dur,
                "n_segments": len(seg_indices),
                **({"ttfs": ttfs} if c == 0 else {}),
            }

        manifest.finalize()


def stream_dub(input: dict, repo=None):
    """Build config + repo from a job ``input`` and stream dubbed HLS segments.

    Yields the same per-segment event dicts as ``StreamingDubber.run``. When
    ``repo`` is None a LocalOnlyFileRepository is used (no S3 needed) — this is
    what the local hls.js preview runner uses.
    """
    from langswap.file_repository import LocalOnlyFileRepository

    public_id = input.get("public_id", str(uuid.uuid4()))
    base_dir = input.get("base_dir", "data")
    if repo is None:
        repo = LocalOnlyFileRepository(public_id, base_dir)

    # The pipeline reads the source straight from this path (ASR extracts audio
    # from it), so no need to copy it into the repo dir first.
    source_path = input["source_video_path"]

    config = TranslationPipelineConfig(
        source_lang=input.get("source_language"),
        target_lang=input["target_language"],
        name=public_id,
        public_id=public_id,
        num_speakers=input.get("count_speakers"),
        source_video_path=source_path,
        base_dir=base_dir,
        device=input.get("device", "cuda"),
        tts_model=input.get("tts_engine", "openvoice"),
        dubbing_algo="speedup",
        watermark=False,
        skip_diarization=input.get("skip_diarization", False),
        asr_backend=input.get("asr_backend", "vad"),
        translation_backend=input.get("translation_backend", "marian"),
        speaker_voices=input.get("speaker_voices", {}),
    )
    dubber = StreamingDubber(config, repo,
                             chunk_target=input.get("chunk_target", DEFAULT_CHUNK_TARGET))
    yield from dubber.run()
