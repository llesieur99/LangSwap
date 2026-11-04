"""Tests for the streaming-dubbing path (Phase 0).

Two groups:

1. HLS / fMP4 mux mechanics — pure ffmpeg + box-splitting, no ML deps. These
   assert the streaming segmenter produces frame-conserved, independently
   decodable, hls.js-playable output.

2. Batch-vs-streaming audio consistency — the streaming and batch paths share
   ASR/translate/TTS and diverge ONLY in the merge/mux stage, so this feeds
   identical synthetic per-segment audio into both
   `VideoDubbingManager.merge_timestamps_speedup` (batch) and the streaming
   per-chunk assembly (`fit_segment` + chunking), and measures how far the two
   dubbed-audio timelines drift apart. Needs pandas/silero/pyrubberband; skipped
   if unavailable.

Run the full set with an interpreter that has the merge deps installed
(pandas, silero-vad, pyrubberband + the rubberband CLI):
    python -m pytest tests/test_streaming.py -v
The mechanics group also runs under a bare interpreter (ffmpeg only); the
consistency group skips itself when the merge deps are missing.
"""

import os
import struct
import subprocess
import tempfile

import pytest


# --------------------------------------------------------------------------
# torchaudio.load/save shim: torch 2.10 routes them through torchcodec, which
# is ABI-broken on this box. Fall back to soundfile so the real merge code's
# torchaudio.{load,save} calls work. No-op when torchcodec is healthy.
# --------------------------------------------------------------------------
def _install_torchaudio_soundfile_shim():
    try:
        import torch
        import torchaudio
        import soundfile as sf  # noqa: F401  (used by the shimmed _save/_load)
    except Exception:
        return
    try:
        sr = 8000
        torchaudio.save(os.path.join(tempfile.gettempdir(), "_ta_probe.wav"),
                        torch.zeros(1, sr), sr)
        return  # torchaudio.save works as-is
    except Exception:
        pass

    def _save(path, tensor, sample_rate, **_k):
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr.T  # [channels, N] -> [N, channels]
        sf.write(path, arr, int(sample_rate))

    def _load(path, **_k):
        arr, sr = sf.read(path, dtype="float32", always_2d=True)  # [N, ch]
        return torch.from_numpy(arr.T.copy()), sr  # -> [ch, N]

    torchaudio.save = _save
    torchaudio.load = _load


_install_torchaudio_soundfile_shim()


# ==========================================================================
# Group 1 — HLS / fMP4 mux mechanics (ffmpeg only)
# ==========================================================================

def _have(cmd):
    from shutil import which
    return which(cmd) is not None


pytestmark_ffmpeg = pytest.mark.skipif(
    not _have("ffmpeg") or not _have("ffprobe"), reason="ffmpeg/ffprobe not found")


@pytest.fixture(scope="module")
def synth_clip(tmp_path_factory):
    """6s H.264 clip, forced keyframe every 1s, with a tone audio track."""
    d = tmp_path_factory.mktemp("synth")
    src = str(d / "synth.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-c:v", "libx264", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
        "-force_key_frames", "expr:gte(t,n_forced*1)", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", src], check=True)
    return src, str(d)


def _frame_count(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.strip()
    return int(out)


@pytestmark_ffmpeg
def test_probe_keyframes_and_fps(synth_clip):
    from langswap.ml.ffmpeg import FFmpegClient
    src, _ = synth_clip
    ff = FFmpegClient()
    assert ff.probe_keyframes(src) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(ff.probe_fps(src) - 30.0) < 1e-6
    assert abs(ff.probe_duration(src) - 6.0) < 0.2


@pytestmark_ffmpeg
def test_split_is_frame_conserving(synth_clip):
    """Per-chunk -ss/-t cutting overlaps; the up-front frame split must not."""
    from langswap.ml.ffmpeg import FFmpegClient
    src, d = synth_clip
    ff = FFmpegClient()
    outdir = tempfile.mkdtemp(dir=d)
    fps = ff.probe_fps(src)
    # boundaries at keyframes 1.0 and 3.0 -> frames 30, 90 -> pieces 30/60/90
    ff.split_video_at_frames(src, [int(round(1.0 * fps)), int(round(3.0 * fps))],
                             os.path.join(outdir, "piece%03d.mp4"))
    pieces = sorted(p for p in os.listdir(outdir) if p.startswith("piece"))
    assert pieces == ["piece000.mp4", "piece001.mp4", "piece002.mp4"]
    counts = [_frame_count(os.path.join(outdir, p)) for p in pieces]
    assert counts == [30, 60, 90], counts
    assert sum(counts) == _frame_count(src) == 180  # zero overlap / zero loss


@pytestmark_ffmpeg
def test_empty_split_single_piece(synth_clip):
    from langswap.ml.ffmpeg import FFmpegClient
    src, d = synth_clip
    ff = FFmpegClient()
    outdir = tempfile.mkdtemp(dir=d)
    ff.split_video_at_frames(src, [], os.path.join(outdir, "piece%03d.mp4"))
    assert os.listdir(outdir) == ["piece000.mp4"]
    assert _frame_count(os.path.join(outdir, "piece000.mp4")) == 180


@pytestmark_ffmpeg
def test_full_hls_assembly_reassembles(synth_clip):
    """Pre-cut pieces -> per-chunk mux -> box split -> EVENT manifest, then read
    the playlist back: frame count and duration must be conserved."""
    from langswap.ml.ffmpeg import FFmpegClient
    from langswap.hls import split_fragmented_mp4, HlsManifest
    src, d = synth_clip
    ff = FFmpegClient()
    out = tempfile.mkdtemp(dir=d)
    fps = ff.probe_fps(src)
    ff.split_video_at_frames(src, [int(round(1.0 * fps)), int(round(3.0 * fps))],
                             os.path.join(out, "piece%03d.mp4"))
    man = HlsManifest(os.path.join(out, "index.m3u8"))
    inits = []
    for c in range(3):
        piece = os.path.join(out, f"piece{c:03d}.mp4")
        pdur = ff.probe_duration(piece)
        aud = os.path.join(out, f"a{c}.wav")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", f"sine=frequency={220*(c+1)}:duration={pdur}",
                        "-ar", "48000", "-ac", "2", aud], check=True)
        frag = os.path.join(out, f"frag{c}.mp4")
        ff.mux_chunk_fmp4(piece, aud, frag, audio_sr=48000)
        init, media = split_fragmented_mp4(frag)
        inits.append(init)
        if c == 0:
            with open(os.path.join(out, "init.mp4"), "wb") as f:
                f.write(init)
        with open(os.path.join(out, f"seg{c:05d}.m4s"), "wb") as f:
            f.write(media)
        man.add_segment(f"seg{c:05d}.m4s", pdur)
    man.finalize()

    text = open(os.path.join(out, "index.m3u8")).read()
    assert "#EXT-X-PLAYLIST-TYPE:EVENT" in text
    assert '#EXT-X-MAP:URI="init.mp4"' in text
    assert text.rstrip().endswith("#EXT-X-ENDLIST")
    assert text.count("seg") >= 3

    reasm = os.path.join(out, "reasm.mp4")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", os.path.join(out, "index.m3u8"), "-c", "copy", reasm], check=True)
    assert _frame_count(reasm) == 180  # no dup/loss across the streamed timeline


def _read_tfdt_bases(frag_path):
    """All moof/traf/tfdt baseMediaDecodeTime values in a fragmented mp4."""
    data = open(frag_path, "rb").read()

    def boxes(s, e):
        o = s
        while o + 8 <= e:
            sz = struct.unpack(">I", data[o:o + 4])[0]
            typ = data[o + 4:o + 8].decode("latin1")
            hdr = 8
            if sz == 1:
                sz = struct.unpack(">Q", data[o + 8:o + 16])[0]; hdr = 16
            elif sz == 0:
                sz = e - o
            yield typ, o, o + sz, hdr
            o += sz

    bases = []
    for t, s, e, h in boxes(0, len(data)):
        if t == "moof":
            for t2, s2, e2, h2 in boxes(s + h, e):
                if t2 == "traf":
                    for t3, s3, e3, h3 in boxes(s2 + h2, e2):
                        if t3 == "tfdt":
                            ver = data[s3 + h3]
                            off = s3 + h3 + 4
                            base = (struct.unpack(">Q", data[off:off + 8])[0] if ver == 1
                                    else struct.unpack(">I", data[off:off + 4])[0])
                            bases.append(base)
    return bases


@pytestmark_ffmpeg
def test_segments_lack_absolute_timeline_REGRESSION(synth_clip):
    """REGRESSION GUARD for review finding #1: every media segment currently
    resets its first moof tfdt to ~0 instead of carrying an absolute timeline.
    This documents the known defect; flip the assertion once -output_ts_offset
    (or running-offset tfdt) is added so segments stitch in MSE."""
    from langswap.ml.ffmpeg import FFmpegClient
    src, d = synth_clip
    ff = FFmpegClient()
    out = tempfile.mkdtemp(dir=d)
    fps = ff.probe_fps(src)
    ff.split_video_at_frames(src, [int(round(2.0 * fps))],
                             os.path.join(out, "p%03d.mp4"))
    first_bases = []
    for c in range(2):
        piece = os.path.join(out, f"p{c:03d}.mp4")
        pdur = ff.probe_duration(piece)
        aud = os.path.join(out, f"a{c}.wav")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", f"sine=frequency=330:duration={pdur}",
                        "-ar", "48000", "-ac", "2", aud], check=True)
        frag = os.path.join(out, f"f{c}.mp4")
        ff.mux_chunk_fmp4(piece, aud, frag, audio_sr=48000)
        first_bases.append(_read_tfdt_bases(frag)[0])
    # Known defect: both segments start at 0 (no absolute placement).
    assert first_bases[0] == 0
    assert first_bases[1] == 0, (
        "Segment 2 now carries an absolute tfdt — finding #1 is fixed; "
        "update this regression guard.")


# ==========================================================================
# Group 2 — batch vs streaming audio consistency (needs merge deps)
# ==========================================================================

def _require_merge_deps():
    for m in ("pandas", "silero_vad", "pyrubberband"):
        pytest.importorskip(m)


class _Seg:
    """Stand-in for TranslatedTextedSegment with the fields fit_segment needs."""
    def __init__(self, start, end, source_file, generated_file):
        self.start = start
        self.end = end
        self.source_file = source_file
        self.generated_file = generated_file
        self.translation = ""
        self.text = ""
        self.speaker = "S0"


def _tone(path, dur, sr, freq=220, amp=0.3):
    import torch
    import torchaudio
    t = torch.arange(0, int(dur * sr)) / sr
    wav = (amp * torch.sin(2 * 3.14159265 * freq * t)).unsqueeze(0)
    torchaudio.save(path, wav, sr)
    return path


def _make_segments(tmp, specs, sr):
    """specs: list of (start, end). generated audio == source-window length,
    continuous tone (no internal pauses, peak < 0.95) so change_pauses is a
    no-op and time_stretch never fires — isolating the assembly math."""
    segs = []
    for i, (s, e) in enumerate(specs):
        srcf = os.path.join(tmp, f"src_{i}.wav")
        genf = os.path.join(tmp, f"gen_{i}.wav")
        _tone(srcf, e - s, sr, freq=200 + 30 * i)
        _tone(genf, e - s, sr, freq=200 + 30 * i)
        segs.append(_Seg(s, e, srcf, genf))
    return segs


class _FakeLogger:
    class _F:
        def debug(self, *a, **k): pass
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
    def __init__(self):
        self.file_logger = self._F()
    def log_json(self, *a, **k): pass


def _streaming_full_audio(dub, segs, chunks, cut_points, sr, mix_sr):
    """Replicate streaming.StreamingDubber._build_chunk_audio (vocals only, no
    background, no per-chunk normalize since peak<0.95) for every chunk and
    concatenate, yielding the full streaming dubbed-vocals timeline."""
    import torch
    import torchaudio.transforms as T
    full = []
    for c, seg_indices in enumerate(chunks):
        chunk_start = cut_points[c]
        chunk_dur = cut_points[c + 1] - cut_points[c]
        parts = []
        first = segs[seg_indices[0]]
        lead = max(first.start - chunk_start, 0.0)
        sr_gen = None
        for j, idx in enumerate(seg_indices):
            seg = segs[idx]
            if j + 1 < len(seg_indices):
                nxt = segs[seg_indices[j + 1]]
                available_pause = max(nxt.start - seg.end, 0.0)
            else:
                available_pause = 0.0
            audio, sr_gen, pause_after = dub.fit_segment(
                seg, source_sr=sr, available_pause=available_pause)
            parts.append(audio)
            if j + 1 < len(seg_indices):
                parts.append(torch.zeros((1, int(pause_after * sr_gen))))
        if sr_gen is None:
            sr_gen = mix_sr
        vocals = torch.cat([torch.zeros((1, int(lead * sr_gen)))] + parts, dim=1)
        if sr_gen != mix_sr:
            vocals = T.Resample(sr_gen, mix_sr)(vocals)
        target_n = int(round(chunk_dur * mix_sr))
        if vocals.shape[1] < target_n:
            vocals = torch.cat([vocals, torch.zeros((1, target_n - vocals.shape[1]))], dim=1)
        else:
            vocals = vocals[:, :target_n]
        full.append(vocals)
    return torch.cat(full, dim=1)


def test_streaming_matches_batch_well_behaved(tmp_path):
    """In the well-behaved case (no stretch needed, chunk cuts land exactly on
    inter-segment pauses), the streaming dubbed-audio timeline should match the
    batch timeline within a tight tolerance — segments stay on their source
    timestamps in both."""
    _require_merge_deps()
    import torch
    import torchaudio
    from langswap.ml.video_dubbing_manager import VideoDubbingManager

    sr = 24000
    mix_sr = sr  # isolate the algorithm from resampling
    tmp = str(tmp_path)
    # segments with clean pauses between them; total span 6.0s
    specs = [(0.5, 1.5), (2.0, 3.0), (3.5, 4.5), (5.0, 5.8)]
    segs = _make_segments(tmp, specs, sr)

    # full source vocals track = 6.0s (the batch "target" length)
    vocals = os.path.join(tmp, "vocals.wav")
    _tone(vocals, 6.0, sr, freq=110, amp=0.05)

    class VT:
        translated_texts = segs
    dub = VideoDubbingManager.__new__(VideoDubbingManager)
    dub._file_repository = None
    dub.logger = _FakeLogger()
    from langswap.ml.video_dubbing_manager import _get_vad_model
    dub.model_vad = _get_vad_model()

    audio_b, sr_b = dub.merge_timestamps_speedup(VT, vocals)
    assert sr_b == sr

    # chunk cuts placed in the middle of each inter-segment pause (= keyframe
    # times in a CFR clip), first chunk = 1 segment.
    chunks = [[0], [1], [2], [3]]
    cut_points = [0.0, 1.75, 3.25, 4.75, 6.0]
    audio_s = _streaming_full_audio(dub, segs, chunks, cut_points, sr, mix_sr)

    len_b = audio_b.shape[1] / sr
    len_s = audio_s.shape[1] / mix_sr
    # total lengths agree to well under a frame-time
    assert abs(len_b - len_s) < 0.05, (len_b, len_s)

    # align lengths and measure energy envelope correlation over 50ms windows
    n = min(audio_b.shape[1], audio_s.shape[1])
    a = audio_b[0, :n].abs()
    b = audio_s[0, :n].abs()
    win = int(0.05 * sr)
    ea = a.unfold(0, win, win).mean(1)
    eb = b.unfold(0, win, win).mean(1)
    ea = (ea - ea.mean()) / (ea.std() + 1e-8)
    eb = (eb - eb.mean()) / (eb.std() + 1e-8)
    corr = float((ea * eb).mean())
    print(f"\n[consistency] batch={len_b:.3f}s streaming={len_s:.3f}s "
          f"envelope_corr={corr:.3f}")
    # speech lands in the same time windows in both paths
    assert corr > 0.85, f"envelope correlation too low: {corr}"


def test_streaming_energy_matches_batch_on_overrun(tmp_path):
    """Even when every segment's dubbed audio overruns its source window (so the
    batch path's global time_stretch fires), the streaming per-chunk assembly
    produces a near-identical total — per-segment fit_segment already compresses
    the overrun, so there is little left for the global pass to do. This is the
    core 'streaming == batch' check on a realistic input."""
    _require_merge_deps()
    import torch
    from langswap.ml.video_dubbing_manager import VideoDubbingManager, _get_vad_model

    sr = 24000
    tmp = str(tmp_path)
    specs = [(0.0, 1.0), (1.5, 2.5), (3.0, 4.0)]  # 1s segments, 0.5s pauses
    segs = []
    for i, (s, e) in enumerate(specs):
        srcf = _tone(os.path.join(tmp, f"s{i}.wav"), e - s, sr, freq=200 + 30 * i)
        genf = _tone(os.path.join(tmp, f"g{i}.wav"), 1.8, sr, freq=200 + 30 * i)  # overlong
        segs.append(_Seg(s, e, srcf, genf))
    vocals = _tone(os.path.join(tmp, "voc.wav"), 4.0, sr, freq=110, amp=0.05)

    class VT:
        translated_texts = segs
    dub = VideoDubbingManager.__new__(VideoDubbingManager)
    dub._file_repository = None
    dub.logger = _FakeLogger()
    dub.model_vad = _get_vad_model()

    audio_b, _ = dub.merge_timestamps_speedup(VT, vocals)
    # one chunk spanning the whole 4.0s source timeline
    audio_s = _streaming_full_audio(dub, segs, [[0, 1, 2]], [0.0, 4.0], sr, sr)

    len_b, len_s = audio_b.shape[1] / sr, audio_s.shape[1] / sr
    e_b, e_s = float(audio_b.abs().sum()), float(audio_s.abs().sum())
    ratio = e_s / e_b
    print(f"\n[consistency-overrun] batch={len_b:.3f}s streaming={len_s:.3f}s "
          f"energy_ratio={ratio:.3f}")
    assert abs(len_b - len_s) < 0.05
    assert 0.95 < ratio < 1.05, f"streaming energy diverged from batch: {ratio}"


def test_streaming_trims_when_video_window_shorter_than_audio(tmp_path):
    """Deterministic guard for review finding #3: streaming has no global
    stretch, so if a chunk's VIDEO-slice duration is shorter than its fitted
    dubbed audio (e.g. a keyframe boundary lands before the segment's content
    ends), the trailing audio is hard-trimmed (vocals[:, :target_n]) and lost.
    Uses gen==source length so no time_stretch is involved — the loss is purely
    the trim. Batch, anchored to the source-vocals length, would keep it."""
    _require_merge_deps()
    import torch
    from langswap.ml.video_dubbing_manager import VideoDubbingManager, _get_vad_model

    sr = 24000
    tmp = str(tmp_path)
    srcf = _tone(os.path.join(tmp, "s.wav"), 1.0, sr, amp=0.3)
    genf = _tone(os.path.join(tmp, "g.wav"), 1.0, sr, amp=0.3)  # well-fit, no stretch
    seg = _Seg(0.0, 1.0, srcf, genf)
    dub = VideoDubbingManager.__new__(VideoDubbingManager)
    dub._file_repository = None
    dub.logger = _FakeLogger()
    dub.model_vad = _get_vad_model()

    full_energy = float(_streaming_full_audio(dub, [seg], [[0]], [0.0, 1.0], sr, sr).abs().sum())
    # force the video window to 0.5s (< the 1.0s of dubbed content)
    trimmed_energy = float(_streaming_full_audio(dub, [seg], [[0]], [0.0, 0.5], sr, sr).abs().sum())
    kept = trimmed_energy / full_energy
    print(f"\n[finding#3] video window 0.5s < audio 1.0s -> streaming kept "
          f"{kept:.0%} of the dubbed energy (rest trimmed, no global compensation)")
    assert kept < 0.7, "expected the short video window to truncate trailing audio"
