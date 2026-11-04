"""Translation editor: reload a finished job and re-dub edited segments.

The full pipeline writes every artifact under ``data/<public_id>/`` (config,
per-segment source/generated audio, separated background stems, the segment
timing/translation JSON and the muxed video).  This module reconstructs a
``VideoTranslation`` straight from those files — without re-running ASR,
translation, Demucs separation or the first TTS pass — so an editor UI can show
the segments, let a user fix the timing, speaker, recognised source text and
translation of each line, and re-dub only the changed lines before re-muxing.

Re-dub reuses the existing stage helpers: when a segment's timing changes the
source reference slice is re-cut from the vocals stem
(``AudioDubbingManager.split_audio_seconds``), changed lines are re-synthesised
(``TextToSpeechManager.synthesize_segment``), and the whole audio is re-mixed and
re-muxed (``VideoTranslationPipeline._merge``) — the same primitives the pipeline
and the RunPod ``process_update_translation`` handler use.
"""

from __future__ import annotations

import dataclasses
import json
import os
import math
from dataclasses import dataclass
from typing import List, Optional

from langswap.file_repository import LocalOnlyFileRepository
from langswap.ml.text_to_speech_service.audio_dubbing_manager import AudioDubbingManager
from langswap.ml.text_to_speech_service import TextToSpeechManager
from langswap.pipeline_models.models import (
    RemoteFile,
    TextedSegment,
    TranslatedTextedSegment,
    TranslationPipelineConfig,
    VideoTranslation,
)
from langswap.translation_pipeline import VideoTranslationPipeline

# Demucs stems written under background_files/ by the ASR stage.  "vocals.wav" is
# the one the dubbing merge mixes the generated speech over; the rest are added
# back as the music/effects bed.
_BG_STEMS = ("vocals.wav", "bass.wav", "drums.wav", "other.wav")


def _seg_audio_name(start: float, end: float) -> str:
    """Per-segment wav filename, matching TextToSpeechManager.synthesize_segment."""
    return f"{start}_{end}.wav"


def list_jobs(base_dir: str) -> List[str]:
    """Return the public_ids under ``base_dir`` that hold an editable job
    (segment timings + translations on disk)."""
    if not os.path.isdir(base_dir):
        return []
    jobs = []
    for name in sorted(os.listdir(base_dir)):
        job_dir = os.path.join(base_dir, name)
        if os.path.isfile(os.path.join(job_dir, "translations.json")) and os.path.isfile(
            os.path.join(job_dir, "splitted_sentences_pauses.json")
        ):
            jobs.append(name)
    return jobs


def _load_config(job_dir: str, base_dir: str, public_id: str) -> TranslationPipelineConfig:
    """Load config.json, tolerating keys from older pipeline versions (e.g.
    ``voice_conv``/``eleven_api_token``) by keeping only current dataclass fields."""
    with open(os.path.join(job_dir, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    known = {field.name for field in dataclasses.fields(TranslationPipelineConfig)}
    cfg = {k: v for k, v in raw.items() if k in known}
    # The job may have been moved between data dirs; trust the caller's location.
    cfg["base_dir"] = base_dir
    cfg["public_id"] = public_id
    return TranslationPipelineConfig(**cfg)


@dataclass
class EditorSession:
    """An opened job: the reconstructed state plus the pipeline used to re-dub."""

    public_id: str
    base_dir: str
    job_dir: str
    config: TranslationPipelineConfig
    pipeline: VideoTranslationPipeline
    video_translation: VideoTranslation

    @property
    def segments(self) -> List[TranslatedTextedSegment]:
        return self.video_translation.translated_texts

    @property
    def source_video_path(self) -> Optional[str]:
        src = self.config.source_video_path
        return str(src) if src and os.path.exists(str(src)) else None

    @property
    def result_video_path(self) -> Optional[str]:
        pv = self.video_translation.processed_video
        if pv and pv.file_path and os.path.exists(pv.file_path):
            return pv.file_path
        return None


def load_job(base_dir: str, public_id: str) -> EditorSession:
    """Rebuild a ``VideoTranslation`` from a finished job dir, ready to edit.

    Reads the segment timing/speaker list (``splitted_sentences_pauses.json``)
    and the translations (``translations.json``), wires each segment to its
    on-disk source/generated wav, and points ``background_audio`` at the Demucs
    stems.  No models are loaded here — that happens lazily in ``apply_edits``.
    """
    if not public_id or public_id in (".", "..") or any(c in public_id for c in "/\\:"):
        raise ValueError("The job id must be a single directory name.")
    job_dir = os.path.realpath(os.path.join(base_dir, public_id))
    base_path = os.path.realpath(base_dir)
    if os.path.commonpath((base_path, job_dir)) != base_path:
        raise ValueError("The job directory must stay within the selected data directory.")
    if not os.path.isdir(job_dir):
        raise FileNotFoundError(f"Job directory not found: {job_dir}")

    with open(os.path.join(job_dir, "splitted_sentences_pauses.json"), encoding="utf-8") as f:
        seg_rows = json.load(f)
    with open(os.path.join(job_dir, "translations.json"), encoding="utf-8") as f:
        translation_rows = json.load(f)

    if len(seg_rows) != len(translation_rows):
        raise ValueError(
            f"Segment/translation count mismatch for {public_id}: "
            f"{len(seg_rows)} segments vs {len(translation_rows)} translations."
        )

    config = _load_config(job_dir, base_dir, public_id)

    splitted_dir = os.path.join(job_dir, "splitted_audio")
    generated_dir = os.path.join(job_dir, "generated_audio")

    recognized_texts: List[TextedSegment] = []
    translated_texts: List[TranslatedTextedSegment] = []
    for seg, tr in zip(seg_rows, translation_rows):
        wav_name = _seg_audio_name(seg["start"], seg["end"])
        recognized_texts.append(
            TextedSegment(
                text=seg["text"], start=seg["start"], end=seg["end"],
                speaker=seg.get("speaker", "SPEAKER_00"),
            )
        )
        translated_texts.append(
            TranslatedTextedSegment(
                text=seg["text"],
                start=seg["start"],
                end=seg["end"],
                translation=tr["translation"],
                source_file=os.path.join(splitted_dir, wav_name),
                generated_file=os.path.join(generated_dir, wav_name),
                speaker=seg.get("speaker", "SPEAKER_00"),
            )
        )

    # background_audio is a name->path map at runtime (the merge calls
    # torchaudio.load on the value), despite the dataclass' RemoteFile hint.
    bg_dir = os.path.join(job_dir, "background_files")
    background_audio = {
        stem: os.path.join(bg_dir, stem)
        for stem in _BG_STEMS
        if os.path.exists(os.path.join(bg_dir, stem))
    }

    lang_code = None
    lang_path = os.path.join(job_dir, "lang_detect_info.json")
    if os.path.exists(lang_path):
        with open(lang_path, encoding="utf-8") as f:
            lang_code = json.load(f).get("detected_language")

    result_file = os.path.join(job_dir, "resulted_video.mp4")
    video_translation = VideoTranslation(
        public_id=public_id,
        source_lang_code=lang_code,
        source_file=RemoteFile(file_path=str(config.source_video_path), name=public_id),
        background_audio=background_audio,
        recognized_texts=recognized_texts,
        translated_texts=translated_texts,
        processed_video=(
            RemoteFile(file_path=result_file, name="resulted_video.mp4")
            if os.path.exists(result_file)
            else None
        ),
    )

    repo = LocalOnlyFileRepository(public_id, base_dir)
    pipeline = VideoTranslationPipeline(config=config, file_repository=repo)
    pipeline.video_translation = video_translation

    return EditorSession(
        public_id=public_id,
        base_dir=base_dir,
        job_dir=job_dir,
        config=config,
        pipeline=pipeline,
        video_translation=video_translation,
    )


@dataclass
class SegmentEdit:
    """One row of edits: any field the user may have changed."""

    start: float
    end: float
    speaker: str
    text: str
    translation: str


def _coerce_edit(row) -> SegmentEdit:
    """Build a SegmentEdit from a raw table row [#, start, end, speaker, text, translation]."""
    try:
        start = float(row[1])
        end = float(row[2])
    except (TypeError, ValueError) as e:
        raise ValueError(f"start/end must be numbers (got {row[1]!r}, {row[2]!r}).") from e
    if not (math.isfinite(start) and math.isfinite(end) and end > start >= 0):
        raise ValueError(f"Each segment needs 0 <= start < end (got start={start}, end={end}).")
    return SegmentEdit(
        start=start,
        end=end,
        speaker=str(row[3]).strip() or "SPEAKER_00",
        text=str(row[4]),
        translation=str(row[5]),
    )


def _persist_segments(session: EditorSession) -> None:
    """Write the edited segments back to the job's JSON sidecars so a reload
    reflects the changes (translations + timing/speaker/source text)."""
    segs = session.video_translation.translated_texts
    session.pipeline.logger.log_json(
        file_name="translations.json",
        data=[{"translation": s.translation, "text": s.text} for s in segs],
    )
    from langswap.ml.translation_service.translation_manager import translation_fingerprint
    session.pipeline.logger.log_json(file_name="translations.cache.json", data={
        "fingerprint": translation_fingerprint(
            session.video_translation.recognized_texts,
            session.config.source_lang or session.video_translation.source_lang_code,
            session.config.target_lang, session.config.translation_backend),
    })
    seg_rows = [
        {"text": s.text, "start": s.start, "end": s.end, "speaker": s.speaker}
        for s in segs
    ]
    # raw_transcribed_info.json + splitted_sentences_pauses.json share this shape
    # and are what the ASR cache / the editor reload read back.
    session.pipeline.logger.log_json(file_name="raw_transcribed_info.json", data=seg_rows)
    session.pipeline.logger.log_json(file_name="splitted_sentences_pauses.json", data=seg_rows)


def _parse_key(value, n: int) -> Optional[int]:
    """The '#' cell is a stable 1-based segment id (locked for existing rows).
    Return the 0-based index of the existing segment, or None for a new row
    (blank/unknown id), so add/delete can be told apart from edits."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none", "new"):
        return None
    try:
        k = int(float(s)) - 1
    except (TypeError, ValueError):
        return None
    return k if 0 <= k < n else None


def _make_segments(job_dir: str, edit: "SegmentEdit"):
    """Build a (recognized, translated) segment pair wired to its on-disk wavs."""
    wav = _seg_audio_name(edit.start, edit.end)
    rec = TextedSegment(text=edit.text, start=edit.start, end=edit.end, speaker=edit.speaker)
    tr = TranslatedTextedSegment(
        text=edit.text, start=edit.start, end=edit.end, translation=edit.translation,
        source_file=os.path.join(job_dir, "splitted_audio", wav),
        generated_file=os.path.join(job_dir, "generated_audio", wav),
        speaker=edit.speaker,
    )
    return rec, tr


def apply_edits(session: EditorSession, rows: List[list]) -> EditorSession:
    """Reconcile the edited segment table against the job and re-dub.

    The table is the desired final set of segments.  The '#' column is a stable
    id: rows keep their existing segment, blank-'#' rows are **added**, and
    existing ids absent from the table are **deleted**.  Segments that are new or
    whose timing changed get their source reference slice re-cut from the vocals
    stem and are re-synthesised; rows whose source text/translation changed are
    re-synthesised too.  Any add/delete/re-synth re-mixes the track and re-muxes
    the video; speaker edits also regenerate the voice. The session,
    the JSON sidecars and the SRTs are updated.
    """
    old_rec = session.video_translation.recognized_texts
    old_tr = session.video_translation.translated_texts
    n_old = len(old_tr)
    job_dir = session.pipeline._file_repository.directory

    final_rec: List[TextedSegment] = []
    final_tr: List[TranslatedTextedSegment] = []
    synth: List[int] = []   # indices (into final_tr) needing TTS
    slice_: List[int] = []  # indices needing a (re)cut source slice
    kept: set[int] = set()
    changed = False

    for row in rows:
        key = _parse_key(row[0], n_old)
        edit = _coerce_edit(row)
        rec, tr = _make_segments(job_dir, edit)
        idx = len(final_tr)
        final_rec.append(rec)
        final_tr.append(tr)

        if key is None:  # newly added segment
            if not edit.translation.strip():
                raise ValueError("A new segment needs a non-empty translation.")
            synth.append(idx)
            slice_.append(idx)
            changed = True
            continue

        if key in kept:
            raise ValueError("An existing segment cannot appear twice in an edit.")
        kept.add(key)
        base = old_tr[key]
        timing_changed = (edit.start != base.start) or (edit.end != base.end)
        text_changed = edit.text != base.text
        tr_changed = edit.translation != base.translation
        speaker_changed = edit.speaker != base.speaker
        if timing_changed:
            slice_.append(idx)
        if timing_changed or text_changed or tr_changed or speaker_changed:
            if not edit.translation.strip():
                raise ValueError("A segment needs a non-empty translation.")
            synth.append(idx)
        if timing_changed or text_changed or tr_changed or speaker_changed:
            changed = True

    deleted = [k for k in range(n_old) if k not in kept]
    if deleted:
        changed = True
    if not changed:
        return session
    if not final_tr:
        raise ValueError("At least one segment is required.")
    if len({(s.start, s.end) for s in final_tr}) != len(final_tr):
        raise ValueError("Segments need distinct time windows to keep their audio files separate.")

    # Keep segments time-ordered (the merge places audio by start time); remap
    # the synth/slice index sets through the sort.
    order = sorted(range(len(final_tr)), key=lambda i: final_tr[i].start)
    synth_set, slice_set = set(synth), set(slice_)
    final_rec = [final_rec[i] for i in order]
    final_tr = [final_tr[i] for i in order]
    synth = [new_i for new_i, old_i in enumerate(order) if old_i in synth_set]
    slice_ = [new_i for new_i, old_i in enumerate(order) if old_i in slice_set]

    vt = session.video_translation
    vt.recognized_texts = final_rec
    vt.translated_texts = final_tr
    session.pipeline.video_translation = vt

    if synth or deleted:
        session.pipeline._release_models()
        vocals_path = vt.background_audio["vocals.wav"]
        tts_manager = TextToSpeechManager(
            session.config.public_id,
            session.pipeline._file_repository,
            device=session.config.device,
            logger=session.pipeline.logger,
            tts_sample_rate=24000,
            tts_name=session.config.tts_model,
            speaker_voices=session.config.speaker_voices,
        )
        # split_audio_seconds writes {start}_{end}.wav (skipping existing files)
        # and points each segment.source_file at it — only new windows are cut.
        if slice_:
            AudioDubbingManager(session.pipeline._file_repository).split_audio_seconds(
                vt, vocals_path,
                session.pipeline._file_repository.subdir("splitted_audio"),
                sample_rate=tts_manager.tts_sample_rate,
            )
        tts_manager.prepare_speakers(vt)
        # A timing/speaker edit may change the shared reference. Validate all
        # segment caches; only those whose voice/text identity changed regenerate.
        for i in range(len(final_tr)):
            tts_manager.synthesize_segment(
                final_tr[i], target_lang=session.config.target_lang, vocals_path=vocals_path,
            )

    if synth or deleted:
        video_translation = session.pipeline._merge(session.config.dubbing_algo)
        session.video_translation = video_translation
        session.pipeline.video_translation = video_translation

    _persist_segments(session)
    session.pipeline.generate_srt_files()
    return session
