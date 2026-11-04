"""VAD-first ASR backend: faster-whisper (transcription + word timestamps) +
Silero VAD (segment boundaries).  No forced aligner, no per-language model.

Why this is the default: the dubbing pipeline consumes only segment {start,end}
(word-level data is dropped downstream).  A forced aligner (wav2vec2 per-language,
or the 1.8 GB Qwen aligner) is heavier than that job needs.  Silero VAD (~1.4 MB,
language-agnostic, CPU) places segment boundaries at real speech edges as tightly
as the Qwen aligner and more reliably than WhisperX — whose boundaries are only
Whisper's own segmentation, which merges/over-splits inconsistently.

Pipeline: faster-whisper transcribes the whole clip with word timestamps; Silero
VAD finds speech regions; each word is assigned to its VAD region by midpoint;
segments = VAD regions (accurate edges) carrying their whisper words' text.

Same constructor signature, __enter__/__exit__, and transcribe() return type
as the other ASR backends.
"""

import logging
from typing import Optional

from langswap.model_config import MODEL_WEIGHTS_DIR
from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code

# Reuse the structures + pause threshold from the shared types module (single source).
from langswap.ml.speech_to_text_service.asr_types import (
    Output,
    TranscriptionData,
    PAUSE_THRESHOLD_SECONDS,
)
import cattrs

logger = logging.getLogger(__name__)

_VAD_MODEL = None


def _get_vad_model():
    """Module-level cached Silero VAD (matches video_dubbing_manager)."""
    global _VAD_MODEL
    if _VAD_MODEL is None:
        from silero_vad import load_silero_vad
        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


# Splitting constants for music-bridged VAD regions.  Background music keeps
# Silero VAD from seeing silence, so one region can span minutes of narration
# bridged by music.  Without a split the whole region becomes one dubbing slot:
# all TTS is placed at the region start and the rest is silence-padded, so the
# voice disappears for the remainder of the slot.  Splitting at sentence-level
# word gaps keeps each sub-segment at its original position in the timeline.
_SEGMENT_MAX_WORD_GAP = 1.5   # split when a word gap reaches this (seconds)
_SEGMENT_MAX_SPAN = 15.0      # split a longer span at the next gap of at least...
_SEGMENT_MIN_SPLIT_GAP = 0.3  # ...this (seconds)


def _split_at_pauses(words: list[dict]) -> list[list[dict]]:
    """Split a VAD-region word bucket into sub-segments at meaningful gaps."""
    if not words:
        return []
    parts = []
    cur = [words[0]]
    for w in words[1:]:
        gap = float(w.get("start", 0.0) or 0.0) - float(cur[-1].get("end", 0.0) or 0.0)
        span = float(w.get("end", 0.0) or 0.0) - float(cur[0].get("start", 0.0) or 0.0)
        if gap >= _SEGMENT_MAX_WORD_GAP or (span > _SEGMENT_MAX_SPAN and gap >= _SEGMENT_MIN_SPLIT_GAP):
            parts.append(cur)
            cur = [w]
        else:
            cur.append(w)
    parts.append(cur)
    return parts


def _group_words_by_vad(words: list[dict], regions: list[tuple]) -> list[dict]:
    """Group whisper words into segments by VAD speech region.

    Each word is assigned to the region containing its midpoint, or — if it falls
    in a gap (whisper text in a VAD-silence) — to the temporally nearest region,
    so no transcript text is lost.  Segment start/end come from the VAD region
    (the accurate boundary); text/words come from whisper.
    """
    if not regions:
        if not words:
            return []
        # A missed VAD region must still preserve speaker turns and narration gaps.
        return [_seg_from_words(sub, part["speaker"])
                for part in _split_by_speaker(words)
                for sub in _split_at_pauses(part["words"])]

    buckets: list[list[dict]] = [[] for _ in regions]
    for w in words:
        mid = (w.get("start", 0.0) + w.get("end", 0.0)) / 2.0
        best, best_d = 0, float("inf")
        for i, (rs, re) in enumerate(regions):
            if rs <= mid <= re:
                best, best_d = i, -1.0
                break
            d = rs - mid if mid < rs else mid - re
            if d < best_d:
                best, best_d = i, d
        buckets[best].append(w)

    segments = []
    for (rs, re), bucket in zip(regions, buckets):
        if not bucket:
            continue
        parts = [_seg_from_words(sub, part["speaker"])
                 for part in _split_by_speaker(bucket)
                 for sub in _split_at_pauses(part["words"])]
        # Ordinary unsplit speech keeps Silero's accurate edges. Split music
        # regions use word edges, so music padding does not fill the speech gaps.
        if len(parts) == 1:
            part = parts[0]
            if part["start"] - rs < _SEGMENT_MAX_WORD_GAP:
                part["start"] = min(rs, part["start"])
            if re - part["end"] < _SEGMENT_MAX_WORD_GAP:
                part["end"] = max(re, part["end"])
        segments.extend(parts)
    return segments


def _split_by_speaker(bucket: list[dict]) -> list[dict]:
    """One VAD region may contain several people; split on speaker changes."""
    if not bucket:
        return []
    out = []
    cur = [bucket[0]]
    spk = bucket[0].get("speaker", "SPEAKER_00")
    for w in bucket[1:]:
        wspk = w.get("speaker", "SPEAKER_00")
        if wspk != spk:
            out.append(_seg_from_words(cur, spk))
            cur = [w]
            spk = wspk
        else:
            cur.append(w)
    out.append(_seg_from_words(cur, spk))
    return out


def _seg_from_words(bucket: list[dict], speaker: str) -> dict:
    return {
        "text": " ".join(w["word"] for w in bucket).strip(),
        "start": float(bucket[0].get("start", 0.0)),
        "end": max(float(w.get("end", 0.0)) for w in bucket),
        "words": bucket,
        "speaker": speaker or "SPEAKER_00",
    }


class VADWhisperASR:
    """faster-whisper transcription + Silero VAD segmentation (no forced aligner)."""

    def __init__(
        self,
        device: str,
        language: str,
        skip_diarization: bool = False,
        whisper_model_id: Optional[str] = None,
    ) -> None:
        self.device = device
        self.skip_diarization = skip_diarization
        self.whisper_model_id = whisper_model_id or "large-v3"

        if language is not None:
            self.language = map_language_to_code(language, system="whisper")
        else:
            self.language = None

        self.asr_model = None
        self.vad_model = None
        self._embed_diarize = not skip_diarization

        self.load_models()

    def __enter__(self):
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # warm reuse is always on; keep models resident across jobs
        return False

    def load_models(self):
        if getattr(self, "asr_model", None) is not None:
            return
        self._load_asr_model()
        self.vad_model = _get_vad_model()

    def _load_asr_model(self):
        import faster_whisper
        # faster-whisper's CTranslate2 backend needs the CUDA 12 libs (libcublas.so.12,
        # cuDNN 9).  Where those are present (cu12 libs on LD_LIBRARY_PATH) the GPU is
        # used; otherwise ASR runs on CPU (int8).
        device = "cuda" if str(self.device).startswith("cuda") else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.asr_model = faster_whisper.WhisperModel(
            str(self.whisper_model_id), device=device,
            compute_type=compute_type, download_root=MODEL_WEIGHTS_DIR,
        )
        logger.info("faster-whisper loaded (%s, device=%s, %s)",
                    self.whisper_model_id, device, compute_type)

    @staticmethod
    def _load_audio_16k(source_file: str):
        """Load an audio file as 16 kHz mono float32 (via torchaudio, no whisperx)."""
        import numpy as np
        import torchaudio
        wav, sr = torchaudio.load(source_file)  # (channels, samples)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.squeeze(0).contiguous().numpy().astype(np.float32)

    def transcribe(self, source_file: str, num_speakers=None) -> Output:
        if num_speakers is not None and num_speakers < 1:
            raise ValueError("num_speakers must be positive")
        import numpy as np
        import torch

        audio = self._load_audio_16k(source_file)  # 16 kHz mono float32

        # 1. Transcribe with word timestamps.
        segments_gen, info = self.asr_model.transcribe(
            audio, language=self.language, word_timestamps=True,
            condition_on_previous_text=False,
        )
        words: list[dict] = []
        for seg in segments_gen:
            if not seg.words and seg.text.strip():
                words.append({"word": seg.text.strip(), "start": float(seg.start),
                              "end": float(seg.end)})
            for w in (seg.words or []):
                words.append({"word": w.word.strip(), "start": float(w.start),
                              "end": float(w.end)})
        detected_language = self.language or getattr(info, "language", None) or "en"

        # 2. VAD speech regions (segment boundaries).
        ts = []
        if len(audio):
            from silero_vad import get_speech_timestamps
            ts = get_speech_timestamps(
                torch.from_numpy(np.asarray(audio, dtype=np.float32)),
                self.vad_model, sampling_rate=16000,
                min_silence_duration_ms=int(PAUSE_THRESHOLD_SECONDS * 1000),
                return_seconds=True,
            )
        regions = [(t["start"], t["end"]) for t in ts]

        # 3. Diarize at word level, then group words by VAD region.
        audio_end = words[-1]["end"] if words else (len(audio) / 16000.0)
        transcript_result = {"segments": [
            {"text": " ".join(w["word"] for w in words), "start": 0.0,
             "end": audio_end, "words": words}]}
        diarized = False
        if words and not self.skip_diarization and num_speakers != 1 and self._embed_diarize:
            try:
                from langswap.ml.speech_to_text_service.embed_diarize import assign_word_speakers
                assign_word_speakers(words, audio, device=str(self.device), num_speakers=num_speakers)
                diarized = True
                n = len({w.get("speaker") for w in words})
                logger.info("embed diarize assigned %s speaker label(s)", n)
            except Exception as e:
                raise RuntimeError(
                    "Speaker diarization failed. Check the ECAPA/SpeechBrain installation "
                    "and cached model; use skip_diarization=True or num_speakers=1 "
                    "only for a confirmed single speaker."
                ) from e
        if not diarized:
            for w in words:
                w["speaker"] = "SPEAKER_00"

        words_ws = transcript_result["segments"][0]["words"]
        segments = _group_words_by_vad(words_ws, regions)

        full_text = " ".join(w["word"] for w in words_ws)
        final_response = {
            "detected_language": detected_language,
            "device": "cuda" if str(self.device).startswith("cuda") else self.device,
            "model": "vad-whisper",
            "transcription": full_text,
            "translation": "",
            "segments": segments,
        }
        result_data = {"status": "finished", "output": final_response}
        return cattrs.structure(result_data, TranscriptionData).output
