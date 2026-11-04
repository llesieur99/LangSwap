"""Coqui XTTS-v2 local voice clone. Fits a 24 GB 3090 (~2-6 GB VRAM).

Reference audio: ~10 s clip from Demucs vocals.wav (XTTS prefers 6-15 s).
"""
from __future__ import annotations

import logging
import os
import shutil

from tqdm.auto import tqdm

from langswap.ml.text_to_speech_service.speaker_voices import (
    build_speaker_references,
    file_digest,
    speaker_label,
    split_speech_text,
    synthesize_cached_segment,
    validate_audio,
)
from langswap.model_config import MODEL_WEIGHTS_DIR

logger = logging.getLogger(__name__)

_LANG = {
    "french": "fr", "fr": "fr",
    "english": "en", "en": "en",
    "spanish": "es", "es": "es",
    "german": "de", "de": "de",
    "italian": "it", "it": "it",
    "portuguese": "pt", "pt": "pt",
}


class XttsClient:
    def __init__(self, device: str = "cuda"):
        self.device = "cuda" if (device or "").startswith("cuda") else "cpu"
        self.sample_rate = 24000
        self._tts = None
        self._refs = {}

    def _lang(self, language: str) -> str:
        l = (language or "fr").strip().lower()
        return _LANG.get(l, l[:2] if l else "fr")

    def prepare_speakers(self, video_translation, temp_folder):
        vocals = (video_translation.background_audio or {}).get("vocals.wav")
        self._refs = build_speaker_references(
            vocals, video_translation.translated_texts,
            os.path.join(os.path.dirname(temp_folder), "speaker_refs"),
        )

    def load_models(self):
        if self._tts is not None:
            return
        os.environ["COQUI_TOS_AGREED"] = "1"
        os.environ.setdefault("TTS_HOME", MODEL_WEIGHTS_DIR)
        os.environ.setdefault("HF_HOME", MODEL_WEIGHTS_DIR)
        from TTS.api import TTS

        logger.info("Loading XTTS-v2 device=%s cache=%s", self.device, MODEL_WEIGHTS_DIR)
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        if self.device == "cuda":
            self._tts = self._tts.to("cuda")

    def _chunk_text(self, text: str, max_chars: int = 200):
        return split_speech_text(text, max_chars)

    def _tts_one(self, text: str, ref: str, language: str, save_path: str):
        kwargs = {
            "text": text, "speaker_wav": ref,
            "language": self._lang(language), "file_path": save_path,
        }
        try:
            self._tts.tts_to_file(split_sentences=True, **kwargs)
        except TypeError:
            self._tts.tts_to_file(**kwargs)

    def generate_audio(
        self,
        text: str,
        source_audio_file: str,
        source_text: str,
        save_path: str,
        language: str,
        duration=None,
        speaker=None,
    ):
        self.load_models()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("Cannot synthesize empty text")
        ref = self._refs.get(speaker) if speaker is not None else source_audio_file
        if not ref:
            raise ValueError(f"No speaker reference for {speaker!r}")
        validate_audio(ref)
        tmp_dir = save_path + ".parts"
        os.makedirs(tmp_dir, exist_ok=True)
        parts = []
        try:
            for i, chunk in enumerate(chunks):
                part = os.path.join(tmp_dir, f"{i:03d}.wav")
                try:
                    self._tts_one(chunk, ref, language, part)
                except Exception as exc:
                    logger.warning("XTTS chunk %s failed (%s); retry shorter", i, exc)
                    if self.device == "cuda":
                        import torch
                        torch.cuda.empty_cache()
                    half = self._chunk_text(chunk, max_chars=110)
                    ok = True
                    subparts = []
                    for j, h in enumerate(half):
                        sp = os.path.join(tmp_dir, f"{i:03d}_{j:02d}.wav")
                        try:
                            self._tts_one(h, ref, language, sp)
                            subparts.append(sp)
                        except Exception:
                            logger.exception("XTTS subchunk failed")
                            ok = False
                            break
                    if not ok or not subparts:
                        raise RuntimeError("XTTS synthesis failed after shorter-chunk retry") from exc
                    elif len(subparts) == 1:
                        os.replace(subparts[0], part)
                    else:
                        self._concat_wavs(subparts, part)
                validate_audio(part)
                parts.append(part)
            if len(parts) == 1:
                os.replace(parts[0], save_path)
            else:
                self._concat_wavs(parts, save_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _concat_wavs(self, paths, out_path):
        import torch
        import torchaudio
        waves = []
        sr = self.sample_rate
        for p in paths:
            w, s = torchaudio.load(p)
            if s != sr:
                w = torchaudio.functional.resample(w, s, sr)
            if w.shape[0] > 1:
                w = w.mean(dim=0, keepdim=True)
            waves.append(w)
        torchaudio.save(out_path, torch.cat(waves, dim=1), sr)

    def tts_pipeline(self, video_translation, temp_folder, language="fr"):
        self.prepare_speakers(video_translation, temp_folder)
        self.load_models()
        for segment in tqdm(video_translation.translated_texts, desc="XTTS clone", leave=True):
            speaker = speaker_label(segment)
            synthesize_cached_segment(
                segment, temp_folder, language, "xtts", file_digest(self._refs[speaker]),
                lambda path, segment=segment, speaker=speaker: self.generate_audio(
                    text=segment.translation, source_audio_file=self._refs[speaker],
                    source_text=segment.text, save_path=path, language=language,
                    duration=segment.end - segment.start, speaker=speaker,
                ),
            )
        return video_translation
