"""Resemble Chatterbox multilingual TTS + voice clone.

One segment at a time; ~4-6 GB VRAM. French = language_id=fr.
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


class ChatterboxClient:
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
        os.environ.setdefault("HF_HOME", MODEL_WEIGHTS_DIR)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", MODEL_WEIGHTS_DIR)
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        logger.info("Loading Chatterbox multilingual device=%s cache=%s",
                    self.device, MODEL_WEIGHTS_DIR)
        try:
            self._tts = ChatterboxMultilingualTTS.from_pretrained(
                device=self.device, t3_model="v3")
        except TypeError:
            self._tts = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        sr = getattr(self._tts, "sr", None)
        if sr:
            self.sample_rate = int(sr)

    def _chunk_text(self, text: str, max_chars: int = 200):
        return split_speech_text(text, max_chars)

    def _tts_one(self, text: str, ref: str, language: str, save_path: str):
        import torchaudio as ta
        kwargs = {"language_id": self._lang(language)}
        if ref and os.path.exists(ref):
            kwargs["audio_prompt_path"] = ref
        wav = self._tts.generate(text, **kwargs)
        ta.save(save_path, wav, self.sample_rate)

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
                self._tts_one(chunk, ref, language, part)
                validate_audio(part)
                parts.append(part)
            if len(parts) == 1:
                os.replace(parts[0], save_path)
            else:
                self._concat_wavs(parts, save_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def tts_pipeline(self, video_translation, temp_folder, language="fr"):
        self.prepare_speakers(video_translation, temp_folder)
        self.load_models()
        for segment in tqdm(video_translation.translated_texts, desc="Chatterbox clone", leave=True):
            speaker = speaker_label(segment)
            synthesize_cached_segment(
                segment, temp_folder, language, "chatterbox", file_digest(self._refs[speaker]),
                lambda path, segment=segment, speaker=speaker: self.generate_audio(
                    text=segment.translation, source_audio_file=self._refs[speaker],
                    source_text=segment.text, save_path=path, language=language,
                    duration=segment.end - segment.start, speaker=speaker,
                ),
            )
        return video_translation
