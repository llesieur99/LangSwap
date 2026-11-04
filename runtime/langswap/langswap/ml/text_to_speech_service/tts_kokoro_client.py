"""Kokoro-82M native French TTS (ff_siwis). No English-speaker clone.

Clone of an EN YouTuber makes FR hard to understand. Kokoro speaks French.
~2 GB VRAM. Needs espeak-ng in the image.
"""
from __future__ import annotations

import logging
import os
import shutil

from tqdm.auto import tqdm

from langswap.ml.text_to_speech_service.speaker_voices import (
    split_speech_text,
    synthesize_cached_segment,
    validate_audio,
    validate_segment_intervals,
)
from langswap.model_config import MODEL_WEIGHTS_DIR

logger = logging.getLogger(__name__)

# hexgrad Kokoro: lang_code f = French, only stock FR voice is ff_siwis (female).
_VOICE = os.environ.get("LANGSWAP_KOKORO_VOICE", "ff_siwis")


class KokoroClient:
    def __init__(self, device: str = "cuda"):
        self.device = "cuda" if (device or "").startswith("cuda") else "cpu"
        self.sample_rate = 24000
        self._pipe = None
        self.voice = _VOICE

    def load_models(self):
        if self._pipe is not None:
            return
        os.environ.setdefault("HF_HOME", MODEL_WEIGHTS_DIR)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", MODEL_WEIGHTS_DIR)
        from kokoro import KPipeline

        logger.info("Loading Kokoro French voice=%s cache=%s", _VOICE, MODEL_WEIGHTS_DIR)
        self._pipe = KPipeline(lang_code="f", device=self.device)

    def _chunk_text(self, text: str, max_chars: int = 280):
        return split_speech_text(text, max_chars)

    def _tts_one(self, text: str, save_path: str):
        import numpy as np
        import soundfile as sf
        pieces = []
        for _gs, _ps, audio in self._pipe(text, voice=self.voice):
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            pieces.append(np.asarray(audio).reshape(-1))
        if not pieces:
            raise RuntimeError("Kokoro returned no audio for non-empty text")
        wav = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        sf.write(save_path, wav, self.sample_rate)

    def generate_audio(
        self,
        text: str,
        source_audio_file: str,
        source_text: str,
        save_path: str,
        language: str,
        duration=None,
    ):
        if (language or "fr").strip().lower() not in {"fr", "french"}:
            raise ValueError("Kokoro backend supports French only")
        self.load_models()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("Cannot synthesize empty text")
        tmp_dir = save_path + ".parts"
        os.makedirs(tmp_dir, exist_ok=True)
        parts = []
        try:
            for i, chunk in enumerate(chunks):
                part = os.path.join(tmp_dir, f"{i:03d}.wav")
                self._tts_one(chunk, part)
                validate_audio(part)
                parts.append(part)
            if len(parts) == 1:
                os.replace(parts[0], save_path)
            else:
                self._concat_wavs(parts, save_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _concat_wavs(self, paths, out_path):
        import numpy as np
        import soundfile as sf
        waves = []
        for p in paths:
            w, sr = sf.read(p, dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            if sr != self.sample_rate:
                import torch
                import torchaudio
                t = torch.from_numpy(w).unsqueeze(0)
                t = torchaudio.functional.resample(t, sr, self.sample_rate)
                w = t.squeeze(0).numpy()
            waves.append(w)
        sf.write(out_path, np.concatenate(waves), self.sample_rate)

    def tts_pipeline(self, video_translation, temp_folder, language="fr"):
        validate_segment_intervals(video_translation.translated_texts)
        self.load_models()
        for segment in tqdm(video_translation.translated_texts, desc="Kokoro FR", leave=True):
            synthesize_cached_segment(
                segment, temp_folder, language, "kokoro", self.voice,
                lambda path, segment=segment: self.generate_audio(
                    text=segment.translation, source_audio_file=segment.source_file,
                    source_text=segment.text, save_path=path, language=language,
                    duration=segment.end - segment.start,
                ),
            )
        return video_translation
