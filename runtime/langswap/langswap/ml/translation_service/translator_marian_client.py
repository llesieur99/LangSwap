"""Local Marian / Opus-MT translation (transformers).

Fallback when llama.cpp is unavailable or incompatible with the host.
Helsinki-NLP models are small, CPU or CUDA, no llama.cpp.
"""
from __future__ import annotations

import logging
import os
from typing import List

from langswap.model_config import MODEL_WEIGHTS_DIR
from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code

logger = logging.getLogger(__name__)

_LANG = {
    "english": "en", "en": "en",
    "french": "fr", "fr": "fr",
    "spanish": "es", "es": "es",
    "german": "de", "de": "de",
    "italian": "it", "it": "it",
    "portuguese": "pt", "pt": "pt",
    "russian": "ru", "ru": "ru",
    "dutch": "nl", "nl": "nl",
    "polish": "pl", "pl": "pl",
    "japanese": "ja", "ja": "ja",
    "korean": "ko", "ko": "ko",
    "chinese": "zh", "zh": "zh",
    "arabic": "ar", "ar": "ar",
}


def _code(lang: str) -> str:
    l = (lang or "").strip().lower()
    if l in _LANG:
        return _LANG[l]
    return map_language_to_code(l, system="whisper")


class MarianTranslationClient:
    def __init__(self, device: str = "cuda"):
        self.device = "cuda" if (device or "").startswith("cuda") else "cpu"
        self._model = None
        self._tok = None
        self._src = None
        self._tgt = None

    def load_models(self):
        # Weights downloaded on first translate() once we know the pair.
        return

    def _ensure(self, src: str, tgt: str) -> None:
        if self._model is not None and self._src == src and self._tgt == tgt:
            return
        from transformers import MarianMTModel, MarianTokenizer

        name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        cache = os.path.join(MODEL_WEIGHTS_DIR, "marian")
        os.makedirs(cache, exist_ok=True)
        logger.info("Loading Marian %s device=%s cache=%s", name, self.device, cache)
        self._tok = MarianTokenizer.from_pretrained(name, cache_dir=cache)
        self._model = MarianMTModel.from_pretrained(name, cache_dir=cache)
        self._model.to(self.device)
        self._model.eval()
        self._src, self._tgt = src, tgt

    def translate(
        self,
        sentences: List[str],
        source_language: str,
        target_language: str,
    ) -> List[str]:
        src = _code(source_language)
        tgt = _code(target_language)
        if src == tgt:
            return list(sentences)
        if not any(s.strip() for s in sentences):
            return ["" for _ in sentences]
        self._ensure(src, tgt)

        import torch

        pieces = []
        positions = []
        for idx, sentence in enumerate(sentences):
            for piece in self._split_text(sentence):
                pieces.append(piece)
                positions.append(idx)
        translated = [[] for _ in sentences]
        batch_size = 8
        for i in range(0, len(pieces), batch_size):
            chunk = pieces[i : i + batch_size]
            enc = self._tok(chunk, return_tensors="pt", padding=True, truncation=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                gen = self._model.generate(**enc, max_new_tokens=511, num_beams=4)
            eos = self._model.config.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
            pad = self._model.config.pad_token_id
            for row in gen:
                tokens = [int(t) for t in row if int(t) != pad]
                if eos_ids and (not tokens or tokens[-1] not in eos_ids):
                    raise ValueError("Marian output reached its token limit without completing the translation.")
            text = self._tok.batch_decode(gen, skip_special_tokens=True)
            if len(text) != len(chunk) or any(not t.strip() for t in text):
                raise ValueError("Marian returned an incomplete translation batch.")
            for idx, value in zip(positions[i : i + batch_size], text):
                translated[idx].append(value.strip())
        out = [" ".join(row) for row in translated]
        logger.info("Marian translated %d sentences (%s -> %s)", len(out), src, tgt)
        return out

    def _split_text(self, text: str) -> List[str]:
        """Translate every word, without tokenizer truncation on long passages."""
        chunks, current = [], ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if len(self._tok.encode(candidate, add_special_tokens=False)) <= 224:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(self._tok.encode(word, add_special_tokens=False)) > 224:
                raise ValueError("A single source word exceeds Marian's supported input length.")
            current = word
        if current:
            chunks.append(current)
        return chunks
