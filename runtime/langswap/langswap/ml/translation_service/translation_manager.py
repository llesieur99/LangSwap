from logging import getLogger
import os
import json
import hashlib
from attr import evolve
from langswap.pipeline_models.models import TranslatedTextedSegment, VideoTranslation
from langswap.file_repository import FileRepository
from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code


logger = getLogger(__name__)


def translation_fingerprint(segments, source_lang, target_lang, backend):
    identity = {
        "version": 1, "backend": backend,
        "source_language": map_language_to_code(source_lang),
        "target_language": map_language_to_code(target_lang),
        "segments": [{"text": s.text, "start": s.start, "end": s.end, "speaker": s.speaker} for s in segments],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class TranslationManager:
    public_id: str

    _file_repository: FileRepository

    def __init__(self, public_id: str, file_repository: FileRepository, device: str, logger, backend: str = "marian"):
        self.public_id = public_id
        self._file_repository = file_repository

        self.device = device
        self.logger = logger
        self.backend = backend
        if backend == "llamacpp":
            # Plain Gemma-4 GGUF via llama-cpp-python — the lightweight default:
            # tiny in-process engine, no second vLLM alongside OmniVoice.
            from langswap.model_pool import get_or_create
            from langswap.ml.translation_service.translator_llamacpp_client import LlamaCppTranslationClient
            self._translator_client = get_or_create(
                ("translate", "llamacpp", self.device),
                lambda: LlamaCppTranslationClient(self.device))
        elif backend == "openai":
            from langswap.ml.translation_service.translator_openai_client import OpenAITranslationClient
            self._translator_client = OpenAITranslationClient(device=self.device)
        elif backend == "marian":
            from langswap.ml.translation_service.translator_marian_client import MarianTranslationClient
            self._translator_client = MarianTranslationClient(device=self.device)
        else:
            raise ValueError(
                f"Unknown translation backend: {backend!r}. Use 'llamacpp', 'marian' or 'openai'.")

    def translate(self, video_translation: VideoTranslation, source_lang: str, target_lang: str) -> VideoTranslation:
        segments = video_translation.recognized_texts
        sentences_texts = [s.text for s in segments]
        source_lang = source_lang or video_translation.source_lang_code
        if not source_lang or not target_lang:
            raise ValueError("Source and target languages are required for translation.")
        self.logger.file_logger.info('Step: Translate the segments')
        fingerprint = translation_fingerprint(segments, source_lang, target_lang, self.backend)
        cache_path = os.path.join(self._file_repository.directory, "translations.json")
        metadata_path = os.path.join(self._file_repository.directory, "translations.cache.json")
        translations = None
        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            with open(cache_path, encoding="utf-8") as f:
                rows = json.load(f)
            if (metadata.get("fingerprint") == fingerprint and isinstance(rows, list)
                    and len(rows) == len(segments)
                    and all(isinstance(t, dict) and t.get("text") == s.text
                            and isinstance(t.get("translation"), str)
                            and (not s.text.strip() or t["translation"].strip())
                            for s, t in zip(segments, rows))):
                translations = [t["translation"] for t in rows]
        except (OSError, ValueError, TypeError, AttributeError):
            pass  # A missing/invalid cache is recomputed, never interpreted as a partial result.
        if translations is None:
            self._translator_client.load_models()
            translations = self._translator_client.translate(sentences=sentences_texts,
                                                         source_language=source_lang,
                                                         target_language=target_lang)
            if (not isinstance(translations, list) or len(translations) != len(segments)
                    or any(not isinstance(t, str) or (s.text.strip() and not t.strip())
                           for s, t in zip(segments, translations))):
                raise ValueError("Translation backend returned missing, empty or invalid translations.")
            json_segments = [{"translation": t, "text": s.text} for s, t in zip(segments, translations)]
            self.logger.log_json(file_name="translations.json", data=json_segments)
            self.logger.log_json(file_name="translations.cache.json", data={"fingerprint": fingerprint})
            for name in ("translations.json", "translations.cache.json"):
                self._file_repository.save_file(self._file_repository.get_file(name))
        translated_segments = [TranslatedTextedSegment(
            text=s.text, start=s.start, end=s.end, translation=t,
            source_file=None, generated_file=None, speaker=s.speaker,
        ) for s, t in zip(segments, translations)]
        return evolve(video_translation, translated_texts=translated_segments)
