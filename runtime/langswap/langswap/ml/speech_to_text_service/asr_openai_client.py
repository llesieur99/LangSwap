import os
import logging
from pathlib import Path

import cattrs

from langswap.model_config import MODEL_WEIGHTS_DIR
from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code
from langswap.ml.speech_to_text_service.asr_types import (
    Output,
    TranscriptionData,
    _group_words_into_segments,
)

logger = logging.getLogger(__name__)


class OpenAIASRClient:
    """
    ASR client: OpenAI Whisper API for transcription + word timestamps,
    pyannote (via whisperx) for speaker diarization.

    Same constructor signature and transcribe() return type as the VAD backend.

    Requires OPENAI_API_KEY env var.
    Diarization requires pyannote model weights (same as other backends); they
    auto-download on first use when HF_TOKEN is set.
    Pass skip_diarization=True to skip speaker assignment (all words → SPEAKER_00).
    """

    def __init__(
        self,
        device: str,
        language: str,
        skip_diarization: bool = False,
        model: str = "whisper-1",
    ) -> None:
        self.device = device
        self.skip_diarization = skip_diarization
        self.model = model
        self._client = None
        self.diarize_model = None

        if language is not None:
            self.language = map_language_to_code(language, system="whisper")
        else:
            self.language = None

        models_base_dir = Path(MODEL_WEIGHTS_DIR)
        diarize_config = models_base_dir / "pyannote/pyannote_diarization_config.yaml"
        self.model_path_diarization = str(diarize_config.resolve())

        if not skip_diarization and not os.path.exists(self.model_path_diarization):
            raise FileNotFoundError(
                f"Diarization model not found at: {self.model_path_diarization}\n"
                "Please set HF_TOKEN — the pyannote diarization model downloads automatically on first use."
            )

        self.load_models()

    # ------------------------------------------------------------------
    # Context-manager protocol (mirrors the VAD backend)
    # ------------------------------------------------------------------

    def __enter__(self):
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._client = None
        self.diarize_model = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self):
        self._load_openai_client()
        if not self.skip_diarization:
            self._load_diarize_model()

    def _load_openai_client(self):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `openai`. Install with: pip install openai"
            ) from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY env var is not set. "
                "Set it in your .env file or shell environment."
            )
        self._client = OpenAI(api_key=api_key)

    def _load_diarize_model(self):
        try:
            from whisperx.diarize import DiarizationPipeline
        except ImportError:
            try:
                import whisperx
                DiarizationPipeline = whisperx.DiarizationPipeline  # type: ignore[attr-defined]
            except Exception as e:
                raise ImportError(
                    "pyannote diarization requires whisperx. Install: pip install whisperx"
                ) from e

        cwd = Path.cwd().resolve()
        try:
            os.chdir(Path(self.model_path_diarization).parent.parent.resolve())
            self.diarize_model = DiarizationPipeline(
                self.model_path_diarization, device=self.device
            )
        finally:
            os.chdir(cwd)

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(self, source_file: str, num_speakers=None) -> Output:
        if self._client is None:
            self.load_models()

        # 1. Transcribe via OpenAI Whisper API (word-level timestamps)
        with open(source_file, "rb") as f:
            response = self._client.audio.transcriptions.create(
                model=self.model,
                file=f,
                language=self.language,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

        # --- Detected language ---
        raw_lang = getattr(response, "language", None) or self.language or "en"
        try:
            detected_language = map_language_to_code(raw_lang.lower(), system="whisper")
        except (AssertionError, KeyError):
            detected_language = self.language or raw_lang

        # --- Build word list ---
        words: list[dict] = []
        api_words = getattr(response, "words", None)
        if api_words:
            for w in api_words:
                words.append(
                    {
                        "word": w.word,
                        "start": float(w.start),
                        "end": float(w.end),
                    }
                )

        if not words:
            # Fallback: segment-level timestamps when word-level is unavailable
            api_segments = getattr(response, "segments", None)
            if api_segments:
                for seg in api_segments:
                    words.append(
                        {
                            "word": seg.text.strip(),
                            "start": float(seg.start),
                            "end": float(seg.end),
                        }
                    )

        # 2. Build transcript_result dict in the shape whisperx expects
        full_text = getattr(response, "text", "") or " ".join(w["word"] for w in words)
        audio_end = words[-1]["end"] if words else 0.0
        transcript_result = {
            "segments": [
                {"text": full_text, "start": 0.0, "end": audio_end, "words": words}
            ]
        }

        # 3. Diarize + assign speakers per word
        if not self.skip_diarization and self.diarize_model is not None:
            import whisperx
            audio = whisperx.load_audio(source_file)
            diarize_df = self.diarize_model(audio, num_speakers=num_speakers)
            transcript_result = whisperx.assign_word_speakers(diarize_df, transcript_result)
        else:
            for seg in transcript_result["segments"]:
                seg["speaker"] = "SPEAKER_00"
                for w in seg["words"]:
                    w["speaker"] = "SPEAKER_00"

        # 4. Group words (with speaker info) into segments
        words_with_speakers = transcript_result["segments"][0]["words"]
        segments = _group_words_into_segments(words_with_speakers)
        full_text_out = " ".join(w["word"] for w in words_with_speakers)

        final_response = {
            "detected_language": detected_language,
            "device": "cpu",
            "model": f"openai/{self.model}",
            "transcription": full_text_out,
            "translation": "",
            "segments": segments,
        }
        result_data = {"status": "finished", "output": final_response}
        return cattrs.structure(result_data, TranscriptionData).output
