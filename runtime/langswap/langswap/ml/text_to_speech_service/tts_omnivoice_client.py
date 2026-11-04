import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code

logger = logging.getLogger(__name__)

MODEL_ID = "k2-fsa/OmniVoice"
STAGE_INIT_TIMEOUT = 600


class OmniVoiceClient:
    """
    OmniVoice TTS client backed by the offline `vllm-omni` Python API.

    Supports auto voice and voice cloning without starting a server.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device
        self.sample_rate = 24000
        self.stage_config_path = str(
            Path(__file__).resolve().parent / "configs" / "omnivoice_vllm_stage.yaml"
        )
        self.stage_init_timeout = STAGE_INIT_TIMEOUT
        self.log_stats = False
        self.model = None
        self._refs = {}
        self._sampling_params_cls = None
        self._warned_duration_unsupported = False
        self.load_models()

    def _normalize_language(self, language: str | None) -> str | None:
        if not language:
            return None

        normalized = language.strip().lower()
        if len(normalized) <= 3:
            return normalized

        try:
            return map_language_to_code(normalized, system="whisper")
        except (AssertionError, KeyError, ValueError):
            return language

    def _load_reference_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio, int(sample_rate)

    def _extract_audio(self, outputs) -> tuple[np.ndarray, int]:
        for output in outputs:
            candidate_payloads = []

            request_output = getattr(output, "request_output", None)
            if request_output is not None:
                candidate_payloads.append(getattr(request_output, "multimodal_output", None))
                for sub_output in getattr(request_output, "outputs", []) or []:
                    candidate_payloads.append(getattr(sub_output, "multimodal_output", None))

            candidate_payloads.append(getattr(output, "multimodal_output", None))

            for payload in candidate_payloads:
                if not payload or "audio" not in payload:
                    continue

                audio = payload["audio"]
                sample_rate = int(payload.get("sr", self.sample_rate))

                if hasattr(audio, "detach"):
                    audio = audio.detach().float().cpu().numpy()
                else:
                    audio = np.asarray(audio)

                audio = np.asarray(audio, dtype=np.float32).squeeze()
                if audio.ndim > 1:
                    audio = audio[0]

                return audio, sample_rate

        raise RuntimeError("vllm-omni returned no audio payload for OmniVoice generation.")

    def load_models(self):
        try:
            from vllm_omni.entrypoints.omni import Omni
            from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `vllm-omni` required for OmniVoice TTS. "
                "Install it with `pip install vllm-omni`."
            ) from e

        if self.model is not None:
            return

        self.model = Omni(
            model=self.model_id,
            stage_configs_path=self.stage_config_path,
            trust_remote_code=True,
            log_stats=self.log_stats,
            stage_init_timeout=self.stage_init_timeout,
        )
        self._sampling_params_cls = OmniDiffusionSamplingParams

    def __enter__(self):
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.model is not None and hasattr(self.model, "close"):
            self.model.close()
        self.model = None

    def generate_audio(
        self,
        text: str,
        source_audio_file: str,
        source_text: str,
        save_path: str,
        language: str,
        duration: float | None = None,
    ):
        """Generate speech with OmniVoice through the offline `vllm-omni` API."""
        if self.model is None:
            self.load_models()

        multi_modal_data = {}
        mm_processor_kwargs = {}

        if source_audio_file:
            reference_audio, reference_sr = self._load_reference_audio(source_audio_file)
            multi_modal_data["audio"] = (reference_audio, reference_sr)
            mm_processor_kwargs["ref_text"] = (source_text or "").strip()
            mm_processor_kwargs["sample_rate"] = reference_sr

        normalized_language = self._normalize_language(language)
        if normalized_language:
            mm_processor_kwargs["lang"] = normalized_language

        if duration is not None:
            mm_processor_kwargs["duration"] = float(duration)
            if not self._warned_duration_unsupported:
                logger.warning(
                    "The current vllm-omni OmniVoice offline pipeline does not enforce explicit duration yet; "
                    "the downstream dubbing stage may still retime segments as needed."
                )
                self._warned_duration_unsupported = True

        prompt = {"prompt": text}
        if multi_modal_data:
            prompt["multi_modal_data"] = multi_modal_data
        if mm_processor_kwargs:
            prompt["mm_processor_kwargs"] = mm_processor_kwargs

        outputs = self.model.generate(
            prompt,
            sampling_params_list=[self._sampling_params_cls()],
            use_tqdm=False,
        )
        wav, sample_rate = self._extract_audio(outputs)
        self.sample_rate = sample_rate

        save_dir = os.path.dirname(save_path) or "."
        os.makedirs(save_dir, exist_ok=True)
        sf.write(save_path, wav, sample_rate)

    def prepare_speakers(self, video_translation, temp_folder):
        from langswap.ml.text_to_speech_service.speaker_voices import (
            isolated_reference_segments,
            validate_audio,
        )
        self._refs = {}
        for speaker, segments in isolated_reference_segments(video_translation.translated_texts).items():
            reference = max(segments, key=lambda item: item.end - item.start)
            validate_audio(reference.source_file)
            self._refs[speaker] = reference

    def tts_pipeline(self, video_translation, temp_folder, language="en"):
        from langswap.ml.text_to_speech_service.speaker_voices import (
            file_digest,
            speaker_label,
            synthesize_cached_segment,
        )
        self.prepare_speakers(video_translation, temp_folder)
        for segment in tqdm(video_translation.translated_texts, desc="OmniVoice generation.", leave=True):
            reference = self._refs[speaker_label(segment)]
            synthesize_cached_segment(
                segment, temp_folder, language, "omnivoice",
                {"audio": file_digest(reference.source_file), "transcript": reference.text},
                lambda path, segment=segment, reference=reference: self.generate_audio(
                    text=segment.translation, source_audio_file=reference.source_file,
                    source_text=reference.text, save_path=path, language=language,
                    duration=segment.end - segment.start,
                ),
            )
        return video_translation
