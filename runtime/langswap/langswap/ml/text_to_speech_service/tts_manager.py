import os
from logging import getLogger

from attr import evolve

from langswap.file_repository import FileRepository
from langswap.ml.text_to_speech_service.audio_dubbing_manager import AudioDubbingManager
from langswap.ml.text_to_speech_service.speaker_voices import (
    file_digest,
    speaker_label,
    synthesize_cached_segment,
    validate_segment_intervals,
)
from langswap.pipeline_models.models import TranslatedTextedSegment, VideoTranslation
from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code

logger = getLogger(__name__)


class TextToSpeechManager:
    public_id: str
    # pass it
    _file_repository: FileRepository
    tts_sample_rate: int
    audio_dubbing_manager: AudioDubbingManager

    def __init__(self,
                public_id: str,
                file_repository: FileRepository,
                tts_sample_rate: int,
                logger, device="cuda",
                tts_name="openvoice", speaker_voices=None):
        self.public_id = public_id
        self._file_repository = file_repository

        self.audio_dubbing_manager = AudioDubbingManager(file_repository, device=device)

        self._tts_client = None
        self.tts_name = tts_name
        self.speaker_voices = dict(speaker_voices or {})
        self.choose_tts_client(tts_name, file_repository, device)

        self.tts_sample_rate = self._tts_client.sample_rate

        self.logger = logger


    def clear_result_video(self, path: str):
        # TODO use file repositoty, for update resulted_video.mp4
        if os.path.exists(path):
            os.remove(path)

    def prepare_speakers(self, video_translation):
        validate_segment_intervals(video_translation.translated_texts)
        prepare = getattr(self._tts_client, "prepare_speakers", None)
        if prepare is not None:
            prepare(video_translation, self._file_repository.subdir("generated_audio"))

    def synthesize_segment(self, segment: TranslatedTextedSegment, target_lang: str, vocals_path: str):
        generated_audio_folder = self._file_repository.subdir("generated_audio")
        language = target_lang if len(target_lang) <= 3 else map_language_to_code(target_lang, "whisper")
        speaker = speaker_label(segment)
        source_audio, source_text = segment.source_file, segment.text
        kwargs = {}
        voice_key = None
        if self.tts_name in {"openvoice", "xtts", "chatterbox"}:
            source_audio = self._tts_client._refs.get(speaker)
            if not source_audio:
                raise ValueError(f"Prepare references before synthesizing speaker {speaker!r}")
            kwargs["speaker"] = speaker
            voice_key = file_digest(source_audio)
            if self.tts_name == "openvoice":
                voice_key = {"base": self._tts_client._base.voice, "reference": voice_key}
        elif self.tts_name == "omnivoice":
            reference = self._tts_client._refs.get(speaker)
            if reference is None:
                raise ValueError(f"Prepare references before synthesizing speaker {speaker!r}")
            source_audio, source_text = reference.source_file, reference.text
            voice_key = {"audio": file_digest(source_audio), "transcript": source_text}
        elif self.tts_name == "edge":
            kwargs["speaker"] = speaker
            voice_key = self._tts_client._voice_for(language, speaker)
        elif self.tts_name == "kokoro":
            voice_key = self._tts_client.voice
        elif self.tts_name == "elevenlabs":
            # This client manages remote voice creation/deletion around its batch.
            from types import SimpleNamespace
            self._tts_client.tts_pipeline(SimpleNamespace(translated_texts=[segment]),
                                          generated_audio_folder, language=language)
            return
        synthesize_cached_segment(
            segment, generated_audio_folder, language, self.tts_name, voice_key,
            lambda path: self._tts_client.generate_audio(
                text=segment.translation, source_audio_file=source_audio,
                source_text=source_text, save_path=path, language=language,
                duration=segment.end - segment.start, **kwargs,
            ),
        )


    def choose_tts_client(self, name: str, file_repository, device):
        if name == "omnivoice":
            from langswap.ml.text_to_speech_service.tts_omnivoice_client import (
                OmniVoiceClient,
            )
            from langswap.model_pool import get_or_create

            self._tts_client = get_or_create(
                ("tts", "omnivoice", device, self.public_id),
                lambda: OmniVoiceClient(device=device))
        elif name == "elevenlabs":
            from langswap.ml.text_to_speech_service.tts_eleven_client import (
                ElevenTTSClient,
            )

            self._tts_client = ElevenTTSClient()
        elif name == "edge":
            from langswap.ml.text_to_speech_service.tts_edge_client import EdgeTTSClient

            self._tts_client = EdgeTTSClient(device=device, speaker_voices=self.speaker_voices)
        elif name == "xtts":
            from langswap.ml.text_to_speech_service.tts_xtts_client import XttsClient
            from langswap.model_pool import get_or_create

            self._tts_client = get_or_create(
                ("tts", "xtts", device, self.public_id),
                lambda: XttsClient(device=device))
        elif name == "chatterbox":
            from langswap.ml.text_to_speech_service.tts_chatterbox_client import (
                ChatterboxClient,
            )
            from langswap.model_pool import get_or_create

            self._tts_client = get_or_create(
                ("tts", "chatterbox", device, self.public_id),
                lambda: ChatterboxClient(device=device))
        elif name == "kokoro":
            from langswap.ml.text_to_speech_service.tts_kokoro_client import (
                KokoroClient,
            )
            from langswap.model_pool import get_or_create

            self._tts_client = get_or_create(
                ("tts", "kokoro", device),
                lambda: KokoroClient(device=device))
        elif name == "openvoice":
            from langswap.ml.text_to_speech_service.tts_openvoice_client import (
                OpenVoiceClient,
            )
            from langswap.model_pool import get_or_create

            self._tts_client = get_or_create(
                ("tts", "openvoice", device, self.public_id),
                lambda: OpenVoiceClient(device=device))
        else:
            raise ValueError(
                f"Unknown TTS engine: {name!r}. Use 'openvoice', 'kokoro', 'chatterbox', 'xtts', 'edge'.")




    def synthesize(self, video_translation: VideoTranslation, source_lang: str, target_lang: str) -> VideoTranslation:
        validate_segment_intervals(video_translation.translated_texts)
        vocals_audio = video_translation.background_audio["vocals.wav"]

        db_manager = AudioDubbingManager(file_repository=self._file_repository)
        # Resample segment copies; the cached Demucs stem stays byte-identical.

        splitted_audio_folder = self._file_repository.subdir("splitted_audio")
        video_translation = db_manager.split_audio_seconds(video_translation,
                                            vocals_audio,
                                            splitted_audio_folder,
                                            sample_rate=self.tts_sample_rate,
                                            )
        self._file_repository.save_dir(self._file_repository.subdir('splitted_audio'))

        self.logger.file_logger.info("Step: text to speech basic pipeline")
        generated_audio_folder = self._file_repository.subdir("generated_audio")
        video_translation = self._tts_client.tts_pipeline(
                    video_translation,
                    generated_audio_folder,
                    language=target_lang)
        self._file_repository.save_dir(self._file_repository.subdir('generated_audio'))


        return evolve(video_translation)
