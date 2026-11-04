import os
import json
import hashlib
from typing import List, Dict
from logging import getLogger

from langswap.ml.ffmpeg import FFmpegClient
from langswap.file_repository import FileRepository
from langswap.pipeline_models.models import RemoteFile
from langswap.pipeline_models.models import TextedSegment, VideoTranslation

logger = getLogger(__name__)


def _audio_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SpeechToTextManager:

    def __init__(
        self,
        language,
        public_id: str,
        file_repository: FileRepository,
        device,
        logger,
        skip_diarization: bool = False,
        backend: str = "vad",
    ):
        """
        Initializes the SpeechToTextManager.

        Args:
            skip_diarization: If True, skips speaker diarization.
            backend: ASR backend to use.  "vad" (default) uses faster-whisper +
                     Silero VAD segmentation (no forced aligner, no per-language
                     model — local, lightest and fastest).  "openai" uses the
                     hosted OpenAI ASR API.
        """
        if backend not in ("vad", "openai"):
            raise ValueError(f"Unknown ASR backend: {backend!r}. Use 'vad' or 'openai'.")
        self.public_id = public_id
        self._backend = backend
        self._device = device
        self._language = language
        self._skip_diarization = skip_diarization
        self._asr_client = None
        self._file_repository = file_repository
        self.logger = logger

        self.audio_extensions = ["mp3", "wav", "MP3"]

    def _get_audio_file(self, video_translation: VideoTranslation) -> str:
        """Gets the audio file path, extracting it from video if necessary."""
        base, extension = os.path.splitext(video_translation.source_file.file_path)
        if extension.lower().lstrip('.') in self.audio_extensions: # Ensure comparison is case-insensitive and handles extensions like .MP3
            return video_translation.source_file.file_path
        else:
            audio_remote_file = self._extract_audio(video_translation.source_file.file_path)
            # Assuming _extract_audio returns a RemoteFile and its path needs to be saved/retrieved
            # If _extract_audio already saves and returns the path, this might simplify
            return self._file_repository.save_file(audio_remote_file).file_path

    def _separate_vocals_and_background(self, audio_file_path: str) -> Dict[str, str]:
        """Separates vocals from background audio using Demucs."""
        bg_dir = self._file_repository.subdir('background_files')
        self.logger.file_logger.info('Step: Demucs separation')
        from langswap.model_pool import get_or_create
        from langswap.ml.text_to_speech_service.demucs_client import DemucsClient
        demucs_client = get_or_create("demucs", lambda: DemucsClient(load_separator=False))
        background_paths = demucs_client.separate(audio_file_path, bg_dir)
        self._file_repository.save_dir(bg_dir)
        return {name: path for path, name in background_paths}

    def _ensure_asr_client(self):
        if self._asr_client is not None:
            return
        from langswap.model_pool import get_or_create
        if self._backend == "vad":
            from langswap.ml.speech_to_text_service.asr_vad_client import VADWhisperASR
            self._asr_client = get_or_create(
                ("asr", "vad", self._device, self._language, self._skip_diarization),
                lambda: VADWhisperASR(
                    device=self._device,
                    language=self._language,
                    skip_diarization=self._skip_diarization,
                ),
            )
        elif self._backend == "openai":
            from langswap.ml.speech_to_text_service.asr_openai_client import OpenAIASRClient
            self._asr_client = OpenAIASRClient(
                device=self._device,
                language=self._language,
                skip_diarization=self._skip_diarization,
            )
        else:
            raise ValueError(f"Unknown ASR backend: {self._backend!r}. Use 'vad' or 'openai'.")

    def _cache_matches(self, name: str, key: dict) -> bool:
        try:
            with open(os.path.join(self._file_repository.directory, name), encoding="utf-8") as cache:
                return json.load(cache) == key
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def _save_cache_key(self, name: str, key: dict):
        handle = self._file_repository.get_file(name)
        with open(handle.file_path, "w", encoding="utf-8") as cache:
            json.dump(key, cache, sort_keys=True)
        self._file_repository.save_file(handle)

    def _load_or_transcribe_audio(self, vocal_file_path: str, num_speakers: int = None,
                                  source_fingerprint: str = None) -> tuple[list, str]:
        """Transcribes audio or loads existing transcription from cache."""
        raw_transcribed_file_name = "raw_transcribed_info.json"
        lang_detect_file_name = "lang_detect_info.json"

        log_text_path = os.path.join(self._file_repository.directory, raw_transcribed_file_name)
        log_lang_path = os.path.join(self._file_repository.directory, lang_detect_file_name)
        cache_name = "asr_cache_key.json"
        cache_key = {
            "audio_sha256": source_fingerprint or _audio_fingerprint(vocal_file_path),
            "language": self._language, "backend": self._backend,
            "skip_diarization": self._skip_diarization, "num_speakers": num_speakers,
            "version": 4,
        }

        source_lang_code = None # Initialize to None
        json_segments = []

        if (self._cache_matches(cache_name, cache_key)
                and os.path.exists(log_text_path) and os.path.exists(log_lang_path)):
            self.logger.file_logger.info('Getting info from cached transcribed samples')
            with open(log_text_path, encoding="utf-8") as f_text, open(log_lang_path, encoding="utf-8") as f_lang:
                json_segments = json.load(f_text)
                detect_lang_data = json.load(f_lang)
                source_lang_code = detect_lang_data["detected_language"]
        else:
            self.logger.file_logger.info('Transcribing audio and caching results')
            self._ensure_asr_client()
            with self._asr_client as asr_client:
                transcription = asr_client.transcribe(vocal_file_path, num_speakers=num_speakers)
                source_lang_code = transcription.detected_language
                json_segments = [{"text": seg.text, "start": seg.start, "end": seg.end, "speaker": seg.speaker} for seg in transcription.segments]
                detect_lang_data = {"detected_language": source_lang_code}

                self.logger.log_json(file_name=lang_detect_file_name, data=detect_lang_data)
                self.logger.log_json(file_name=raw_transcribed_file_name, data=json_segments)

                # Save files to repository
                self._file_repository.save_file(self._file_repository.get_file(raw_transcribed_file_name))
                self._file_repository.save_file(self._file_repository.get_file(lang_detect_file_name))
                self._save_cache_key(cache_name, cache_key)

        return json_segments, source_lang_code

    def _load_or_remap_segments(self, raw_json_segments: list) -> List[TextedSegment]:
        """Remaps pauses in transcribed segments or loads existing remapped segments from cache."""
        remapped_file_name = "splitted_sentences_pauses.json"
        log_text_path = os.path.join(self._file_repository.directory, remapped_file_name)
        cache_name = "asr_remap_cache_key.json"
        serialized = json.dumps(raw_json_segments, sort_keys=True, ensure_ascii=False)
        cache_key = {"raw_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(), "version": 2}

        segments = []

        if self._cache_matches(cache_name, cache_key) and os.path.exists(log_text_path):
            self.logger.file_logger.info('Getting info from cached remapped segments')
            with open(log_text_path, encoding="utf-8") as f:
                loaded_json_segments = json.load(f)
            for seg_data in loaded_json_segments:
                segments.append(TextedSegment(**seg_data))
        else:
            self.logger.file_logger.info('Remapping segments and caching results')
            segments = self._remap_pauses(raw_json_segments)
            json_to_save = [{"text": seg.text, "start": seg.start, "end": seg.end, "speaker": seg.speaker} for seg in segments]
            self.logger.log_json(file_name=remapped_file_name, data=json_to_save)
            self._file_repository.save_file(self._file_repository.get_file(remapped_file_name))
            self._save_cache_key(cache_name, cache_key)

        return segments

    def extract_and_transcribe(self, video_translation: VideoTranslation, num_speakers: int = None) -> VideoTranslation:
        """
        Orchestrates the speech-to-text pipeline for a given video/audio.

        Extracts audio, separates vocals, transcribes speech, remaps pauses for segmentation,
        and handles caching of intermediate results.
        """
        print(f"Extracting and transcribing audio file: {video_translation.source_file.file_path}. Language configured at ASR client initialization: {self._language}")
        if num_speakers is not None:
            num_speakers = int(num_speakers)
            if num_speakers < 1:
                raise ValueError("num_speakers must be positive")

        import threading

        # 1. Get audio file (extract if video)
        audio_file_path = self._get_audio_file(video_translation)
        source_fingerprint = _audio_fingerprint(audio_file_path)

        # 2 & 3: Run Demucs and Whisper concurrently.
        # Demucs (PyTorch/GPU) runs in a thread while Whisper (CTranslate2/GPU) runs in the
        # main thread — both release the GIL during CUDA computation so they genuinely
        # overlap. Whisper transcribes the raw audio (pre-separation), which for typical
        # speech clips is identical to transcribing the demucs vocals stem. Demucs output
        # is still needed for: vocals.wav (TTS reference audio) and background stems
        # (drums/bass/other for re-mixing). This saves ~1s vs sequential execution.
        _bg_result: list = [None]
        _bg_error: list = [None]

        def _run_demucs() -> None:
            try:
                _bg_result[0] = self._separate_vocals_and_background(audio_file_path)
            except Exception as exc:
                _bg_error[0] = exc

        demucs_thread = threading.Thread(target=_run_demucs, daemon=True)
        demucs_thread.start()

        # Transcribe raw audio while Demucs runs in the background.
        try:
            raw_transcribed_segments, source_lang_code = self._load_or_transcribe_audio(
                audio_file_path, num_speakers, source_fingerprint)
        finally:
            # Do not leave a GPU worker running after an ASR error releases the pool.
            demucs_thread.join()
            # Warm reuse belongs to the pool, not to a long-lived manager.
            self._asr_client = None
        if _bg_error[0] is not None:
            raise _bg_error[0]
        background_files = _bg_result[0]

        # 4. Remap pauses in segments (or load from cache)
        final_segments = self._load_or_remap_segments(raw_transcribed_segments)

        # Construct the RemoteFile object for extracted_audio if needed by VideoTranslation
        # This part depends on whether VideoTranslation.extracted_audio expects a path or a RemoteFile object.
        # Assuming it expects a RemoteFile object and audio_file_path is the path to it.
        # We might need to adjust how `audio_file_path` is obtained or passed if it needs to be a RemoteFile object earlier.
        # For now, let's assume `_get_audio_file` returns a path, and we create a RemoteFile if the original was video.
        extracted_audio_remote_file = None
        _base, extension = os.path.splitext(video_translation.source_file.file_path)
        if not (extension.lower().lstrip('.') in self.audio_extensions):
             # If it was a video, audio_file_path is the path to the extracted audio.
             # We need to construct a RemoteFile object for it.
             # This assumes the extracted audio file name matches what _extract_audio uses.
             # This part might need refinement based on how RemoteFile objects are managed for extracted files.
            extracted_audio_remote_file = self._file_repository.get_file(os.path.basename(audio_file_path))


        return VideoTranslation(
            source_lang_code=source_lang_code,
            public_id=video_translation.public_id,
            source_file=video_translation.source_file,
            extracted_audio=extracted_audio_remote_file if extracted_audio_remote_file else video_translation.source_file, # Or adjust based on actual needs
            background_audio=background_files,
            recognized_texts=final_segments,
            processed_video=video_translation.processed_video,
        )

    def _extract_audio(self, video_file_path, audio_file_name='extracted_audio.wav') -> RemoteFile:
        """
        Extracts the audio track from a video file using FFmpeg.

        Saves the extracted audio to a new file managed by the file repository.
        """
        ffmpeg_client = FFmpegClient()
        output_file = self._file_repository.get_file(audio_file_name)
        # extract_audio raises if ffmpeg produced no output, so reaching here means
        # success; only surface ffmpeg's stderr when it actually wrote something.
        _out, err = ffmpeg_client.extract_audio(video_file_path,
                                                output_file.file_path,
                                                time_limit=None)
        if err:
            logger.debug("ffmpeg extract_audio stderr: %s", err)

        return output_file

    def _remap_pauses(self, entries: List[Dict], pause_threshold=0.25, max_length=5, min_length=3):
        """
        Merges raw transcribed segments into more coherent sentences or phrases.

        Combines segments based on pause duration, speaker consistency, and segment length.
        """
        # algorithm to merge sentences according to the threshold
        entries = [entry for entry in entries if any(c.isalnum() for c in entry['text'])]
        if not entries:
            return []
        pairs = []
        prev_end = entries[0]['end']
        start_idx = entries[0]['start']
        cur_pair = [entries[0]]
        cur_speaker = entries[0]['speaker']
        for cur_sample in entries[1:]:
            conditions = [
                # A linguistically meaningful silence ends the current segment so the
                # gap survives as an inter-segment pause: merge_timestamps_stretch_whole
                # reinserts silence only *between* segments, never inside one. The split
                # is intentionally NOT gated on min_length — a long pause after a short
                # utterance must still be preserved, otherwise the dub drifts ahead of
                # the source (this gating is what broke pause handling on Qwen ASR).
                (cur_sample['start'] - prev_end) >= pause_threshold,
                # we have collected max length of audio
                (cur_sample['end'] - start_idx > max_length),
                # speaker changed
                cur_speaker != cur_sample['speaker'],
                ]
            # если пауза достаточно длинная и при этом мы собрали уже около 5 секунд или уже собрали больше 15 секунд
            if any(conditions):
                pairs.append(
                    TextedSegment(
                    text=" ".join([s['text'] for s in cur_pair]),
                    start=start_idx,
                    end=prev_end,
                    speaker=cur_speaker
                    )
                )
                cur_pair = [cur_sample]
                prev_end = cur_sample['end']
                start_idx = cur_sample['start']

                cur_speaker = cur_sample['speaker']
            else:
                cur_pair.append(cur_sample)
                prev_end = max(prev_end, cur_sample['end'])

        if cur_pair:
            pairs.append(
                    TextedSegment(
                    text=" ".join([s['text'] for s in cur_pair]),
                    start=start_idx,
                    end=prev_end,
                    speaker=cur_speaker
                    )
                )
        # last sample use case - sometimes it is too short and should be better merged with a previous one
        last_sample = pairs[-1]
        if (len(pairs) > 1
                and last_sample.end - last_sample.start < min_length
                and pairs[-2].speaker == last_sample.speaker
                and last_sample.start - pairs[-2].end < 1.5
                and last_sample.end - pairs[-2].start <= max_length):
            pairs[-2].text = pairs[-2].text + " " + last_sample.text
            pairs[-2].end = last_sample.end
            pairs.pop()
        return pairs
