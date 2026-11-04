import os
import subprocess
import types
from contextlib import ExitStack

from tqdm import tqdm

from langswap.ml.text_to_speech_service.speaker_voices import (
    file_digest,
    isolated_reference_segments,
    speaker_label,
    synthesize_cached_segment,
    validate_audio,
    validate_segment_intervals,
)
from langswap.pipeline_models.models import TranslatedTextedSegment


class ElevenTTSClient:

    def __init__(self):
        from elevenlabs.client import ElevenLabs

        api_key = os.environ.get("ELEVEN_API_KEY")
        if not api_key:
            raise OSError("ELEVEN_API_KEY env var is not set.")
        self.client = ElevenLabs(api_key=api_key)
        self.sample_rate = 24000
        # If ELEVEN_VOICE_ID is set, skip cloning and use this voice directly.
        self._preset_voice_id = os.environ.get("ELEVEN_VOICE_ID")
        self._source_groups = {}

    def _make_voice_ref(self, voice_id: str):
        """Thin wrapper so callers can always do voice.voice_id."""
        v = types.SimpleNamespace()
        v.voice_id = voice_id
        return v

    def clone_voice(self, video_translation, voice_descr: str = "", voice_name=""):
        # If a preset voice is configured, use it without cloning.
        if self._preset_voice_id:
            return self._make_voice_ref(self._preset_voice_id), False

        segments = video_translation.translated_texts
        speakers = {speaker_label(segment) for segment in segments}
        if len(speakers) != 1:
            raise ValueError("ElevenLabs cloning requires exactly one speaker per reference")
        with ExitStack() as stack:
            file_handles = [stack.enter_context(open(segment.source_file, "rb"))
                            for segment in segments[:24]]
            try:
                voice = self.client.voices.ivc.create(
                    name=voice_name or "cloned_voice", description=voice_descr, files=file_handles,
                )
            except Exception as exc:
                raise RuntimeError("ElevenLabs voice clone failed; choose an explicit preset voice to retry") from exc

        return voice, True

    def generate_audio(self, text: str, voice, save_path: str, source_text=None):
        from elevenlabs import save

        temporary = save_path + ".mp3"
        try:
            audio = self.client.text_to_speech.convert(
                voice_id=voice.voice_id, text=text, output_format="mp3_44100_128",
            )
            save(audio, temporary)
            subprocess.check_call(
                ["ffmpeg", "-y", "-i", temporary, "-ar", "24000", "-ac", "1", save_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            validate_audio(save_path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def prepare_speakers(self, video_translation, temp_folder):
        validate_segment_intervals(video_translation.translated_texts)
        if self._preset_voice_id:
            self._source_groups = {}
            for segment in video_translation.translated_texts:
                self._source_groups.setdefault(speaker_label(segment), []).append(segment)
            return
        self._source_groups = isolated_reference_segments(video_translation.translated_texts)
        for segments in self._source_groups.values():
            for segment in segments:
                validate_audio(segment.source_file)

    def tts_pipeline(
        self, video_translation, temp_folder: str, language="en"
    ) -> list[TranslatedTextedSegment]:
        validate_segment_intervals(video_translation.translated_texts)
        if hasattr(video_translation, "public_id") or not self._source_groups:
            self.prepare_speakers(video_translation, temp_folder)
        grouped = {}
        for segment in video_translation.translated_texts:
            grouped.setdefault(speaker_label(segment), []).append(segment)
        for speaker, segments in grouped.items():
            sources = self._source_groups.get(speaker)
            if sources is None:
                raise ValueError(f"No ElevenLabs reference prepared for speaker {speaker!r}")
            voice_key = self._preset_voice_id or [file_digest(s.source_file) for s in sources[:24]]
            voice, cloned = self.clone_voice(types.SimpleNamespace(translated_texts=sources))
            try:
                for segment in tqdm(segments, desc="Voice generation pipeline.", leave=True):
                    synthesize_cached_segment(
                        segment, temp_folder, language, "elevenlabs", voice_key,
                        lambda path, segment=segment, voice=voice: self.generate_audio(segment.translation, voice=voice, save_path=path),
                    )
            finally:
                if cloned:
                    self.client.voices.delete(voice_id=voice.voice_id)
        return video_translation
