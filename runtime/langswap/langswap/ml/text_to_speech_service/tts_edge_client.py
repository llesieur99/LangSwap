"""Microsoft Edge TTS (online, free). Fallback when OmniVoice/vLLM-Omni is broken.

No voice cloning. Default Quebec French: fr-CA-JeanNeural.
Override with LANGSWAP_EDGE_VOICE.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess

from tqdm.auto import tqdm

from langswap.ml.text_to_speech_service.speaker_voices import (
    speaker_label,
    synthesize_cached_segment,
    validate_audio,
    validate_segment_intervals,
)

logger = logging.getLogger(__name__)

_DEFAULT_VOICE = {
    "fr": "fr-CA-JeanNeural",
    "en": "en-US-GuyNeural",
    "es": "es-MX-JorgeNeural",
}


class EdgeTTSClient:
    def __init__(self, device: str = "cpu", speaker_voices=None):
        self.device = device
        self.speaker_voices = dict(speaker_voices or {})
        self.sample_rate = 24000
        self.voice = os.environ.get("LANGSWAP_EDGE_VOICE") or _DEFAULT_VOICE["fr"]

    def prepare_speakers(self, video_translation, temp_folder):
        validate_segment_intervals(video_translation.translated_texts)
        labels = {speaker_label(s) for s in video_translation.translated_texts}
        missing = labels - self.speaker_voices.keys()
        if len(labels) > 1 and missing and not os.environ.get("LANGSWAP_EDGE_VOICE"):
            raise ValueError("Choose an explicit Edge voice for every narrator with speaker_voices / --speaker-voice.")

    def _voice_for(self, language: str, speaker="SPEAKER_00") -> str:
        if speaker in self.speaker_voices:
            return self.speaker_voices[speaker]
        if os.environ.get("LANGSWAP_EDGE_VOICE"):
            return os.environ["LANGSWAP_EDGE_VOICE"]
        lang = (language or "fr").strip().lower()[:2]
        return _DEFAULT_VOICE.get(lang, _DEFAULT_VOICE["fr"])

    def generate_audio(
        self,
        text: str,
        source_audio_file: str,
        source_text: str,
        save_path: str,
        language: str,
        duration=None,
        speaker="SPEAKER_00",
    ):
        import edge_tts

        text = (text or "").strip()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        if not text:
            raise ValueError("Cannot synthesize empty text")
        voice = self._voice_for(language, speaker)
        tmp = save_path + ".mp3"

        async def _run():
            comm = edge_tts.Communicate(text, voice)
            await comm.save(tmp)

        try:
            asyncio.run(_run())
            subprocess.check_call(
                ["ffmpeg", "-y", "-i", tmp, "-ar", "24000", "-ac", "1", save_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            validate_audio(save_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def tts_pipeline(self, video_translation, temp_folder, language="fr"):
        self.prepare_speakers(video_translation, temp_folder)
        validate_segment_intervals(video_translation.translated_texts)
        for segment in tqdm(video_translation.translated_texts, desc="Edge TTS", leave=True):
            speaker = speaker_label(segment)
            synthesize_cached_segment(
                segment, temp_folder, language, "edge", self._voice_for(language, speaker),
                lambda path, segment=segment, speaker=speaker: self.generate_audio(
                    text=segment.translation, source_audio_file=segment.source_file,
                    source_text=segment.text, save_path=path, language=language,
                    duration=segment.end - segment.start, speaker=speaker,
                ),
            )
        return video_translation
