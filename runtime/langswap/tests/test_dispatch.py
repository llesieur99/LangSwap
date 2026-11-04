"""Backend-dispatch guards: each manager rejects an unknown backend string.

The unknown-backend path hits the dispatch else-branch before any heavy client
is constructed, so these run in the lean env without ASR/TTS deps installed.
"""
from unittest.mock import Mock

import pytest

from langswap.ml.speech_to_text_service import SpeechToTextManager
from langswap.ml.translation_service import TranslationManager
from langswap.ml.text_to_speech_service import TextToSpeechManager


def test_unknown_asr_backend_raises():
    with pytest.raises(ValueError):
        SpeechToTextManager(
            language="english",
            public_id="id",
            file_repository=object(),
            device="cpu",
            logger=object(),
            backend="nope",
        )


def test_unknown_translation_backend_raises():
    with pytest.raises(ValueError):
        TranslationManager(
            public_id="id",
            file_repository=object(),
            device="cpu",
            logger=object(),
            backend="nope",
        )


def test_unknown_tts_backend_raises():
    with pytest.raises(ValueError):
        TextToSpeechManager(
            public_id="id",
            file_repository=Mock(),
            tts_sample_rate=24000,
            logger=object(),
            device="cpu",
            tts_name="nope",
        )
