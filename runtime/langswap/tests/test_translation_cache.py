"""Translation completeness, cache identity and language propagation."""

import json
from unittest.mock import Mock

import pytest

from langswap.file_repository import LocalOnlyFileRepository
from langswap.ml.translation_service.translation_manager import TranslationManager
from langswap.pipeline_models.models import TextedSegment, VideoTranslation
from langswap.utils.logging import Logger


def make_manager(tmp_path):
    repo = LocalOnlyFileRepository("translation-test", str(tmp_path))
    manager = TranslationManager(
        "translation-test", repo, "cpu", Logger(repo.directory), "marian"
    )
    manager._translator_client = Mock()
    manager._translator_client.translate.return_value = [
        "Bonjour à tous.",
        "Voici la suite.",
    ]
    vt = VideoTranslation(
        public_id="translation-test",
        source_lang_code="en",
        recognized_texts=[
            TextedSegment("Hello everyone.", 0.0, 1.0, "A"),
            TextedSegment("Here is the rest.", 2.0, 3.0, "B"),
        ],
    )
    return manager, vt


def test_translation_cache_reuses_only_matching_job(tmp_path):
    manager, vt = make_manager(tmp_path)
    result = manager.translate(vt, None, "french")
    assert result.source_lang_code == "en"
    assert [s.speaker for s in result.translated_texts] == ["A", "B"]
    cached = manager.translate(vt, None, "french")
    assert [s.translation for s in cached.translated_texts] == [
        "Bonjour à tous.",
        "Voici la suite.",
    ]
    assert manager._translator_client.translate.call_count == 1


@pytest.mark.parametrize(
    "change", ["language", "source", "speaker", "timing", "count", "backend"]
)
def test_translation_cache_invalidates_changed_inputs(tmp_path, change):
    manager, vt = make_manager(tmp_path)
    manager.translate(vt, "english", "french")
    language = "french"
    if change == "language":
        language = "spanish"
    elif change == "source":
        vt.recognized_texts[0].text = "A different line."
    elif change == "speaker":
        vt.recognized_texts[0].speaker = "C"
    elif change == "timing":
        vt.recognized_texts[0].start = 0.25
    elif change == "backend":
        manager.backend = "other-tested-translator"
    else:
        vt.recognized_texts.append(TextedSegment("Final words.", 3.0, 4.0, "A"))
        manager._translator_client.translate.return_value.append("Derniers mots.")
    manager.translate(vt, "english", language)
    assert manager._translator_client.translate.call_count == 2


@pytest.mark.parametrize(
    "payload", ["[]", "{", '[{"text":"wrong", "translation":"stale"}]']
)
def test_partial_or_corrupted_cache_is_recomputed(tmp_path, payload):
    manager, vt = make_manager(tmp_path)
    manager.translate(vt, "english", "french")
    (tmp_path / "translation-test" / "translations.json").write_text(
        payload, encoding="utf-8"
    )
    result = manager.translate(vt, "english", "french")
    assert len(result.translated_texts) == 2
    assert manager._translator_client.translate.call_count == 2


@pytest.mark.parametrize(
    "translations", [[], ["Bonjour"], ["Bonjour", ""], [None, "Salut"], "one string"]
)
def test_backend_cannot_drop_or_silence_segments(tmp_path, translations):
    manager, vt = make_manager(tmp_path)
    manager._translator_client.translate.return_value = translations
    with pytest.raises(ValueError, match="missing, empty or invalid"):
        manager.translate(vt, "english", "french")
    assert not (tmp_path / "translation-test" / "translations.cache.json").exists()


def test_manual_translation_edit_can_be_retained_for_same_source(tmp_path):
    manager, vt = make_manager(tmp_path)
    manager.translate(vt, "english", "french")
    cache = tmp_path / "translation-test" / "translations.json"
    rows = json.loads(cache.read_text(encoding="utf-8"))
    rows[0]["translation"] = "Bonjour, tout le monde !"
    cache.write_text(json.dumps(rows), encoding="utf-8")
    assert (
        manager.translate(vt, "english", "french").translated_texts[0].translation
        == "Bonjour, tout le monde !"
    )
    assert manager._translator_client.translate.call_count == 1


def test_language_aliases_do_not_invalidate_manual_translation(tmp_path):
    manager, vt = make_manager(tmp_path)
    manager.translate(vt, "english", "french")
    manager.translate(vt, "en", "fr")
    assert manager._translator_client.translate.call_count == 1
