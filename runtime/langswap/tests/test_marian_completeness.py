"""Long sources are chunked rather than silently truncated by the tokenizer."""

from types import SimpleNamespace

import pytest
import torch

from langswap.ml.translation_service.translator_marian_client import (
    MarianTranslationClient,
    _code,
)


class FakeTokenizer:
    def __init__(self):
        self.inputs = []

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))

    def __call__(self, lines, **kwargs):
        assert kwargs["truncation"] is False
        self.inputs.extend(lines)
        self.batch = lines
        return {"input_ids": torch.ones((len(lines), 1), dtype=torch.long)}

    def batch_decode(self, values, **kwargs):
        return [f"FR {line}" for line in self.batch]


def make_client():
    client = MarianTranslationClient("cpu")
    client._tok = FakeTokenizer()
    client._src, client._tgt = "en", "fr"
    client._model = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=0, pad_token_id=-1)
    )
    client._model.generate = lambda input_ids, **kw: torch.tensor(
        [[1, 0] for _ in input_ids]
    )
    return client


def test_all_words_of_long_sentence_reach_model():
    client = make_client()
    source = " ".join(f"word{i}" for i in range(600))
    result = client.translate([source, "", "the ending"], "english", "french")
    assert " ".join(client._tok.inputs[:-1]) == source
    assert all(len(line.split()) <= 224 for line in client._tok.inputs)
    assert len(result) == 3 and result[1] == ""
    assert "word599" in result[0] and "the ending" in result[2]


def test_generation_token_limit_is_not_reported_as_complete_translation():
    client = make_client()
    client._model.generate = lambda **kw: torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="without completing"):
        client.translate(["some words"], "english", "french")


def test_unknown_language_cannot_be_guessed_from_first_letters():
    with pytest.raises(ValueError):
        _code("unknown language")
    assert _code("French") == "fr"


def test_language_mapper_works_outside_repository(monkeypatch, tmp_path):
    from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code

    monkeypatch.chdir(tmp_path)
    assert map_language_to_code("FR") == "fr"
    assert map_language_to_code("English") == "en"
