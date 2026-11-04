"""Fast unit tests for the Gemma-4-E2B isochrony translation loop.

No llama_cpp needed: we bypass __init__/load_models and inject a scripted
_generate, so these run in the lean test env.
"""
import pytest

from langswap.ml.translation_service.translator_llamacpp_client import (
    LlamaCppTranslationClient,
    _spoken_length,
)


def _client(scripted_outputs):
    """A client whose _generate returns scripted outputs and records prompts."""
    c = object.__new__(LlamaCppTranslationClient)
    c._llm = object()  # non-None so translate() doesn't raise
    c.prompts = []
    outs = list(scripted_outputs)

    def fake_generate(user_content):
        c.prompts.append(user_content)
        return outs.pop(0)

    c._generate = fake_generate
    return c


def test_spoken_length_counts_vowels_across_scripts():
    assert _spoken_length("hello world") == 3          # e, o, o
    assert _spoken_length("privet") == 2               # i, e
    assert _spoken_length("привет") == 2               # и, е (Cyrillic)
    assert _spoken_length("café") == 2                 # diacritic folded: a, e
    assert _spoken_length("你好") == 2                  # CJK fallback: 2 ideographs
    assert _spoken_length("   ") == 1                  # never zero


def test_single_shot_within_tolerance_does_not_iterate():
    # source has 4 vowels; first output also has 4 -> ratio 1.0 -> stop immediately.
    c = _client(["aaaa"])
    out = c._translate_one("aaaa", "English", "Russian")
    assert out == "aaaa"
    assert len(c.prompts) == 1  # only the initial generation ran


def test_loop_converges_to_target_length():
    # target 4 vowels; outputs grow 2 -> 3 -> 4, landing in band on the 3rd gen.
    c = _client(["aa", "aaa", "aaaa"])
    out = c._translate_one("aaaa", "English", "Russian")
    assert out == "aaaa"
    assert len(c.prompts) == 3


def test_returns_closest_to_one_when_never_in_band():
    # target 4; none within +-15%: ratios 0.5, 0.25, 1.5 -> closest is the first (0.5).
    c = _client(["aa", "a", "aaaaaa"])
    out = c._translate_one("aaaa", "English", "Russian")
    assert out == "aa"  # guard: never returns a wildly mismatched padded/truncated retry


def test_feedback_prompt_is_flat_only_latest_attempt_in_context():
    # target 8 vowels; outputs stay out of band so the loop runs every iteration.
    c = _client(["Q_one", "Q_two", "Q_three"])
    c._translate_one("aaaaaaaa", "English", "Russian")
    assert len(c.prompts) == 3
    # 1st feedback references the 1st attempt...
    assert "Q_one" in c.prompts[1]
    # ...but the 2nd feedback references ONLY the 2nd attempt, not the 1st —
    # intermediate history is dropped so the prompt stays flat.
    assert "Q_two" in c.prompts[2]
    assert "Q_one" not in c.prompts[2]


def test_translate_handles_empty_and_blank_lines():
    c = _client(["whatever"])  # should not be consumed for blank input
    out = c.translate(["", "   "], "English", "Russian")
    assert out == ["", ""]
    assert c.prompts == []  # no generation for empty lines
