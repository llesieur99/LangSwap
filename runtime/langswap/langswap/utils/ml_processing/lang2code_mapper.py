import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _language_codes():
    with Path(__file__).with_name("language_codes.json").open(encoding="utf-8") as f:
        return json.load(f)["lang2code"]

def map_language_to_code(language, system="whisper"):
    if not isinstance(language, str) or not language.strip():
        raise ValueError("A non-empty language name or code is required.")
    language = language.strip().lower()
    language2code = _language_codes()

    if system == "reverse_from_whisper":
        # Given a whisper code (e.g. "ru"), return the lowercase language name (e.g. "russian")
        for lang_name, codes in language2code.items():
            if codes.get("whisper") == language:
                return lang_name
        return language  # fallback: return as-is

    language = language2code.get(language, {}).get("reverse_from_whisper", language)
    if language not in language2code:
        language = next((name for name, codes in language2code.items()
                         if codes.get("whisper") == language), language)
    try:
        return language2code[language][system]
    except KeyError as exc:
        raise ValueError(f"Unsupported language or language system: {language!r}, {system!r}.") from exc
