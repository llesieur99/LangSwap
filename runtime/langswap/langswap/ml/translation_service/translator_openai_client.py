import os
from typing import List

from tqdm import tqdm

from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code


class OpenAITranslationClient:
    """
    Translation client using the OpenAI Chat Completions API. No local GPU needed
    — works on Mac.

    Same load_models() / translate() interface as the other translator
    clients.

    Requires OPENAI_API_KEY env var.
    """

    def __init__(
        self,
        device: str = "cpu",
        model: str = "gpt-4o-mini",
    ):
        self.device = device
        self.model = model
        self._client = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `openai`. Install with: pip install openai"
            ) from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY env var is not set. "
                "Set it in your .env file or shell environment."
            )
        self._client = OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def translate(
        self,
        sentences: List[str],
        source_language: str,
        target_language: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
    ) -> list[str]:
        if self._client is None:
            raise RuntimeError(
                "OpenAITranslationClient.load_models() must be called before translate()."
            )

        def _to_lang_name(lang: str) -> str:
            l = (lang or "").strip()
            try:
                return map_language_to_code(l.lower(), system="cohere")
            except Exception:
                return l.capitalize()

        source = _to_lang_name(source_language)
        target = _to_lang_name(target_language)
        system_prompt = (
            f"You are a professional translator. "
            f"Translate the user's text from {source} to {target}. "
            f"Return only the translation, no explanations or additional text."
        )

        translations: list[str] = []
        for sentence in tqdm(sentences, desc="Translating"):
            safe = (sentence or "").strip()
            if not safe:
                translations.append("")
                continue

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": safe},
                ],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            if response.choices[0].finish_reason == "length":
                raise ValueError("OpenAI translation exceeded its token limit; split the source passage.")
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("OpenAI returned an empty translation.")
            translations.append(content.strip())

        return translations
